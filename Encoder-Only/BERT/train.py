"""

     

# CE full FT
nlprun -q jag -p standard -r 40G -c 2 \
  -n PUSHnovocab_modernbert_fullft_ce \
  -o slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation/Encoder-Only/BERT && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python train.py CGEdit AAE \
     --loss_type ce \
     --freeze_mode none \
     --auto_unfreeze_epoch 0 \
     --lr 1e-5 \
     --bs 32 \
     --epochs 200 \
     --warmup 500 \
     --wandb_project modernbert-fullft"

"""


import sys
import os
import site
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split
from sklearn.metrics import average_precision_score, f1_score, accuracy_score
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer, TrainerCallback
from sklearn.metrics import average_precision_score
import json, os
from multitask_modeling import MultitaskModel, MultitaskModelConfig

# --- Environment Setup ---
site_dirs = []
try:
    site_dirs.extend(site.getsitepackages())
except Exception:
    pass
site_dirs.append(site.getusersitepackages())
for d in reversed(site_dirs):
    if os.path.isdir(d):
        sys.path.insert(0, d)

print("USING TRANSFORMERS FROM:", transformers.__file__, file=sys.stderr)

# Try wandb
try:
    import wandb
    use_wandb = True
except ImportError:
    wandb = None
    use_wandb = False

# --- Callback for Plan B (Auto-Unfreeze) ---
class UnfreezeCallback(TrainerCallback):
    def __init__(self, unfreeze_epoch, model):
        self.unfreeze_epoch = unfreeze_epoch
        self.model = model
        self.already_unfrozen = False

    def on_epoch_begin(self, args, state, control, **kwargs):
        if state.epoch >= self.unfreeze_epoch and not self.already_unfrozen:
            print(f"\n{'='*40}\n>>> EPOCH {state.epoch}: UNFREEZING ENCODER (Plan B) <<<\n{'='*40}")
            # Unlock the encoder
            for param in self.model.encoder.parameters():
                param.requires_grad = True
            self.already_unfrozen = True

# --- Model Definitions ---
class MultitaskModelConfig(transformers.PretrainedConfig):
    model_type = "multitask_model"
    def __init__(self, base_model_name="answerdotai/ModernBERT-large", head_names=None, loss_type="ce", **kwargs):
        self.base_model_name = base_model_name
        self.head_names = head_names
        self.loss_type = loss_type
        super().__init__(**kwargs)


class MultitaskModel(transformers.PreTrainedModel):
    config_class = MultitaskModelConfig

    def __init__(self, encoder, taskmodels_dict, config):
        super().__init__(config)
        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    @classmethod
    def create(cls, model_name, head_type_list, freeze_mode="none", new_vocab_size=None, loss_type="ce"):
        class TaskHead(nn.Module):
            def __init__(self, hidden_size, dropout=0.1):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, 2),
                )
            def forward(self, x):
                return self.net(x)
            

        config = transformers.AutoConfig.from_pretrained(model_name)
        encoder = transformers.AutoModel.from_pretrained(model_name, config=config)
        
        # Resize embeddings if we added tokens
        if new_vocab_size is not None and new_vocab_size > config.vocab_size:
            print(f">>> Resizing embeddings to {new_vocab_size}")
            encoder.resize_token_embeddings(new_vocab_size)

        # --- FREEZING LOGIC ---
        if freeze_mode == "all":
            print(">>> FREEZE MODE: ALL (Linear Probe)")
            for param in encoder.parameters():
                param.requires_grad = False
                
        elif freeze_mode == "bottom_12":
            print(">>> FREEZE MODE: BOTTOM 12 (Stabilized)")
            # Freeze embeddings
            for name, param in encoder.named_parameters():
                if "embeddings" in name:
                    param.requires_grad = False
                # Freeze layers 0-11
                if "layers" in name:
                    try:
                        parts = name.split(".")
                        layer_idx = next((int(p) for p in parts if p.isdigit()), -1)
                        if layer_idx != -1 and layer_idx < 12:
                            param.requires_grad = False
                    except ValueError:
                        pass
        else:
            print(">>> FREEZE MODE: NONE (Masis Style / Full Finetune)")

        hidden_size = config.hidden_size
        taskmodels_dict = {}
        for task_name in head_type_list:
            if loss_type == "ce":
                # 2 logits: no vs yes
                taskmodels_dict[task_name] = nn.Linear(hidden_size, 2)
            else:  # "bce"
                # 1 logit: P(feature present)
                taskmodels_dict[task_name] = nn.Linear(hidden_size, 1)
        return cls(encoder=encoder, taskmodels_dict=taskmodels_dict, config=config)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = out.last_hidden_state[:, 0, :]
        logits_per_task = []
        for _, head in self.taskmodels_dict.items():
            logits_per_task.append(head(cls_repr))  # [B, 2] or [B, 1]
        logits = torch.stack(logits_per_task, dim=1)  # [B, T, C]

        # If C == 1, squeeze that last dim → [B, T]
        if logits.size(-1) == 1:
            logits = logits.squeeze(-1)
        return logits


AutoConfig.register("multitask_model", MultitaskModel.config_class)
AutoModel.register(MultitaskModel.config_class, MultitaskModel)

class MultitaskTrainer(transformers.Trainer):
    def __init__(self, *args, loss_type="ce", pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        if loss_type == "bce":
            self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]  # [B, T]
        logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )

        if self.loss_type == "ce":
            # logits: [B, T, 2], labels: [B, T] (0/1)
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            labels_flat = labels.view(B * T)
            loss = nn.CrossEntropyLoss()(logits_flat, labels_flat)
        else:
            # BCE: logits: [B, T], labels: [B, T] (0/1)
            loss = self.bce(logits, labels.float())

        if return_outputs:
            return (loss, logits)
        return loss


    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Shallow copy so we don't mutate original dict
        inputs = inputs.copy()
        labels = inputs.get("labels", None)

        model.eval()
        with torch.no_grad():
            loss, logits = self.compute_loss(
                model, inputs, return_outputs=True
            )

        # Detach tensors
        if loss is not None:
            loss = loss.detach()
        if logits is not None:
            logits = logits.detach()
        if labels is not None and isinstance(labels, torch.Tensor):
            labels = labels.detach()

        if prediction_loss_only:
            return loss, None, None

        return loss, logits, labels


class AAEFeatureDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "attention_mask": self.attention_mask[idx], "labels": self.labels[idx]}

def load_head_list(lang):
    if lang == "AAE":
        return ["zero-poss", "zero-copula", "double-tense", "be-construction", "resultant-done", "finna", "come", "double-modal", "multiple-neg", "neg-inversion", "n-inv-neg-concord", "aint", "zero-3sg-pres-s", "is-was-gen", "zero-pl-s", "double-object", "wh-qu"]
    elif lang == "IndE":
        return ["foc_self", "foc_only", "left_dis", "non_init_exis", "obj_front", "inv_tag", "cop_omis", "res_obj_pron", "res_sub_pron", "top_non_arg_con"]
    raise ValueError("Invalid lang")

def build_dataset(tokenizer, train_f, max_length=64):
    input_ids_list, attn_list, labels_list = [], [], []
    with open(train_f) as f:
        for i, line in enumerate(f):
            parts = line.strip().split("\t")
            if i == 0 and len(parts) > 1 and not parts[1].isdigit(): continue
            enc = tokenizer(parts[0], max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
            input_ids_list.append(enc["input_ids"].squeeze(0))
            attn_list.append(enc["attention_mask"].squeeze(0))
            labels_list.append(torch.tensor([int(x) for x in parts[1:]], dtype=torch.long))
    return AAEFeatureDataset(torch.stack(input_ids_list), torch.stack(attn_list), torch.stack(labels_list))

def compute_metrics(eval_pred):
    logits, labels = eval_pred  # numpy
    # logits: [B,T,2] (CE) or [B,T] (BCE)
    if logits.ndim == 3:
        # CE: probs1 from softmax
        B, T, C = logits.shape
        # stable softmax
        exps = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = exps / exps.sum(axis=-1, keepdims=True)  # [B,T,2]
        probs1 = probs[..., 1]  # [B,T]
    else:
        # BCE: sigmoid
        if logits.ndim == 3 and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        probs1 = 1 / (1 + np.exp(-logits))  # [B,T]

    B, T = probs1.shape
    y_true = labels.reshape(B * T)
    y_score = probs1.reshape(B * T)

    # Global AP and thresholded metrics (0.5)
    ap_global = average_precision_score(y_true, y_score)
    y_pred = (y_score >= 0.5).astype(int)
    f1 = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)

    # Optional: per-feature AP and Prec@K
    aps = []
    prec100s = []
    for t in range(T):
        yt = labels[:, t]
        ys = probs1[:, t]
        if yt.sum() == 0:
            continue  # skip all-negative feature in this split
        aps.append(average_precision_score(yt, ys))

        # Prec@100 style: sort by score, take top K, compute precision
        order = ys.argsort()[::-1]
        K = min(100, len(order))
        topk = order[:K]
        prec100s.append(yt[topk].mean())

    metrics = {
        "eval_ap_global": float(ap_global),
        "eval_f1": float(f1),
        "eval_accuracy": float(acc),
    }
    if aps:
        metrics["eval_ap_macro"] = float(np.mean(aps))
        metrics["eval_prec100_macro"] = float(np.mean(prec100s))
    return metrics

def find_best_thresholds(logits, labels, num_steps=21):
    # logits, labels: numpy arrays [N, T]
    import numpy as np
    from sklearn.metrics import f1_score

    N, T = logits.shape
    thresholds = np.zeros(T)
    for f in range(T):
        y_true = labels[:, f]
        z = logits[:, f]
        best_f1 = 0.0
        best_tau = 0.0
        # sweep over probabilities 0..1
        for p in np.linspace(0.0, 1.0, num_steps):
            tau = np.log(p / (1 - p)) if p > 0 and p < 1 else (float("inf") if p == 1 else -float("inf"))
            y_pred = (z > tau).astype(int)
            if y_true.sum() == 0:
                continue
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_tau = tau
        thresholds[f] = best_tau
    return thresholds

if __name__ == "__main__":
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser()
    parser.add_argument("gen_method", type=str)
    parser.add_argument("lang", type=str)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--wandb_project", type=str, default="cgedit-aae")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_mode", type=str, default="none", choices=["all", "bottom_12", "none"], help="Strategy for freezing layers")
    parser.add_argument("--auto_unfreeze_epoch", type=int, default=0, help="Epoch to unfreeze everything (Plan B)")
    parser.add_argument("--fix_vocab", action="store_true", help="Add dialect tokens to vocab")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a Trainer checkpoint to resume from",)

    parser.add_argument(
        "--loss_type",
        type=str,
        default="ce",
        choices=["ce", "bce"],
        help="Use CrossEntropy (ce) or BCEWithLogits (bce)")

    args = parser.parse_args()

    warmup = args.warmup
    max_len = args.max_length
    weight_decay = 0.01

    # WandB setup
    if use_wandb:
        run = wandb.init(project=args.wandb_project)
        cfg = wandb.config
        lr = getattr(cfg, "learning_rate", args.lr)
        bs = getattr(cfg, "batch_size", args.bs)
        epochs = getattr(cfg, "epochs", args.epochs)
        # use the W&B run id (or name) for matching
        run_name = run.id   # or run.name if you prefer
    else:
        lr, bs, epochs = args.lr, args.bs, args.epochs
        run_name = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


    MODEL_NAME = "answerdotai/ModernBERT-large"
    train_file = f"./data/{args.gen_method}/{args.lang}.tsv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Setup Tokenizer & Vocab
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    new_vocab_size = None
    if args.fix_vocab:
        # Crucial AAE terms
        new_tokens = ["aint", "finna", "tryna", "ion", "talmbout"]
        num_added = tokenizer.add_tokens(new_tokens)
        if num_added > 0:
            print(f">>> ADDED {num_added} TOKENS TO VOCAB: {new_tokens}")
            new_vocab_size = len(tokenizer)

    # 2. Create Model
    head_type_list = load_head_list(args.lang)
    
    mt_config = MultitaskModelConfig(
        base_model_name=MODEL_NAME,
        head_names=head_type_list,
        loss_type=args.loss_type,
    )

    # If using Auto-Unfreeze, we MUST start with 'all' or 'bottom_12' frozen
    freeze_mode = args.freeze_mode
    if args.auto_unfreeze_epoch > 0 and args.freeze_mode == "none":
        raise ValueError("auto_unfreeze_epoch>0 requires freeze_mode != 'none'")

        
    model = MultitaskModel.create(
    MODEL_NAME,
    head_type_list,
    freeze_mode=freeze_mode,
    new_vocab_size=new_vocab_size,
    loss_type=args.loss_type,
    ).to(device)

# : attach the multitask config so Trainer saves it
    model.config = mt_config

    # Optional but nice: make it auto‑class aware so HF writes auto_map
    mt_config.register_for_auto_class()
    MultitaskModel.register_for_auto_class("AutoModel")

    # Compile for speed
    try:
        model = torch.compile(model)
    except:
        pass

    

    # 3. Data & Training
    # dataset = build_dataset(tokenizer, train_file, max_length=max_len)
    dataset = build_dataset(tokenizer, train_file, max_length=128)
    print("Total examples:", len(dataset))
    print("Example[0] keys:", dataset[0].keys())
    print("input_ids shape:", dataset[0]["input_ids"].shape)
    print("labels shape:", dataset[0]["labels"].shape)
    print("First label vector:", dataset[0]["labels"])

    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print("Lengths:", len(dataset), len(train_ds), len(val_ds))


    out_dir = f"ModernBERT_{args.gen_method}_{args.lang}_{run_name}"

    callbacks = []
    if args.auto_unfreeze_epoch > 0:
        callbacks.append(UnfreezeCallback(args.auto_unfreeze_epoch, model))

    training_args = transformers.TrainingArguments(
        output_dir="./models/" + out_dir,
        overwrite_output_dir=False,
        report_to="wandb",
        run_name="modernbert-sweep",
        learning_rate=lr,
        do_train=True,
        do_eval=True,
        optim="adamw_torch_fused",
        seed=args.seed,
        lr_scheduler_type="linear",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=warmup,
        num_train_epochs=epochs,
        push_to_hub=True, 
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        weight_decay=weight_decay,
        remove_unused_columns=False,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_ap_macro",  # or eval_ap_global
        greater_is_better=True, 
        dataloader_drop_last=True,   # <- add just this line
    )
    pos_weight = None
    if args.loss_type == "bce":
        all_labels = dataset.labels  # [N, T]
        pos_counts = all_labels.sum(dim=0)
        neg_counts = all_labels.shape[0] - pos_counts
        eps = 1e-6
        pos_weight = (neg_counts / (pos_counts + eps)).to(device)
        max_w = 20.0
        pos_weight = torch.clamp(pos_weight, max=max_w)



    trainer = MultitaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        loss_type=args.loss_type,
        pos_weight=pos_weight,
    )

    # trainer.train()
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    extra_cfg = {"loss_type": args.loss_type}
    with open(f"./models/{out_dir}/extra_config.json", "w") as f:
        json.dump(extra_cfg, f)

    if args.loss_type == "bce":
        preds_output = trainer.predict(val_ds)
        val_logits = preds_output.predictions   # [N, T]
        val_labels = preds_output.label_ids     # [N, T]

        thresholds = find_best_thresholds(val_logits, val_labels, num_steps=21)
        import numpy as np, os
        os.makedirs(f"./models/{out_dir}", exist_ok=True)
        np.save(f"./models/{out_dir}/thresholds.npy", thresholds)


    if use_wandb:
        repo_name = f"modernbert-aae-{args.gen_method}-{args.lang}-{wandb.run.name}"
    else:
        repo_name = f"modernbert-aae-{args.gen_method}-{args.lang}-local"
        
    trainer.save_model(f"./models/{out_dir}")
    tokenizer.save_pretrained(f"./models/{out_dir}")
    
    try:
        trainer.push_to_hub(repo_name)
    except Exception as e:
        print(f"Skipping hub push: {e}")
    # Add this block after trainer.push_to_hub(repo_name)
    try:
        trainer.create_model_card(model_name=repo_name)
        # Manually upload the extra config file
        if use_wandb:
            # Use huggingface_hub API directly since Trainer won't push arbitrary JSONs
            from huggingface_hub import upload_file
            upload_file(
                path_or_fileobj=f"./models/{out_dir}/extra_config.json",
                path_in_repo="extra_config.json",
                repo_id=f"{trainer.args.hub_model_id}", # Or construct your user/repo string
            )
            # Also upload thresholds.npy if it exists
            if os.path.exists(f"./models/{out_dir}/thresholds.npy"):
                upload_file(
                    path_or_fileobj=f"./models/{out_dir}/thresholds.npy",
                    path_in_repo="thresholds.npy",
                    repo_id=f"{trainer.args.hub_model_id}",
                )
    except Exception as e:
        print(f"Skipping extra file push: {e}")

    metrics = trainer.evaluate()
    print(">>> eval metrics dict:", metrics)

    import shutil
    shutil.rmtree(f"./models/{out_dir}", ignore_errors=True)
    if use_wandb: 
        wandb.finish()
    