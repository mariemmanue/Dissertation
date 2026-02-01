import sys
import os
import site
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split
from sklearn.metrics import accuracy_score, f1_score
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer, TrainerCallback

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

class MultitaskModel(transformers.PreTrainedModel):
    config_class = MultitaskModelConfig

    def __init__(self, encoder, taskmodels_dict, config):
        super().__init__(config)
        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    @classmethod
    def create(cls, model_name, head_type_list, freeze_mode="none", new_vocab_size=None):
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
            taskmodels_dict[task_name] = nn.Linear(hidden_size, 2)

        return cls(encoder=encoder, taskmodels_dict=taskmodels_dict, config=config)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = out.last_hidden_state[:, 0, :]
        logits_per_task = []
        for _, head in self.taskmodels_dict.items():
            logits_per_task.append(head(cls_repr))
        return torch.stack(logits_per_task, dim=1)

AutoConfig.register("multitask_model", MultitaskModel.config_class)
AutoModel.register(MultitaskModel.config_class, MultitaskModel)

class MultitaskTrainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]          # (B, T)
        logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )                                  # (B, T, 2)
        B, T, C = logits.shape
        logits_flat = logits.view(B * T, C)
        labels_flat = labels.view(B * T)
        loss = nn.CrossEntropyLoss()(logits_flat, labels_flat)
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
    print(">>> compute_metrics CALLED")
    logits, labels = eval_pred
    B, T, C = logits.shape
    logits_flat = logits.reshape(B * T, C)
    labels_flat = labels.reshape(B * T)
    preds = logits_flat.argmax(axis=1)
    return {"eval_f1": f1_score(labels_flat, preds, average="macro"), "eval_accuracy": accuracy_score(labels_flat, preds)}

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

    args = parser.parse_args()

    warmup = args.warmup
    max_len = args.max_length
    weight_decay = 0.01

    # WandB setup
    if use_wandb:
        wandb.init(project=args.wandb_project)
        cfg = wandb.config
        lr = getattr(cfg, "learning_rate", args.lr)
        bs = getattr(cfg, "batch_size", args.bs)
        epochs = getattr(cfg, "epochs", args.epochs)
    else:
        lr, bs, epochs = args.lr, args.bs, args.epochs

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
    
    # If using Auto-Unfreeze, we MUST start with 'all' or 'bottom_12' frozen
    freeze_mode = args.freeze_mode
    if args.auto_unfreeze_epoch > 0 and freeze_mode == "none":
        freeze_mode = "all" # Default to Linear Probe start if unspecified
        
    model = MultitaskModel.create(
        MODEL_NAME, 
        head_type_list, 
        freeze_mode=freeze_mode,
        new_vocab_size=new_vocab_size
    ).to(device)

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


    run_name = os.environ.get("WANDB_RUN_ID") or f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
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
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        weight_decay=weight_decay,
        remove_unused_columns=False,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        # load_best_model_at_end=True,
        # metric_for_best_model="eval_f1",
        greater_is_better=True,
        dataloader_drop_last=True,   # <- add just this line
    )


    trainer = MultitaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )

    trainer.train()

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

    metrics = trainer.evaluate()
    print(">>> eval metrics dict:", metrics)

    import shutil
    shutil.rmtree(f"./models/{out_dir}", ignore_errors=True)
    if use_wandb: 
        wandb.finish()
    