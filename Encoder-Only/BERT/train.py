import sys
import os
import site
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split
from sklearn.metrics import accuracy_score, f1_score
import transformers
from datetime import datetime 
from transformers import AutoConfig, AutoModel, AutoTokenizer

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
print("TRANSFORMERS VERSION:", transformers.__version__, file=sys.stderr)

# Try wandb
try:
    import wandb
    use_wandb = True
except ImportError:
    wandb = None
    use_wandb = False


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
    def create(cls, model_name, head_type_list, freeze_layers=0, freeze_encoder=False):
        config = transformers.AutoConfig.from_pretrained(model_name)
        encoder = transformers.AutoModel.from_pretrained(model_name, config=config)
        hidden_size = config.hidden_size

        # LOGIC: "Freeze Encoder" overrides "Freeze Layers"
        if freeze_encoder:
            print(">>> LOCKING ENTIRE ENCODER (Linear Probe Mode)")
            for param in encoder.parameters():
                param.requires_grad = False
        
        elif freeze_layers > 0:
            print(f">>> Freezing bottom {freeze_layers} layers")
            # Freeze embeddings
            for name, param in encoder.named_parameters():
                if "embeddings" in name:
                    param.requires_grad = False
                
                # Freeze bottom N layers (ModernBERT layers are usually named 'layers.0', etc.)
                if "layers" in name:
                    try:
                        parts = name.split(".")
                        layer_idx = next((int(p) for p in parts if p.isdigit()), -1)
                        if layer_idx != -1 and layer_idx < freeze_layers:
                            param.requires_grad = False
                    except ValueError:
                        pass

        taskmodels_dict = {}
        for task_name in head_type_list:
            taskmodels_dict[task_name] = nn.Linear(hidden_size, 2)

        return cls(encoder=encoder, taskmodels_dict=taskmodels_dict, config=config)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use CLS token representation (Index 0)
        cls_repr = out.last_hidden_state[:, 0, :]  # (B, H)
        
        logits_per_task = []
        for _, head in self.taskmodels_dict.items():
            logits_per_task.append(head(cls_repr))  # (B, 2)
            
        # Stack into (B, num_tasks, 2)
        logits = torch.stack(logits_per_task, dim=1)
        return logits


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
        inputs = inputs.copy()
        labels = inputs.get("labels", None)

        model.eval()
        with torch.no_grad():
            loss, logits = self.compute_loss(model, inputs, return_outputs=True)

        if loss is not None: loss = loss.detach()
        if logits is not None: logits = logits.detach()
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

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def load_head_list(lang: str):
    if lang == "AAE":
        return ["zero-poss", "zero-copula", "double-tense", "be-construction", "resultant-done", "finna", "come", "double-modal", "multiple-neg", "neg-inversion", "n-inv-neg-concord", "aint", "zero-3sg-pres-s", "is-was-gen", "zero-pl-s", "double-object", "wh-qu"]
    elif lang == "IndE":
        return ["foc_self", "foc_only", "left_dis", "non_init_exis", "obj_front", "inv_tag", "cop_omis", "res_obj_pron", "res_sub_pron", "top_non_arg_con"]
    raise ValueError("lang must be 'AAE' or 'IndE'")


def build_dataset(tokenizer, train_f, max_length=64):
    input_ids_list = []
    attn_list = []
    labels_list = []

    with open(train_f) as f:
        for i, line in enumerate(f):
            parts = line.strip().split("\t")
            if i == 0 and len(parts) > 1 and not parts[1].isdigit(): continue
            
            text = parts[0]
            enc = tokenizer(
                text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids_list.append(enc["input_ids"].squeeze(0))
            attn_list.append(enc["attention_mask"].squeeze(0))
            labels_list.append(torch.tensor([int(x) for x in parts[1:]], dtype=torch.long))

    return AAEFeatureDataset(torch.stack(input_ids_list), torch.stack(attn_list), torch.stack(labels_list))


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    B, T, C = logits.shape
    logits_flat = logits.reshape(B * T, C)
    labels_flat = labels.reshape(B * T)
    preds = logits_flat.argmax(axis=1)
    return {
        "eval_f1": f1_score(labels_flat, preds, average="macro"),
        "eval_accuracy": accuracy_score(labels_flat, preds)
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("gen_method", type=str)
    parser.add_argument("lang", type=str)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--wandb_project", type=str, default="cgedit-aae")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimizer", type=str, default="adamw_torch_fused")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--freeze_layers", type=int, default=12)
    parser.add_argument("--freeze_encoder", action="store_true")
    args = parser.parse_args()

    if use_wandb:
        wandb.init(project=args.wandb_project)
        cfg = wandb.config
        lr = getattr(cfg, "learning_rate", args.lr)
        bs = getattr(cfg, "batch_size", args.bs)
        epochs = getattr(cfg, "epochs", args.epochs)
        warmup = getattr(cfg, "warmup_steps", args.warmup)
        max_len = getattr(cfg, "max_length", args.max_length)
        weight_decay = getattr(cfg, "weight_decay", 0.01)
    else:
        lr, bs, epochs, warmup, max_len = args.lr, args.bs, args.epochs, args.warmup, args.max_length
        weight_decay = 0.01

    MODEL_NAME = "answerdotai/ModernBERT-large"
    train_file = f"./data/{args.gen_method}/{args.lang}.tsv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Correctly passes the freeze_encoder flag
    head_type_list = load_head_list(args.lang)
    model = MultitaskModel.create(
        MODEL_NAME, 
        head_type_list, 
        freeze_layers=args.freeze_layers,
        freeze_encoder=args.freeze_encoder
    ).to(device)

    dataset = build_dataset(tokenizer, train_file, max_length=max_len)
    val_size = max(1, int(0.1 * len(dataset)))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])

    run_name = os.environ.get("WANDB_RUN_ID") or os.environ.get("WANDB_RUN_NAME")
    if not run_name:
        # Append timestamp to ensure unique directory for local runs
        run_name = f"no-wandb-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    out_dir = f"{MODEL_NAME.replace('/', '_')}_{args.gen_method}_{args.lang}_{run_name}"

    training_args = transformers.TrainingArguments(
        output_dir="./models/" + out_dir,
        overwrite_output_dir=True,
        report_to="wandb" if use_wandb else "none",
        run_name="modernbert-sweep",
        learning_rate=lr,
        do_train=True,
        do_eval=True,
        seed=args.seed,
        optim=args.optimizer,
        lr_scheduler_type=args.lr_scheduler_type,
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
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        torch_compile=True,  # <--- Best practice for compilation
    )

    trainer = MultitaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
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
    if use_wandb: wandb.finish()
