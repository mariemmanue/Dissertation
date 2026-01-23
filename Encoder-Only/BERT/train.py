import sys, os, site

# make sure we look in the env's site-packages first
site_dirs = []
try:
    site_dirs.extend(site.getsitepackages())
except Exception:
    pass
site_dirs.append(site.getusersitepackages())

for d in reversed(site_dirs):
    if os.path.isdir(d):
        sys.path.insert(0, d)

import transformers
print("USING TRANSFORMERS FROM:", transformers.__file__, file=sys.stderr)
print("TRANSFORMERS VERSION:", transformers.__version__, file=sys.stderr)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import PreTrainedConfig, AutoConfig, AutoModel

# wandb (keep as-is)
use_wandb = False
try:
    import wandb
    use_wandb = True
except ImportError:
    pass

class MultitaskConfig(PreTrainedConfig):
    model_type = "multitask"
    def __init__(self, head_type_list=None, **kwargs):
        super().__init__(**kwargs)
        self.head_type_list = head_type_list or []

class MultitaskModel(transformers.PreTrainedModel):
    config_class = MultitaskConfig
    def __init__(self, encoder, taskmodels_dict, config):
        super().__init__(config)
        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    @classmethod
    def create(cls, model_name, head_type_list):
        config = MultitaskConfig.from_pretrained(model_name, head_type_list=head_type_list)
        encoder = transformers.AutoModel.from_pretrained(model_name, config=config)
        hidden_size = config.hidden_size
        taskmodels_dict = {task: nn.Linear(hidden_size, 2) for task in head_type_list}
        return cls(encoder=encoder, taskmodels_dict=taskmodels_dict, config=config)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = out.last_hidden_state[:, 0, :]
        logits_per_task = [head(cls_repr) for head in self.taskmodels_dict.values()]
        return torch.stack(logits_per_task, dim=1)  # (B, num_tasks, 2)

# Register for HF save/load
AutoConfig.register("multitask", MultitaskConfig)
AutoModel.register(MultitaskConfig, MultitaskModel)

class MultitaskTrainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["labels"]  # (B, num_tasks)
        logits = model(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"))  # (B, num_tasks, 2)
        loss = nn.CrossEntropyLoss()(logits.view(-1, 2), labels.view(-1))
        return (loss, logits) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = inputs.copy()
        labels = inputs.pop("labels", None)
        model.eval()
        with torch.no_grad():
            loss, logits = self.compute_loss(model, inputs, return_outputs=True)
        loss = loss.detach() if loss is not None else None
        logits = logits.detach() if logits is not None else None
        labels = labels.detach() if labels is not None else None
        if prediction_loss_only:
            return (loss, None, None)
        return loss, logits, labels

class AAEFeatureDataset(Dataset):  # unchanged
    def __init__(self, input_ids, attention_mask, labels):  # labels now (N, num_tasks)
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "attention_mask": self.attention_mask[idx], "labels": self.labels[idx]}

def load_head_list(lang: str):
    if lang == "AAE":
        return [
            "zero-poss",
            "zero-copula",
            "double-tense",
            "be-construction",
            "resultant-done",
            "finna",
            "come",
            "double-modal",
            "multiple-neg",
            "neg-inversion",
            "n-inv-neg-concord",
            "aint",
            "zero-3sg-pres-s",
            "is-was-gen",
            "zero-pl-s",
            "double-object",
            "wh-qu",
        ]
    elif lang == "IndE":
        return [
            "foc_self",
            "foc_only",
            "left_dis",
            "non_init_exis",
            "obj_front",
            "inv_tag",
            "cop_omis",
            "res_obj_pron",
            "res_sub_pron",
            "top_non_arg_con",
        ]
    else:
        raise ValueError("lang must be 'AAE' or 'IndE'")
    
def build_dataset(tokenizer, train_f, max_length=64):
    input_ids_list = []
    attn_list = []
    labels_list = []

    with open(train_f) as f:
        for i, line in enumerate(f):
            parts = line.strip().split("\t")
            # skip header
            if i == 0 and len(parts) > 1 and not parts[1].isdigit():
                continue
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

    input_ids = torch.stack(input_ids_list)
    attention_mask = torch.stack(attn_list)
    labels = torch.stack(labels_list)

    return AAEFeatureDataset(input_ids, attention_mask, labels)

def compute_metrics(eval_pred):
    logits, labels = eval_pred  # logits (TotalB, num_tasks, 2), labels (TotalB, num_tasks)
    preds = logits.argmax(-1).flatten()
    labels_flat = labels.flatten()
    return {
        "eval_f1": f1_score(labels_flat, preds, average="macro"),
        "eval_accuracy": accuracy_score(labels_flat, preds)
    }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("gen_method", type=str)
    parser.add_argument("lang", type=str)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--wandb_project", type=str, default="cgedit-aae")
    parser.add_argument("--encoder", type=str, default="modernbert",
                    choices=["modernbert", "neobert", "roberta", "bert"])
    args = parser.parse_args()

    # --- pull from wandb if running under a sweep ---
    if use_wandb:
        wandb.init(project=args.wandb_project)
        cfg = wandb.config
        lr = getattr(cfg, "learning_rate", args.lr)
        bs = getattr(cfg, "batch_size", args.bs)
        epochs = getattr(cfg, "epochs", args.epochs)
        warmup = getattr(cfg, "warmup_steps", args.warmup)
        max_len = getattr(cfg, "max_length", args.max_length)
        weight_decay = getattr(cfg, "weight_decay", 0.0)
    else:
        lr = args.lr
        bs = args.bs
        epochs = args.epochs
        warmup = args.warmup
        max_len = args.max_length
        weight_decay = 0.0

    ENCODER_MAP = {
        "modernbert": "answerdotai/ModernBERT-large",
        "neobert": "sage-fc/neoBERT-large",
        "roberta": "roberta-large",
        "bert": "bert-large-uncased",
    }
    MODEL_NAME = ENCODER_MAP[args.encoder]

    train_file = f"./data/{args.gen_method}/{args.lang}.tsv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    head_type_list = load_head_list(args.lang)
    model = MultitaskModel.create(MODEL_NAME, head_type_list).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = build_dataset(tokenizer, train_file, max_length=args.max_length)

    val_size = max(1, int(0.1 * len(dataset)))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])

    run_name = os.environ.get("WANDB_RUN_ID") or os.environ.get("WANDB_RUN_NAME") or "no-wandb"
    out_dir = MODEL_NAME.replace("/", "_") + f"_{args.gen_method}_{args.lang}_{run_name}"
    if use_wandb:
        repo_name = f"{args.encoder}-{args.gen_method}-{args.lang}-{wandb.run.name}-lr{lr}-bs{bs}"
    else:
        repo_name = f"{args.encoder}-{args.gen_method}-{args.lang}-lr{lr}-bs{bs}"

    training_args = transformers.TrainingArguments(
        output_dir=f"./models/{out_dir}",
        overwrite_output_dir=True,
        report_to="wandb" if use_wandb else "none",
        run_name=f"{args.encoder}-sweep",
        learning_rate=lr,
        do_train=True,
        do_eval=True,
        warmup_steps=args.warmup,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        weight_decay=weight_decay,
        remove_unused_columns=False,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
    )

    trainer = MultitaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model()  # Full HF save
    tokenizer.save_pretrained(f"./models/{out_dir}")
    model.config.save_pretrained(f"./models/{out_dir}")
    trainer.push_to_hub(repo_name)  # Now works fully
    metrics = trainer.evaluate()
    print("Final eval:", metrics)

    # Local .pt for compat
    torch.save({"model_state_dict": model.state_dict()}, f"./models/{out_dir}/final.pt")

    # Optional cleanup
    # import shutil; shutil.rmtree(f"./models/{out_dir}")

    if use_wandb:
        wandb.finish()

