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

import transformers
import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# try wandb
use_wandb = False
try:
    import wandb
    use_wandb = True
except ImportError:
    wandb = None
    use_wandb = False


class MultitaskModel(transformers.PreTrainedModel):
    def __init__(self, encoder, taskmodels_dict, config):
        super().__init__(config)
        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    @classmethod
    def create(cls, model_name, head_type_list):
        config = transformers.AutoConfig.from_pretrained(model_name)
        encoder = transformers.AutoModel.from_pretrained(model_name, config=config)
        hidden_size = config.hidden_size

        taskmodels_dict = {}
        for task_name in head_type_list:
            taskmodels_dict[task_name] = nn.Linear(hidden_size, 2)

        return cls(encoder=encoder, taskmodels_dict=taskmodels_dict, config=config)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = out.last_hidden_state[:, 0, :]  # (B, H)
        logits_per_task = []
        for _, head in self.taskmodels_dict.items():
            logits_per_task.append(head(cls_repr))  # (B, 2)
        # stack into (B, num_tasks, 2)
        logits = torch.stack(logits_per_task, dim=1)
        return logits



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

    def prediction_step(
        self, model, inputs, prediction_loss_only, ignore_keys=None
    ):
        """
        Custom eval step that always returns (loss, logits, labels) with no grad,
        so Trainer can compute eval_loss and call compute_metrics.
        """
        # Shallow copy so we don't mutate original
        inputs = inputs.copy()

        # Extract labels
        labels = inputs.get("labels", None)

        # Eval mode, no grad
        model.eval()
        with torch.no_grad():
            loss, logits = self.compute_loss(
                model, inputs, return_outputs=True
            )

        # Detach for numpy conversion
        if loss is not None:
            loss = loss.detach()
        if logits is not None:
            logits = logits.detach()
        if labels is not None and isinstance(labels, torch.Tensor):
            labels = labels.detach()

        if prediction_loss_only:
            return loss, None, None

        return loss, logits, labels

# class DebugMultitaskTrainer(MultitaskTrainer):
#     def prediction_step(
#         self, model, inputs, prediction_loss_only, ignore_keys=None
#     ):
#         # Shallow copy so we don't mutate original
#         inputs = inputs.copy()

#         # Extract labels
#         labels = inputs.get("labels", None)

#         # Disable grad during eval
#         model.eval()
#         with torch.no_grad():
#             loss, logits = self.compute_loss(
#                 model, inputs, return_outputs=True
#             )

#         # Detach for Trainer's numpy conversion
#         if loss is not None:
#             loss = loss.detach()
#         if logits is not None:
#             logits = logits.detach()

#         if labels is not None and isinstance(labels, torch.Tensor):
#             labels = labels.detach()

#         # Debug print (eval only)
#         if not model.training:
#             print(
#                 ">>> prediction_step (eval):",
#                 "loss ok" if loss is not None else "loss None",
#                 "logits", getattr(logits, "shape", None),
#                 "labels", getattr(labels, "shape", None),
#             )

#         if prediction_loss_only:
#             return loss, None, None

#         return loss, logits, labels






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
    print(">>> compute_metrics CALLED")
    logits, labels = eval_pred    # logits: (B, T, 2), labels: (B, T)
    # flatten both
    B, T, C = logits.shape
    logits_flat = logits.reshape(B * T, C)
    labels_flat = labels.reshape(B * T)

    preds = logits_flat.argmax(axis=1)
    f1 = f1_score(labels_flat, preds, average="macro")
    acc = accuracy_score(labels_flat, preds)
    return {"eval_f1": f1, "eval_accuracy": acc}





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
        "neobert": "sage-fc/neoBERT-large",          # example; pick the right IDs
        "roberta": "roberta-large",
        "bert": "bert-large-uncased",
    }
    MODEL_NAME = ENCODER_MAP[args.encoder]

    # MODEL_NAME = "answerdotai/ModernBERT-large"
    train_file = f"./data/{args.gen_method}/{args.lang}.tsv"
    # out_dir = MODEL_NAME.replace("/", "_") + f"_{args.gen_method}_{args.lang}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    head_type_list = load_head_list(args.lang)
    model = MultitaskModel.create(MODEL_NAME, head_type_list).to(device)

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = build_dataset(tokenizer, train_file, max_length=max_len)

    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    run_name = os.environ.get("WANDB_RUN_ID", None) or \
            os.environ.get("WANDB_RUN_NAME", None) or "no-wandb"

    out_dir = MODEL_NAME.replace("/", "_") + f"_{args.gen_method}_{args.lang}_{run_name}"
    base = f"{args.encoder}-{args.gen_method}-{args.lang}"
    repo_name = f"{base}-{wandb.run.name}-lr{lr}-bs{bs}"


    training_args = transformers.TrainingArguments(
        output_dir="./models/" + out_dir,
        overwrite_output_dir=False,
        report_to="wandb",
        run_name=f"{args.encoder}-sweep",   # e.g., modernbert-sweep, neobert-sweep
        learning_rate=lr,
        do_train=True,
        do_eval=True,
        warmup_steps=warmup,
        num_train_epochs=epochs,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        weight_decay=weight_decay,
        remove_unused_columns=False,
        logging_steps=50,
        # --- Evaluation & saving ---
        eval_strategy="epoch",
        save_strategy="epoch",   # match eval
        save_total_limit=1,      # keep only latest checkpoint
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

    # if use_wandb:
    #     base = f"modernbert-aae-{args.gen_method}-{args.lang}"
    #     repo_name = f"{base}-{wandb.run.name}-lr{lr}-bs{bs}"
    # else:
    #     repo_name = f"modernbert-aae-{args.gen_method}-{args.lang}-lr{lr}-bs{bs}"

    # Push best-epoch model to Hugging Face
    trainer.push_to_hub(repo_name)
    metrics = trainer.evaluate()
    print(">>> eval metrics dict:", metrics)

    # Optional: save a small local copy before deleting
    os.makedirs(f"./models/{out_dir}", exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict()},
        f"./models/{out_dir}/final.pt",
    )

    # Immediately delete the whole output_dir to free space
    import shutil
    shutil.rmtree(f"./models/{out_dir}", ignore_errors=True)

    if use_wandb:
        wandb.finish()
