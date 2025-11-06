import transformers
import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split
import sys
import os
import numpy as np

# ModernBERT wants transformers >= 4.48.0
MODEL_NAME = "answerdotai/ModernBERT-large"

gen_method = sys.argv[1]    # 'CGEdit' or 'CGEdit-ManualGen'
lang = sys.argv[2]          # 'AAE' or 'IndE'

train_file = f"./data/{gen_method}/{lang}.tsv"
out_dir = MODEL_NAME.replace("/", "_") + f"_{gen_method}_{lang}"

# try to init wandb
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
        cls_repr = out.last_hidden_state[:, 0, :]
        logits = []
        for _, head in self.taskmodels_dict.items():
            logits.append(head(cls_repr))
        return torch.vstack(logits)


class MultitaskTrainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = torch.transpose(inputs["labels"], 0, 1)
        labels = torch.flatten(labels)
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs, labels)
        return (loss, outputs) if return_outputs else loss


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
    logits, labels = eval_pred
    # logits: [num_tasks * batch, 2]
    # labels: [num_tasks * batch]
    preds = np.argmax(logits, axis=-1)
    labels = labels.reshape(-1)
    preds = preds.reshape(-1)

    # accuracy
    acc = (preds == labels).mean()

    # macro F1 (no sklearn)
    f1s = []
    for cls in [0, 1]:
        tp = np.sum((preds == cls) & (labels == cls))
        fp = np.sum((preds == cls) & (labels != cls))
        fn = np.sum((preds != cls) & (labels == cls))
        if tp == 0 and fp == 0 and fn == 0:
            f1 = 0.0
        else:
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s))
    return {"accuracy": acc, "eval_f1": macro_f1}


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    head_type_list = load_head_list(lang)
    model = MultitaskModel.create(MODEL_NAME, head_type_list)
    model.to(device)

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)

    # get default hyperparams
    default_lr = 1e-4
    default_bs = 32
    default_epochs = 20
    default_warmup = 300
    default_max_len = 64

    if use_wandb:
        wandb.init(project="aae-ddm-modernbert")
        lr = wandb.config.get("learning_rate", default_lr)
        bs = wandb.config.get("batch_size", default_bs)
        epochs = wandb.config.get("epochs", default_epochs)
        warmup = wandb.config.get("warmup_steps", default_warmup)
        max_len = wandb.config.get("max_length", default_max_len)
    else:
        lr = default_lr
        bs = default_bs
        epochs = default_epochs
        warmup = default_warmup
        max_len = default_max_len

    dataset = build_dataset(tokenizer, train_file, max_length=max_len)

    # 90/10 split for val
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    training_args = transformers.TrainingArguments(
        output_dir="./models/" + out_dir,
        overwrite_output_dir=True,
        learning_rate=lr,
        do_train=True,
        do_eval=True,
        warmup_steps=warmup,
        num_train_epochs=epochs,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        save_steps=500,
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        report_to=["wandb"] if use_wandb else [],
    )

    trainer = MultitaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    os.makedirs(f"./models/{out_dir}", exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict()},
        f"./models/{out_dir}/final.pt",
    )

    if use_wandb:
        wandb.finish()
