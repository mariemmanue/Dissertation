import transformers
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import sys

# ModernBERT wants transformers >= 4.48.0 (per model card)
MODEL_NAME = "answerdotai/ModernBERT-large"

gen_method = sys.argv[1]    # 'CGEdit' or 'CGEdit-ManualGen'
lang = sys.argv[2]          # 'AAE' or 'IndE'

# we'll keep your data layout:
# ./data/CGEdit/AAE.tsv or ./data/CGEdit-ManualGen/AAE.tsv
train_file = f"./data/{gen_method}/{lang}.tsv"

# output dir: swap "/" so the folder name is valid
out_dir = MODEL_NAME.replace("/", "_") + f"_{gen_method}_{lang}"


class MultitaskModel(transformers.PreTrainedModel):
    def __init__(self, encoder, taskmodels_dict, config):
        super().__init__(config)
        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    @classmethod
    def create(cls, model_name, head_type_list):
        config = transformers.AutoConfig.from_pretrained(model_name)
        encoder = transformers.AutoModel.from_pretrained(model_name, config=config)
        hidden_size = config.hidden_size  # don't hardcode 768 for ModernBERT

        taskmodels_dict = {}
        for task_name in head_type_list:
            taskmodels_dict[task_name] = nn.Linear(hidden_size, 2)

        return cls(encoder=encoder, taskmodels_dict=taskmodels_dict, config=config)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # ModernBERT doesn't use token_type_ids, so we omit them
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = out.last_hidden_state[:, 0, :]  # CLS
        logits = []
        for name, head in self.taskmodels_dict.items():
            logits.append(head(cls_repr))
        return torch.vstack(logits)


class MultitaskTrainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # inputs["labels"]: [batch, num_feats]
        labels = torch.transpose(inputs["labels"], 0, 1)
        labels = torch.flatten(labels)  # [num_feats * batch]

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
            # skip header with feature names
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


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    head_type_list = load_head_list(lang)
    model = MultitaskModel.create(MODEL_NAME, head_type_list)
    model.to(device)

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)

    dataset = build_dataset(tokenizer, train_file, max_length=64)

    trainer = MultitaskTrainer(
        model=model,
        args=transformers.TrainingArguments(
            output_dir="./models/" + out_dir,
            overwrite_output_dir=True,
            learning_rate=1e-4,
            do_train=True,
            warmup_steps=300,
            num_train_epochs=20,          # was 500
            per_device_train_batch_size=32,  # was 64; safer on cluster
            save_steps=500,
            remove_unused_columns=False,
        ),
        train_dataset=dataset,
    )


    trainer.train()

    torch.save(
        {"model_state_dict": model.state_dict()},
        f"./models/{out_dir}/final.pt",
    )
