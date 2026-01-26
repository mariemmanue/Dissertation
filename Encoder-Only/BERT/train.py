import transformers
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import sys
import os

"""
nlprun -q jag -p standard -r 8G -c 2 \
  -n bert-train-flowing-field \
  -o slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation/Encoder-Only/BERT && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python train.py CGEdit AAE FullTest_Final SociauxLing/modernbert-aae-CGEdit-flowing-field-372-lr2e-05-bs16"
"""

if len(sys.argv) != 5:
    raise SystemExit("Usage: python train.py <gen_method> <lang> <train_set> <model_id>")

gen_method = sys.argv[1]
lang = sys.argv[2]
train_set = sys.argv[3]
MODEL_ID = sys.argv[4]

model_tag = MODEL_ID.replace("/", "_")
out_dir = f"{model_tag}_{gen_method}_{lang}_{train_set}"

csv_path = f"./data/{train_set}.csv"
txt_path = f"./data/{train_set}.txt"
train_file = csv_path if os.path.exists(csv_path) else txt_path if os.path.exists(txt_path) else None
if train_file is None:
    raise FileNotFoundError(f"Could not find ./data/{train_set}.csv or .txt")

class MultitaskModel(transformers.PreTrainedModel):
    def __init__(self, encoder, taskmodels_dict, config):
        super().__init__(config)
        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = out.last_hidden_state[:, 0, :]
        logits_per_task = [head(cls_repr) for head in self.taskmodels_dict.values()]
        return torch.stack(logits_per_task, dim=1)  # (B, num_tasks, 2)

def train_dataloader(dataset, batch_size=64):
    return DataLoader(dataset, batch_size=batch_size, sampler=SequentialSampler(dataset))

class CustomDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels, texts):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.texts = texts

    def __len__(self): return len(self.input_ids)
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
            "text": self.texts[idx],
        }

def build_dataset(tokenizer, train_f, max_length=64):
    input_ids_list, attn_list, labels_list, texts = [], [], [], []
    with open(train_f, 'r', encoding='utf-8') as r:
        header = next(r)  # Skip the header row
        for i, line in enumerate(r):
            line = line.strip()
            if len(line.split()) < 2:  # Skip empty or malformed lines
                print(f"Skipping line {i+1}: {line}")
                continue
            parts = line.split()
            text = parts[0]
            try:
                labels = [int(x) for x in parts[1:]]
            except ValueError as e:
                print(f"Error parsing labels in line {i+1}: {line}")
                continue
            enc = tokenizer(text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
            input_ids_list.append(enc["input_ids"].squeeze(0))
            attn_list.append(enc["attention_mask"].squeeze(0))
            labels_list.append(torch.tensor(labels))
            texts.append(text)
    
    if not input_ids_list:  # Check if any valid data points were found
        raise ValueError("No valid data points found in the training file.")
    
    return CustomDataset(torch.stack(input_ids_list), torch.stack(attn_list), torch.stack(labels_list), texts)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Head lists (exact training order)
    if lang == "AAE":
        head_type_list = ["zero-poss", "zero-copula", "double-tense", "be-construction", "resultant-done", "finna", "come", "double-modal", "multiple-neg", "neg-inversion", "n-inv-neg-concord", "aint", "zero-3sg-pres-s", "is-was-gen", "zero-pl-s", "double-object", "wh-qu"]
    elif lang == "IndE":
        head_type_list = ["foc_self", "foc_only", "left_dis", "non_init_exis", "obj_front", "inv_tag", "cop_omis", "res_obj_pron", "res_sub_pron", "top_non_arg_con"]
    else:
        raise ValueError("lang must be AAE or IndE")

    num_tasks = len(head_type_list)
    # Load tokenizer and model with eager attention implementation
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    model = transformers.AutoModel.from_pretrained(MODEL_ID, attn_implementation="eager")

    # Manually specify the heads list
    heads = head_type_list

    # Create your multitask model
    taskmodels_dict = {task: nn.Linear(model.config.hidden_size, 2) for task in heads}
    multitask_model = MultitaskModel(encoder=model, taskmodels_dict=taskmodels_dict, config=model.config)

    # Perform training
    multitask_model.to(device)
    multitask_model.train()

    dataset = build_dataset(tokenizer, train_file)
    dataloader = train_dataloader(dataset)

    optimizer = torch.optim.Adam(multitask_model.parameters(), lr=2e-5)

    for epoch in range(3):  # Example: 3 epochs
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            texts = batch["text"]

            optimizer.zero_grad()
            logits = multitask_model(input_ids=input_ids, attention_mask=attention_mask)  # (bsz, num_tasks, 2)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, 2), labels.view(-1))
            loss.backward()
            optimizer.step()

    # Save the entire model
    torch.save(multitask_model.state_dict(), f"./models/{out_dir}.pt")
