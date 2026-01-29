import transformers
import torch
from torch.utils.data import Dataset, DataLoader
import sys
import os
import torch.nn as nn

"""
nlprun -q jag -p standard -r 8G -c 2 \
  -n multilabel_modernbert_eval \
  -o slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation/Encoder-Only/BERT && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python eval.py CGEdit AAE SociauxLing/multilabel_modernbert"
"""

gen_method = sys.argv[1]
lang = sys.argv[2]
test_set = sys.argv[3]
MODEL_ID = sys.argv[4]

model_tag = MODEL_ID.replace("/", "_")
out_dir = f"{model_tag}_{gen_method}_{lang}_{test_set}.tsv"

csv_path = f"./data/{test_set}.csv"
txt_path = f"./data/{test_set}.txt"
test_file = csv_path if os.path.exists(csv_path) else txt_path if os.path.exists(txt_path) else None
if test_file is None:
    raise FileNotFoundError(f"Could not find ./data/{test_set}.csv or .txt")

# Register your model
from transformers import AutoConfig, AutoModel

class MultitaskModelConfig(transformers.PretrainedConfig):
    model_type = "multitask_model"

class MultitaskModel(transformers.PreTrainedModel):
    config_class = MultitaskModelConfig

    def __init__(self, encoder, taskmodels_dict, config):
        super().__init__(config)
        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = out.last_hidden_state[:, 0, :]
        logits_per_task = [head(cls_repr) for head in self.taskmodels_dict.values()]
        return torch.stack(logits_per_task, dim=1)

# Register model and config
AutoConfig.register("multitask_model", MultitaskModelConfig)
AutoModel.register(MultitaskModelConfig, MultitaskModel)

class CustomDataset(Dataset):
    def __init__(self, input_ids, attention_mask, texts):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.texts = texts

    def __len__(self): return len(self.input_ids)
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "text": self.texts[idx],
        }

def build_dataset(tokenizer, test_f, max_length=64):
    input_ids_list, attn_list, texts = [], [], []
    with open(test_f, 'r', encoding='utf-8') as r:
        for line in r:
            line = line.strip()
            if len(line.split()) < 2:
                continue
            text = line.split()[0]
            enc = tokenizer(text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
            input_ids_list.append(enc["input_ids"].squeeze(0))
            attn_list.append(enc["attention_mask"].squeeze(0))
            texts.append(text)
    return CustomDataset(torch.stack(input_ids_list), torch.stack(attn_list), texts)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)

    # Load model with trust_remote_code=True to use custom class
    model = transformers.AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.to(device)
    model.eval()

    # Write predictions
    dataset = build_dataset(tokenizer, test_file)
    dataloader = DataLoader(dataset, batch_size=16)

    os.makedirs("./data/results", exist_ok=True)
    with open(f"./data/results/{out_dir}", "w") as f:
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            texts = batch["text"]
            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=-1)[:, :, 1].cpu().numpy()
            for i, text in enumerate(texts):
                prob_strs = "\t".join(f"{p:.4f}" for p in probs[i])
                f.write(f"{text}\t{prob_strs}\n")
