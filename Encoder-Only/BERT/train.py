import transformers
import torch
from torch.utils.data import Dataset, DataLoader
import sys
import os

"""
nlprun -q jag -p standard -r 8G -c 2 \
  -n multilabel_modernbert_train \
  -o slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation/Encoder-Only/BERT && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python train.py CGEdit AAE SociauxLing/multilabel_modernbert"

"""

gen_method = sys.argv[1]
lang = sys.argv[2]
MODEL_ID = sys.argv[3]

model_tag = MODEL_ID.replace("/", "_")
out_dir = f"{model_tag}_{gen_method}_{lang}_{lang}"

csv_path = f"./data/{lang}.csv"
txt_path = f"./data/{lang}.txt"
train_file = csv_path if os.path.exists(csv_path) else txt_path if os.path.exists(txt_path) else None
if train_file is None:
    raise FileNotFoundError(f"Could not find ./data/{lang}.csv or .txt")

class CustomDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        return {"text": self.texts[idx], "labels": self.labels[idx]}

def build_dataset(train_f, max_length=64):
    texts, labels = [], []
    with open(train_f, 'r', encoding='utf-8') as r:
        header = next(r)
        for line in r:
            line = line.strip()
            if len(line.split()) < 2:
                continue
            parts = line.split()
            texts.append(parts[0])
            try:
                labels.append([int(x) for x in parts[1:]])
            except ValueError:
                continue
    return CustomDataset(texts, labels)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    model = transformers.ModernBertForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=17, problem_type="multi_label_classification"
    )
    model.to(device)

    dataset = build_dataset(train_file)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.train()

    for epoch in range(3):
        for batch in dataloader:
            texts = batch["text"]
            labels = torch.tensor(batch["labels"]).float().to(device)
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=64).to(device)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    model.save_pretrained(f"./models/{out_dir}")
    model.push_to_hub(f"SociauxLing/modernbert")
