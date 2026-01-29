import transformers
import torch
from torch.utils.data import Dataset, DataLoader
import sys
import os

"""
nlprun -q jag -p standard -r 8G -c 2 \
  -n multilabel_modernbert_eval \
  -o slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation/Encoder-Only/BERT && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python eval.py CGEdit AAE FullTest_Final SociauxLing/multilabel_modernbert"

"""

if len(sys.argv) != 5:
    raise SystemExit("Usage: python eval.py <gen_method> <lang> <test_set> <model_id>")

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

class CustomDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        return {"text": self.texts[idx]}

def build_dataset(test_f, max_length=64):
    texts = []
    with open(test_f, 'r', encoding='utf-8') as r:
        for line in r:
            line = line.strip()
            if len(line.split()) < 2:
                continue
            texts.append(line.split()[0])
    return CustomDataset(texts)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    model = transformers.ModernBertForSequenceClassification.from_pretrained(MODEL_ID, num_labels=17, problem_type="multi_label_classification")
    model.to(device)
    model.eval()

    dataset = build_dataset(test_file)
    dataloader = DataLoader(dataset, batch_size=16)

    os.makedirs("./data/results", exist_ok=True)
    with open(f"./data/results/{out_dir}", "w") as f:
        for batch in dataloader:
            texts = batch["text"]
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=64).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.sigmoid(outputs.logits).cpu().numpy()
            for i, text in enumerate(texts):
                prob_strs = "\t".join(f"{p:.4f}" for p in probs[i])
                f.write(f"{text}\t{prob_strs}\n")
