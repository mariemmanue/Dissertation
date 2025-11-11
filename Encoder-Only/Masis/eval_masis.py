#!/usr/bin/env python
import os
import sys
import csv
import torch
import torch.nn as nn
import transformers

from torch.utils.data import Dataset, DataLoader, SequentialSampler

"""
Run Masis et al.'s 17-feature BERT model over a directory of transcripts
stored as .csv or .txt (one utterance per line / per row), and write out
a single TSV with:

filename \t utterance \t <17 probs...>

Usage (typical):

    python eval_masis_csv.py \
        --model ./models/masis-final.pt \
        --data_dir ./data \
        --out ./data/results/masis_coraal.tsv


nlprun -q jag -p standard -r 8G -c 2 -t 0-2 \
  -n masis-eval-coraal \
  -o slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation/Encoder-Only/Masis && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   mkdir -p data/results slurm_logs && \
   python eval_masis_csv.py \
      --model ./models/masis-final.pt \
      --data_dir ./data \
      --out ./data/results/masis_coraal.tsv"

"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True,
                    help="path to Masis checkpoint, e.g. ./models/masis-final.pt")
parser.add_argument("--data_dir", required=True,
                    help="folder containing .csv or .txt transcripts")
parser.add_argument("--out", required=True,
                    help="output tsv file")
parser.add_argument("--max_files", type=int, default=None,
                    help="optional: limit number of files (for testing)")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "bert-base-cased"

# the 17 heads in the original Masis repo
HEADS = [
    "1-zero-poss",
    "3-zero-copula",
    "4-double-tense",
    "5-be-construction",
    "5-resultant-done",
    "6-finna",
    "6-come",
    "6-double-modal",
    "7-multiple-neg",
    "7-neg-inversion",
    "7-n-inv-neg-concord",
    "7-aint",
    "8-zero-3sg-pres-s",
    "8-is-was-gen",
    "9-zero-pl-s",
    "10-double-object",
    "11-wh-qu",
]

# ------------------ model definition (same shape as their eval) ------------------
class MultitaskModel(transformers.PreTrainedModel):
    def __init__(self, encoder, taskmodels_dict):
        super().__init__(transformers.PretrainedConfig())
        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    @classmethod
    def create(cls, model_name, head_type_list):
        taskmodels_dict = {}
        shared_encoder = transformers.AutoModel.from_pretrained(
            model_name,
            config=transformers.AutoConfig.from_pretrained(model_name)
        )
        for head_name in head_type_list:
            taskmodels_dict[head_name] = nn.Linear(768, 2)
        return cls(shared_encoder, taskmodels_dict)

    def forward(self, inputs, **kwargs):
        out = self.encoder(inputs)
        cls_vec = out.last_hidden_state[:, 0, :]
        logits = []
        for name, head in self.taskmodels_dict.items():
            logits.append(head(cls_vec))
        return torch.vstack(logits)


class TextDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        return {"input_ids": self.tensors[idx]}


def make_dataloader(dataset, batch_size=64):
    sampler = SequentialSampler(dataset)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


# ------------------ load model ------------------
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
model = MultitaskModel.create(MODEL_NAME, HEADS)
ckpt = torch.load(args.model, map_location=device)
state = ckpt["model_state_dict"]
model.load_state_dict(state, strict=False)
model.to(device)
model.eval()

os.makedirs(os.path.dirname(args.out), exist_ok=True)
out_f = open(args.out, "w", encoding="utf-8")

# nice header for Excel
out_header = ["filename", "utterance"] + HEADS
out_f.write("\t".join(out_header) + "\n")

# ------------------ helper to read a file into utterances ------------------
def read_utterances_from_file(path):
    utts = []
    if path.endswith(".csv"):
        # assume first column is the transcript line
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                text = row[0].strip()
                if len(text.split()) < 1:
                    continue
                utts.append(text)
    else:
        # .txt or anything else: one utterance per line
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # original script skipped 1-word lines; keep that behavior:
                if len(line.split()) < 2:
                    continue
                utts.append(line)
    return utts

# ------------------ main loop over data_dir ------------------
files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir)
         if f.endswith(".csv") or f.endswith(".txt")]

files = sorted(files)
if args.max_files:
    files = files[:args.max_files]

for path in files:
    fname = os.path.basename(path)
    print(f"Processing {fname} ...", file=sys.stderr)
    utterances = read_utterances_from_file(path)
    if not utterances:
        continue

    tensor_list = []
    for utt in utterances:
        tok = tokenizer.encode(
            utt,
            max_length=64,
            padding="max_length",
            truncation=True,
        )
        tensor_list.append(torch.LongTensor(tok))
    batch_ds = TextDataset(tensor_list)
    loader = make_dataloader(batch_ds, batch_size=64)

    # run model
    for batch_idx, batch in enumerate(loader):
        ids = batch["input_ids"].to(device)
        with torch.no_grad():
            outputs = model(ids)
        outputs = torch.nn.functional.softmax(outputs, dim=1)
        num_heads = len(HEADS)
        bsz = ids.size(0)
        outputs = outputs.view(num_heads, bsz, 2)

        for i in range(bsz):
            text = utterances[batch_idx * 64 + i]
            probs = []
            for h in range(num_heads):
                probs.append(str(float(outputs[h, i, 1].cpu())))
            out_f.write(fname + "\t" + text.replace("\t", " ") + "\t" + "\t".join(probs) + "\n")

out_f.close()
print(f"Done. Wrote to {args.out}")
