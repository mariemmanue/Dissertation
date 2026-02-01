
"""
nlprun -q jag -p standard -r 8G -c 2 \
  -n multilabel_modernbert_eval \
  -o slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation/Encoder-Only/BERT && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python eval.py CGEdit AAE FullTest_Final SociauxLing/ModernBERT_CGEdit_AAE_39hmoef9"



nlprun -q jag -p standard -r 8G -c 2 \
  -n multilabel_modernbert_eval \
  -o slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation/Encoder-Only/BERT && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python eval.py CGEdit AAE FullTest_Final SociauxLing/ModernBERT_CGEdit_AAE_jsq8h8xo"
"""

import transformers
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys
import os
import numpy as np
from transformers import AutoConfig, AutoModel, AutoTokenizer
import json, os



class MultitaskModel(nn.Module):
    def __init__(self, encoder, taskmodels_dict):
        super().__init__()
        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = out.last_hidden_state[:, 0, :]
        logits_per_task = []
        for head in self.taskmodels_dict.values():
            logits_per_task.append(head(cls_repr))
        logits = torch.stack(logits_per_task, dim=1)  # [B,T,C] or [B,T,1]
        return logits



# --- 2. Helper to reconstruct the model instance ---
def load_multitask_model(model_id, head_list, loss_type):
    from transformers.utils import cached_file
    from safetensors.torch import load_file

    # 1) Load full encoder (with resized embeddings) from MODEL_ID
    encoder = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    hidden_size = encoder.config.hidden_size

    # 2) Build heads consistent with loss_type
    taskmodels_dict = {}
    for name in head_list:
        if loss_type == "ce":
            taskmodels_dict[name] = nn.Linear(hidden_size, 2)
        else:
            taskmodels_dict[name] = nn.Linear(hidden_size, 1)

    model = MultitaskModel(encoder=encoder, taskmodels_dict=taskmodels_dict)

    # 3) Load checkpoint state dict, strip _orig_mod.
    model_file = cached_file(model_id, "model.safetensors")
    sd = load_file(model_file)
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("_orig_mod."):
            new_k = k[len("_orig_mod."):]
        else:
            new_k = k
        new_sd[new_k] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print("Missing:", len(missing), "Unexpected:", len(unexpected))
    if missing:
        print("Example missing:", missing[:10])
    if unexpected:
        print("Example unexpected:", unexpected[:10])

    return model



# --- 3. Feature Definition (Must match train.py exactly) ---

def load_head_list(lang: str):
    if lang == "AAE":
        return [
            "zero-poss", "zero-copula", "double-tense", "be-construction",
            "resultant-done", "finna", "come", "double-modal",
            "multiple-neg", "neg-inversion", "n-inv-neg-concord", "aint",
            "zero-3sg-pres-s", "is-was-gen", "zero-pl-s", "double-object", "wh-qu",
        ]
    elif lang == "IndE":
        return [
            "foc_self", "foc_only", "left_dis", "non_init_exis", "obj_front",
            "inv_tag", "cop_omis", "res_obj_pron", "res_sub_pron", "top_non_arg_con",
        ]
    else:
        raise ValueError("lang must be 'AAE' or 'IndE'")

# --- 4. Data Loading ---

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

def build_dataset(tokenizer, test_f, max_length=128):
    input_ids_list, attn_list, texts = [], [], []

    # 1. Robust File Path Resolution
    if not os.path.exists(test_f):
        # Check if it's in ./data/
        if os.path.exists(os.path.join("./data", test_f)):
            test_f = os.path.join("./data", test_f)
        # Check if it's missing extension
        elif os.path.exists(test_f + ".csv"):
            test_f = test_f + ".csv"
        elif os.path.exists(test_f + ".txt"):
            test_f = test_f + ".txt"
        # Check if it's in ./data/ AND missing extension
        elif os.path.exists(os.path.join("./data", test_f + ".csv")):
             test_f = os.path.join("./data", test_f + ".csv")
        elif os.path.exists(os.path.join("./data", test_f + ".txt")):
             test_f = os.path.join("./data", test_f + ".txt")
        else:
            raise FileNotFoundError(f"Could not find file: {test_f}")

    print(f"Reading {test_f}...")
    
    if test_f.endswith('.csv'):
        import pandas as pd
        df = pd.read_csv(test_f)
        # Try to find the sentence column
        possible_cols = ['sentence', 'text', 'utterance']
        col = next((c for c in possible_cols if c in df.columns), df.columns[0])
        lines = df[col].astype(str).tolist()
    else:
        with open(test_f, 'r', encoding='utf-8') as r:
            lines = [line.strip() for line in r if line.strip()]

    print(f"Tokenizing {len(lines)} examples...")

    for text in lines:
        if not text: continue
        enc = tokenizer(text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
        input_ids_list.append(enc["input_ids"].squeeze(0))
        attn_list.append(enc["attention_mask"].squeeze(0))
        texts.append(text)
        
    return CustomDataset(torch.stack(input_ids_list), torch.stack(attn_list), texts)

# --- 5. Main ---

if __name__ == "__main__":
    # Usage: python eval.py <gen_method> <lang> <test_file> <model_id> [eval_labels_file]
    if len(sys.argv) < 5:
        print("Usage: python eval.py <gen_method> <lang> <test_file> <model_id> [eval_labels_file]")
        sys.exit(1)

    gen_method = sys.argv[1]
    lang = sys.argv[2]
    test_file = sys.argv[3]
    MODEL_ID = sys.argv[4]
    eval_labels_file = sys.argv[5] if len(sys.argv) > 5 else None
    # Optional gold labels (CGEdit TSV) for ranking metrics
    gold_labels = None
    if eval_labels_file is not None:
        texts_g = []
        labels_g = []
        with open(eval_labels_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                parts = line.rstrip("\n").split("\t")
                if i == 0 and len(parts) > 1 and not parts[1].isdigit():
                    continue
                texts_g.append(parts[0])
                labels_g.append([int(x) for x in parts[1:]])
        gold_labels = np.array(labels_g, dtype=np.int64)  # [N,T]

    # After parsing MODEL_ID
    model_dir = MODEL_ID  # works both for local path and HF repo if you mirror layout

    # Try to load extra_config.json written by train.py
    loss_type = "ce"
    try:
        with open(os.path.join(model_dir, "extra_config.json")) as f:
            extra_cfg = json.load(f)
            loss_type = extra_cfg.get("loss_type", "ce")
    except FileNotFoundError:
        pass

    thresholds = None
    if loss_type == "bce":
        thr_path = os.path.join(model_dir, "thresholds.npy")
        if os.path.exists(thr_path):
            thresholds = np.load(thr_path)  # [T]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Get Head List
    head_list = load_head_list(lang)
    print(f"Task Heads ({len(head_list)}): {head_list}")

    # 2. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # 3. Load Model (Reconstructing Architecture)

    print(f"Loading model directly from {MODEL_ID}...")
    model = load_multitask_model(MODEL_ID, head_list, loss_type)
    model.to(device)
    model.eval()


    # 4. Prepare Data
    dataset = build_dataset(tokenizer, test_file)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 5. Inference
    os.makedirs("./data/results", exist_ok=True)
    out_name = f"{MODEL_ID.split('/')[-1]}_{gen_method}_{lang}_{os.path.basename(test_file)}_preds.tsv"
    out_path = f"./data/results/{out_name}"
    aug_out_path = f"./data/results/{MODEL_ID.split('/')[-1]}_{gen_method}_{lang}_{os.path.basename(test_file)}_probs_and_preds.tsv"
    labels_out_path = f"./data/results/{MODEL_ID.split('/')[-1]}_{gen_method}_{lang}_{os.path.basename(test_file)}_labels.tsv"

    print(f"Starting inference, writing to {out_path}...")
    
    with open(out_path, "w", encoding='utf-8') as f_probs, open(aug_out_path, "w", encoding="utf-8") as f_aug, open(labels_out_path, "w", encoding="utf-8") as f_labels:

        all_texts = []
        all_probs = []

        # Header for simple probs file (same as before)
        header_probs = "sentence\t" + "\t".join(head_list) + "\n"
        f_probs.write(header_probs)

        # Header for augmented file: for each feature, prob0, prob1, pred
        if loss_type == "ce":
            aug_cols = []
            for feat in head_list:
                aug_cols.extend([f"{feat}_p0", f"{feat}_p1", f"{feat}_pred"])
        else:
            aug_cols = []
            for feat in head_list:
                aug_cols.extend([f"{feat}_p1", f"{feat}_pred"])
        header_aug = "sentence\t" + "\t".join(aug_cols) + "\n"
        f_aug.write(header_aug)

        header_labels = "sentence\t" + "\t".join(head_list) + "\n"
        f_labels.write(header_labels)

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            texts = batch["text"]

            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask)

                if loss_type == "ce":
                    probs = torch.softmax(logits, dim=-1)       # [B,T,2]
                    probs_np = probs.cpu().numpy()
                    prob1 = probs_np[..., 1]                    # [B,T]
                    preds_np = probs_np.argmax(axis=-1)         # [B,T]
                else:
                    if logits.ndim == 3 and logits.shape[-1] == 1:
                        logits = logits.squeeze(-1)
                    z = logits.cpu().numpy()                    # [B,T]
                    prob1 = 1.0 / (1.0 + np.exp(-z))            # [B,T]
                    if thresholds is not None:
                        preds_np = (z > thresholds).astype(int) # [B,T]
                    else:
                        preds_np = (z > 0).astype(int)          # [B,T]

            for i, text in enumerate(texts):
                clean_text = text.replace("\t", " ").replace("\n", " ")

                # labels file
                label_strs = "\t".join(str(int(preds_np[i, t])) for t in range(len(head_list)))
                f_labels.write(f"{clean_text}\t{label_strs}\n")

                # probs1 file
                prob1_strs = "\t".join(f"{p1:.4f}" for p1 in prob1[i])
                f_probs.write(f"{clean_text}\t{prob1_strs}\n")

                # augmented file
                if loss_type == "ce":
                    feat_cells = []
                    for t in range(len(head_list)):
                        p0 = probs_np[i, t, 0]
                        p1 = probs_np[i, t, 1]
                        pred = preds_np[i, t]
                        feat_cells.extend([f"{p0:.4f}", f"{p1:.4f}", str(pred)])
                    f_aug.write(f"{clean_text}\t" + "\t".join(feat_cells) + "\n")
                else:
                    feat_cells = []
                    for t in range(len(head_list)):
                        p1 = prob1[i, t]
                        pred = preds_np[i, t]
                        feat_cells.extend([f"{p1:.4f}", str(pred)])
                    f_aug.write(f"{clean_text}\t" + "\t".join(feat_cells) + "\n")

                all_texts.append(clean_text)
                all_probs.append(prob1[i])

    all_probs = np.stack(all_probs)  # [N,T]

    if gold_labels is not None:
        assert all_probs.shape == gold_labels.shape
        from sklearn.metrics import average_precision_score

        N, T = all_probs.shape
        aps = []
        prec100s = []
        for t in range(T):
            y_true = gold_labels[:, t]
            y_score = all_probs[:, t]
            if y_true.sum() == 0:
                continue
            ap_t = average_precision_score(y_true, y_score)
            aps.append(ap_t)

            order = np.argsort(y_score)[::-1]
            K = min(100, N)
            topk = order[:K]
            prec_t = y_true[topk].mean()
            prec100s.append(prec_t)

        print("Macro AP:", float(np.mean(aps)) if aps else 0.0)
        print("Macro Prec@100:", float(np.mean(prec100s)) if prec100s else 0.0)

    print("Done.")