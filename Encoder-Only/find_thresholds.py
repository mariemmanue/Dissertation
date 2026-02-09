import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import numpy as np
import argparse
import os
from tqdm import tqdm
import json

# --- ARGS ---
parser = argparse.ArgumentParser()
parser.add_argument("model_dir", type=str, help="Path to the trained model directory")
parser.add_argument("data_file", type=str, help="Path to .tsv")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

# --- MODEL DEFINITION (Must match training) ---
class MultitaskModel(torch.nn.Module):
    def __init__(self, encoder, taskmodels_dict):
        super().__init__()
        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    @classmethod
    def load(cls, model_dir, head_type_list):
        # Load Config & Encoder
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        # Fix for ModernBERT compilation issue if present in config
        if hasattr(config, "reference_compile"):
            config.reference_compile = False
            
        encoder = AutoModel.from_pretrained(model_dir, config=config, trust_remote_code=True)
        
        # Recreate Heads
        taskmodels_dict = {}
        hidden_size = config.hidden_size
        for task_name in head_type_list:
            head = torch.nn.Linear(hidden_size, 2)
            taskmodels_dict[task_name] = head
        
        model = cls(encoder, taskmodels_dict)
        
        # Load State Dict
        state_dict_path = os.path.join(model_dir, "pytorch_model.bin")
        print(f"Loading weights from {state_dict_path}...")
        state_dict = torch.load(state_dict_path, map_location="cpu")
        model.load_state_dict(state_dict)
        
        return model

    def forward(self, input_ids, attention_mask=None):
        x = self.encoder(input_ids, attention_mask=attention_mask)
        # ModernBERT CLS token is at index 0
        cls_token = x.last_hidden_state[:, 0, :]
        
        out_list = []
        for task_name in self.taskmodels_dict.keys():
            head = self.taskmodels_dict[task_name]
            out_list.append(head(cls_token))
        return torch.stack(out_list, dim=1) # [B, T, 2]

# --- DATASET ---
class InferenceDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def main():
    # 1. FEATURES LIST (Must match training order exactly)
    features = [
        "zero-poss", "zero-copula", "double-tense", "be-construction",
        "resultant-done", "finna", "come", "double-modal",
        "multiple-neg", "neg-inversion", "n-inv-neg-concord", "aint",
        "zero-3sg-pres-s", "is-was-gen", "zero-pl-s", "double-object", "wh-qu"
    ]

    # 2. LOAD DATA
    full_data_path = os.path.join("..", "Datasets", args.data_file)

    print(f"Loading data from {full_data_path}...")
    input_ids = []
    labels = []
    
    # Use full_data_path here instead of args.data_file directly
    with open(full_data_path, 'r', encoding='utf-8') as r:
        for line in r:
            line = line.strip()
            if not line or not line.split("\t")[1].isdigit(): 
                continue
            parts = line.split("\t")
            input_ids.append(parts[0])
            labels.append([int(x) for x in parts[1:]])

    # RE-CREATE SPLIT (Critical: Must match training split seed=42)
    from sklearn.model_selection import train_test_split
    _, dev_texts, _, dev_labels = train_test_split(
        input_ids, labels, test_size=0.2, random_state=42, stratify=None
    )
    print(f"Evaluating on {len(dev_texts)} Dev examples.")

    # 3. PREPARE MODEL & TOKENIZER
    print(f"Loading model from {args.model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = MultitaskModel.load(args.model_dir, features)
    model.to(args.device)
    model.eval()

    # 4. TOKENIZE & DATALOADER
    print("Tokenizing...")
    dev_enc = tokenizer(dev_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
    dataset = InferenceDataset(dev_enc, dev_labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # 5. GET LOGITS
    all_probs = []
    all_labels = []
    
    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids_batch = batch['input_ids'].to(args.device)
            mask_batch = batch['attention_mask'].to(args.device)
            labels_batch = batch['labels'].numpy()
            
            logits = model(input_ids_batch, mask_batch) # [B, NumFeatures, 2]
            probs = torch.softmax(logits, dim=2)[:, :, 1] # Get Prob of Class 1
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels_batch)

    all_probs = np.vstack(all_probs) # [N, NumFeatures]
    all_labels = np.vstack(all_labels) # [N, NumFeatures]

    # 6. FIND OPTIMAL THRESHOLDS
    thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
    best_config = {}
    
    print("\n--- THRESHOLD OPTIMIZATION RESULTS ---")
    print(f"{'Feature':<20} | {'Best Thresh':<12} | {'Old F1 (0.5)':<12} | {'New F1':<12} | {'Gain'}")
    print("-" * 75)

    total_old_f1 = 0
    total_new_f1 = 0

    for i, feature in enumerate(features):
        y_true = all_labels[:, i]
        y_probs = all_probs[:, i]
        
        # Calculate F1 for default threshold 0.5
        y_pred_default = (y_probs >= 0.5).astype(int)
        old_f1 = f1_score(y_true, y_pred_default, zero_division=0)
        
        best_f1 = -1
        best_thresh = 0.5
        
        # Search for best threshold
        for t in thresholds:
            y_pred = (y_probs >= t).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
        
        gain = best_f1 - old_f1
        total_old_f1 += old_f1
        total_new_f1 += best_f1
        
        best_config[feature] = best_thresh
        
        # Highlight significant gains
        gain_str = f"+{gain:.3f}" if gain > 0 else "0.000"
        if gain > 0.05: gain_str += " ***"
        
        print(f"{feature:<20} | {best_thresh:<12} | {old_f1:.3f}        | {best_f1:.3f}        | {gain_str}")

    print("-" * 75)
    print(f"Macro F1 Improvement: {total_old_f1/len(features):.3f} -> {total_new_f1/len(features):.3f}")
    
    # 7. SAVE BEST THRESHOLDS
    output_path = os.path.join(args.model_dir, "optimal_thresholds.json")
    with open(output_path, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"\nSaved optimal thresholds to {output_path}")

if __name__ == "__main__":
    main()
