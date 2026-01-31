
"""
nlprun -q jag -p standard -r 8G -c 2 \
  -n multilabel_modernbert_eval \
  -o slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation/Encoder-Only/BERT && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python eval.py CGEdit AAE FullTest_Final SociauxLing/answerdotai_ModernBERT-large_CGEdit_AAE_no-wandb-20260130-133308"
"""
import transformers
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys
import os
import numpy as np
from transformers import AutoConfig, AutoModel, AutoTokenizer

# --- 1. Re-define the EXACT classes from train.py ---

class MultitaskModelConfig(transformers.PretrainedConfig):
    model_type = "multitask_model"

class MultitaskModel(transformers.PreTrainedModel):
    config_class = MultitaskModelConfig

    def __init__(self, encoder, taskmodels_dict, config):
        super().__init__(config)
        self.encoder = encoder
        # The key is that taskmodels_dict keys must match what was saved.
        # When loading from config, we usually need to reconstruct the heads.
        # But 'from_pretrained' handles state_dict loading if structure matches.
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    @classmethod
    def create(cls, model_name, head_type_list):
        # We don't need this for inference, but keeping it for reference
        pass

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # ModernBERT uses last_hidden_state
        cls_repr = out.last_hidden_state[:, 0, :]  # (B, H)
        
        logits_per_task = []
        # Crucial: We must iterate in the SAME ORDER as trained/saved.
        # The ModuleDict is ordered by insertion in Python 3.7+, but we need to match keys.
        # We will iterate over self.taskmodels_dict.keys() which should be preserved.
        for task_name in self.taskmodels_dict.keys():
            head = self.taskmodels_dict[task_name]
            logits_per_task.append(head(cls_repr))  # (B, 2)
            
        # stack into (B, num_tasks, 2)
        logits = torch.stack(logits_per_task, dim=1)
        return logits

# Register classes so AutoModel knows them (just in case)
AutoConfig.register("multitask_model", MultitaskModelConfig)
AutoModel.register(MultitaskModelConfig, MultitaskModel)

# --- 2. Helper to reconstruct the model instance ---

def load_multitask_model(model_id, head_list):
    """
    Manually reconstructs the MultitaskModel to ensure architecture matches.
    """
    print(f"Loading base encoder from config in {model_id}...")
    # Load config first
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    
    # We need the base encoder (ModernBERT)
    # The saved config likely has the base model type.
    # We can try loading the encoder from the original base if needed, 
    # but ideally we load weights from the checkpoint.
    
    # 1. Initialize fresh encoder (skeleton)
    # We assume the base model is "answerdotai/ModernBERT-large" based on train.py
    # But we should respect the config from the checkpoint.
    base_model_name = "answerdotai/ModernBERT-large" # hardcoded from train.py
    encoder = AutoModel.from_pretrained(base_model_name)
    
    # 2. Recreate heads
    hidden_size = encoder.config.hidden_size
    taskmodels_dict = {}
    for task_name in head_list:
        taskmodels_dict[task_name] = nn.Linear(hidden_size, 2)
        
    # 3. Instantiate our wrapper
    model = MultitaskModel(encoder=encoder, taskmodels_dict=taskmodels_dict, config=config)
    
    # 4. Load State Dict
    print(f"Loading state dict from {model_id}...")
    # This downloads model.safetensors or pytorch_model.bin
    from transformers.utils import cached_file
    try:
        model_file = cached_file(model_id, "model.safetensors")
        state_dict = torch.load(model_file) if model_file.endswith('.pt') else {} # safetensors handled below
        if model_file.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(model_file)
    except:
        # Fallback to pytorch_model.bin
        model_file = cached_file(model_id, "pytorch_model.bin")
        state_dict = torch.load(model_file, map_location="cpu")

    # Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    if len(missing) > 0:
        print("Missing keys (check if serious):", missing[:5])
    
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
    # Usage: python eval.py <gen_method> <lang> <test_file> <model_id>
    if len(sys.argv) < 5:
        print("Usage: python eval.py <gen_method> <lang> <test_file> <model_id>")
        sys.exit(1)

    gen_method = sys.argv[1]
    lang = sys.argv[2] # "AAE"
    test_file = sys.argv[3]
    MODEL_ID = sys.argv[4]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Get Head List
    head_list = load_head_list(lang)
    print(f"Task Heads ({len(head_list)}): {head_list}")

    # 2. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # 3. Load Model (Reconstructing Architecture)
    model = load_multitask_model(MODEL_ID, head_list)
    model.to(device)
    model.eval()

    # 4. Prepare Data
    dataset = build_dataset(tokenizer, test_file)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 5. Inference
    os.makedirs("./data/results", exist_ok=True)
    out_name = f"{MODEL_ID.split('/')[-1]}_{gen_method}_{lang}_{os.path.basename(test_file)}_preds.tsv"
    out_path = f"./data/results/{out_name}"

    print(f"Starting inference, writing to {out_path}...")
    
    with open(out_path, "w", encoding='utf-8') as f:
        all_texts = []
        all_probs = []   # list of np arrays, each shape (num_tasks,)
        # Write Header
        header = "sentence\t" + "\t".join(head_list) + "\n"
        f.write(header)
        
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            texts = batch["text"]
            
            with torch.no_grad():
                # Forward pass returns [B, Num_Tasks, 2]
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Softmax over last dim (classes 0 vs 1) -> get prob of class 1
                probs = torch.softmax(logits, dim=-1)[:, :, 1] # [B, Num_Tasks]
                probs = probs.cpu().numpy()
                for i, text in enumerate(texts):
                    clean_text = text.replace('\t', ' ').replace('\n', ' ')
                    all_texts.append(clean_text)
                    all_probs.append(probs[i])  # shape (num_tasks,)
                    # still write to file as before
                    prob_strs = "\t".join(f"{p:.4f}" for p in probs[i])
                    f.write(f"{clean_text}\t{prob_strs}\n")

            # Write batch
            for i, text in enumerate(texts):
                # Clean text of tabs/newlines for safety
                clean_text = text.replace('\t', ' ').replace('\n', ' ')
                prob_strs = "\t".join(f"{p:.4f}" for p in probs[i])
                f.write(f"{clean_text}\t{prob_strs}\n")

            all_probs = np.stack(all_probs)
    print("Done.")
