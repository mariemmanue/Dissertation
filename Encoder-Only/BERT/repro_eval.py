import transformers
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys
import os
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download

# Usage: python repro_eval.py <repo_id> <test_file_name>
# Example: python repro_eval.py mtano/repro-bert-base-cased-CGEdit-AAE-run123 FullTest_Final
if len(sys.argv) < 3:
    print("Usage: python repro_eval.py <hf_repo_id> <test_file_name>")
    sys.exit(1)

REPO_ID = sys.argv[1]
test_file_name = sys.argv[2]
model_name = 'bert-base-cased' # Hardcoded base for architecture reconstruction

class MultitaskModel(transformers.PreTrainedModel):
    def __init__(self, encoder, taskmodels_dict):
        super().__init__(transformers.PretrainedConfig())
        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    @classmethod
    def create(cls, model_name, head_type_list):
        taskmodels_dict = {}
        shared_encoder = transformers.AutoModel.from_pretrained(model_name)
        for task_name in head_type_list:
            head = torch.nn.Linear(768, 2)
            taskmodels_dict[task_name] = head
        return cls(encoder=shared_encoder, taskmodels_dict=taskmodels_dict)

    def forward(self, input_ids, **kwargs):
        x = self.encoder(input_ids)
        cls_token = x.last_hidden_state[:, 0, :]
        out_list = []
        for task_name, head in self.taskmodels_dict.items():
            out_list.append(head(cls_token))
        return torch.stack(out_list, dim=1)

def load_from_hub(repo_id, head_list):
    print(f"Downloading model from {repo_id}...")
    
    # 1. Initialize Blank Model
    model = MultitaskModel.create(model_name, head_list)
    
    # 2. Download Weights
    try:
        model_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
        sd = torch.load(model_path, map_location="cpu")
    except:
        model_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
        from safetensors.torch import load_file
        sd = load_file(model_path)

    # 3. Clean Keys (Strip prefix if needed)
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("module."): k = k[7:]
        if k.startswith("_orig_mod."): k = k[10:]
        new_sd[k] = v
        
    # 4. Load
    model.load_state_dict(new_sd, strict=False)
    return model

# --- INFERENCE LOOP ---
def predict(model, tokenizer, test_f):
    texts = []
    print(f"Reading {test_f}...")
    
    if test_f.endswith('.csv'):
        import pandas as pd
        df = pd.read_csv(test_f)
        texts = df.iloc[:, 0].astype(str).tolist()
    else:
        with open(test_f, 'r', encoding='utf-8') as r:
            texts = [line.strip() for line in r if line.strip()]

    encodings = tokenizer(texts, truncation=True, padding=True, max_length=64, return_tensors='pt')
    
    class TestDS(Dataset):
        def __init__(self, enc): self.enc = enc
        def __len__(self): return len(texts)
        def __getitem__(self, i): return {k:v[i] for k,v in self.enc.items()}
    
    dataloader = DataLoader(TestDS(encodings), batch_size=64, shuffle=False)
    
    out_name = f"data/results/{REPO_ID.split('/')[-1]}_{os.path.basename(test_f)}_preds.tsv"
    os.makedirs("data/results", exist_ok=True)
    
    head_list = list(model.taskmodels_dict.keys())
    
    print(f"Writing results to {out_name}...")
    with open(out_name, 'w', encoding='utf-8') as f:
        f.write("sentence\t" + "\t".join(head_list) + "\n")
        
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            
            with torch.no_grad():
                logits = model(input_ids)
                probs = torch.softmax(logits, dim=-1)
                
            probs_pos = probs[:, :, 1].cpu().numpy()
            
            start_idx = batch_idx * 64
            for i, scores in enumerate(probs_pos):
                text_idx = start_idx + i
                if text_idx >= len(texts): break
                
                line_scores = "\t".join([f"{s:.4f}" for s in scores])
                clean_text = texts[text_idx].replace("\t", " ").replace("\n", " ")
                f.write(f"{clean_text}\t{line_scores}\n")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    head_type_list=[
        "zero-poss", "zero-copula", "double-tense", "be-construction",
        "resultant-done", "finna", "come", "double-modal",
        "multiple-neg", "neg-inversion", "n-inv-neg-concord", "aint",
        "zero-3sg-pres-s", "is-was-gen", "zero-pl-s", "double-object", "wh-qu"
    ]

    model = load_from_hub(REPO_ID, head_type_list)
    model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID) # Loads the one we saved
    
    test_path = f"./data/{test_file_name}.txt"
    predict(model, tokenizer, test_path)
    print("Done.")
