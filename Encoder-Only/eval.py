import transformers
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys
import os
from transformers import AutoModel, AutoTokenizer, AutoConfig
from huggingface_hub import hf_hub_download

"""
nlprun -q jag -p standard -r 40G -c 2 \
  -n eval_modernbert \
  -o ModernBERT/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python eval.py SociauxLing/modernbert-CGEdit-AAE_final FullTest_Final.txt ModernBERT"
"""

# Usage: python modernbert_eval.py <repo_id> <test_file_name_with_ext>
if len(sys.argv) < 3:
    print("Usage: python modernbert_eval.py <hf_repo_id> <test_file_name>")
    sys.exit(1)

REPO_ID = sys.argv[1]
test_file_name = sys.argv[2]  # e.g., "FullTest_Final.txt" or "data.csv"
model_name = "answerdotai/ModernBERT-large"

class MultitaskModel(transformers.PreTrainedModel):
    def __init__(self, encoder, taskmodels_dict):
        super().__init__(transformers.PretrainedConfig())
        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    @classmethod
    def create(cls, model_name, head_type_list):
        taskmodels_dict = {}
        
        # 1. Config logic
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        if hasattr(config, "reference_compile"):
            config.reference_compile = False
            
        shared_encoder = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)
        
        hidden_size = config.hidden_size
        for task_name in head_type_list:
            head = torch.nn.Linear(hidden_size, 2)
            taskmodels_dict[task_name] = head
        return cls(encoder=shared_encoder, taskmodels_dict=taskmodels_dict)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = self.encoder(input_ids, attention_mask=attention_mask)
        cls_token = x.last_hidden_state[:, 0, :]
        out_list = []
        for task_name, head in self.taskmodels_dict.items():
            out_list.append(head(cls_token))
        return torch.stack(out_list, dim=1)

def load_from_hub(repo_id, head_list):
    print(f"Downloading model from {repo_id}...")
    model = MultitaskModel.create(model_name, head_list)
    
    # 2. Download Weights
    try:
        model_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
        sd = torch.load(model_path, map_location="cpu")
    except:
        model_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
        from safetensors.torch import load_file
        sd = load_file(model_path)

    # 3. Clean Keys
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("module."): k = k[7:]
        if k.startswith("_orig_mod."): k = k[10:]
        new_sd[k] = v
        
    # Resize if needed
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    vocab_size = len(tokenizer)
    current_emb_size = model.encoder.embeddings.tok_embeddings.weight.shape[0]
    if vocab_size != current_emb_size:
        print(f"Resizing embeddings {current_emb_size} -> {vocab_size}")
        model.encoder.resize_token_embeddings(vocab_size)
        
    model.load_state_dict(new_sd, strict=False)
    return model, tokenizer


def predict(model, tokenizer, test_f, clean_filename_no_ext, output_dir):
    texts = []
    print(f"Reading {test_f}...")
    
    if test_f.endswith('.csv'):
        import pandas as pd
        df = pd.read_csv(test_f)
        texts = df.iloc[:, 0].astype(str).tolist()
    else:
        with open(test_f, 'r', encoding='utf-8') as r:
            texts = [line.strip() for line in r if line.strip()]

    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
    
    class TestDS(Dataset):
        def __init__(self, enc): self.enc = enc
        def __len__(self): return len(texts)
        def __getitem__(self, i): return {k:v[i] for k,v in self.enc.items()}
    
    dataloader = DataLoader(TestDS(encodings), batch_size=32, shuffle=False)
    
    # --- OUTPUT FILENAME LOGIC ---
    os.makedirs(output_dir, exist_ok=True)
    out_name = os.path.join(output_dir, f"{REPO_ID.split('/')[-1]}_{clean_filename_no_ext}_preds.tsv")

    head_list = list(model.taskmodels_dict.keys())
    
    print(f"Writing results to {out_name}...")
    with open(out_name, 'w', encoding='utf-8') as f:
        f.write("sentence\t" + "\t".join(head_list) + "\n")
        
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            with torch.no_grad():
                logits = model(input_ids, attention_mask=attention_mask)
                probs = torch.softmax(logits, dim=-1)
                
            probs_pos = probs[:, :, 1].cpu().numpy()
            
            start_idx = batch_idx * 32
            for i, scores in enumerate(probs_pos):
                text_idx = start_idx + i
                if text_idx >= len(texts): break
                
                line_scores = "\t".join([f"{s:.4f}" for s in scores])
                clean_text = texts[text_idx].replace("\t", " ").replace("\n", " ")
                f.write(f"{clean_text}\t{line_scores}\n")

if __name__ == "__main__":
    # UPDATED USAGE: 
    # python Encoder-Only/eval.py <repo_id> <test_file> <arch_folder>
    
    if len(sys.argv) < 3:
        print("Usage: python Encoder-Only/eval.py <hf_repo_id> <test_file_name> [arch_folder]")
        print("Example: python Encoder-Only/eval.py SociauxLing/model AAE.txt ModernBERT")
        sys.exit(1)

    REPO_ID = sys.argv[1]
    test_file_name = sys.argv[2]
    
    # NEW ARGUMENT: arch_folder (default to ModernBERT if not provided)
    arch_folder = sys.argv[3] if len(sys.argv) > 3 else "ModernBERT"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    head_type_list=[
        "zero-poss", "zero-copula", "double-tense", "be-construction",
        "resultant-done", "finna", "come", "double-modal",
        "multiple-neg", "neg-inversion", "n-inv-neg-concord", "aint",
        "zero-3sg-pres-s", "is-was-gen", "zero-pl-s", "double-object", "wh-qu"
    ]

    model, tokenizer = load_from_hub(REPO_ID, head_type_list)
    model.to(device)
    model.eval()
    
    # --- UPDATED PATH LOGIC ---
    
    # 1. INPUT: Look in Datasets folder (relative to Dissertation root)
    # No more ".." since we are running FROM Dissertation
    full_data_path = os.path.join("Datasets", test_file_name)
    
    # 2. OUTPUT DIR: Encoder-Only/<arch_folder>/results/
    output_dir = os.path.join("Encoder-Only", arch_folder, "results")
    
    # 3. Clean filename for output naming
    clean_filename_no_ext = os.path.splitext(test_file_name)[0]

    # Pass the calculated output_dir to predict function
    predict(model, tokenizer, full_data_path, clean_filename_no_ext, output_dir)
    print("Done.")

