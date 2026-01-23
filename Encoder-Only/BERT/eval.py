import transformers
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from transformers import PretrainedConfig, AutoConfig, AutoModel
import sys
import os


"""
nlprun -q jag -p standard -r 8G -c 2 \
  -n bert-eval-fixed2 \
  -o slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation/Encoder-Only/BERT && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python eval.py CGEdit AAE FullTest_Final SociauxLing/answerdotai_ModernBERT-large_CGEdit_AAE_no-wandb"

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

class MultitaskConfig(PretrainedConfig):
    model_type = "multitask"
    def __init__(self, head_type_list=None, **kwargs):
        super().__init__(**kwargs)
        self.head_type_list = head_type_list or []

class MultitaskModel(transformers.PreTrainedModel):
    config_class = MultitaskConfig
    def __init__(self, encoder, taskmodels_dict, config):
        super().__init__(config)
        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    @classmethod
    def create(cls, model_name, head_type_list):
        config = MultitaskConfig.from_pretrained(model_name, head_type_list=head_type_list)
        encoder = transformers.AutoModel.from_pretrained(model_name, config=config)
        hidden_size = config.hidden_size
        taskmodels_dict = {task: nn.Linear(hidden_size, 2) for task in head_type_list}
        return cls(encoder=encoder, taskmodels_dict=taskmodels_dict, config=config)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = out.last_hidden_state[:, 0, :]
        logits_per_task = [head(cls_repr) for head in self.taskmodels_dict.values()]
        return torch.stack(logits_per_task, dim=1)  # (B, num_tasks, 2)

# Register for HF compatibility
AutoConfig.register("multitask", MultitaskConfig)
AutoModel.register(MultitaskConfig, MultitaskModel)

def eval_dataloader(dataset, batch_size=64):
    return DataLoader(dataset, batch_size=batch_size, sampler=SequentialSampler(dataset))

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
    with open(test_f) as r:
        for line in r:
            line = line.strip()
            if len(line.split()) < 2: continue
            enc = tokenizer(line, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
            input_ids_list.append(enc["input_ids"].squeeze(0))
            attn_list.append(enc["attention_mask"].squeeze(0))
            texts.append(line)
    return CustomDataset(torch.stack(input_ids_list), torch.stack(attn_list), texts)

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

    # Model loading
    if MODEL_ID.endswith(".pt"):
        base_model = "bert-base-cased" if lang == "AAE" else "bert-base-uncased"
        tokenizer = transformers.AutoTokenizer.from_pretrained(base_model)
        model = MultitaskModel.create(base_model, head_type_list)
        checkpoint = torch.load(MODEL_ID, map_location=device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
    else:  # HF ModernBERT - ✅ Fixed loading
        tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
        config = transformers.AutoConfig.from_pretrained(MODEL_ID)
        model = MultitaskModel.create(MODEL_ID, config.head_type_list or head_type_list)  # Use config or fallback

    model.to(device)
    model.eval()

    dataset = build_dataset(tokenizer, test_file)
    dataloader = eval_dataloader(dataset)

    os.makedirs("./data/results", exist_ok=True)
    with open(f"./data/results/{out_dir}", "w") as f:
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            texts = batch["text"]
            bsz = input_ids.size(0)

            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask)  # (bsz, num_tasks, 2)
            probs = torch.softmax(logits, dim=-1)[:, :, 1]  # (bsz, num_tasks)

            for i in range(bsz):
                # Clean text like BERT eval
                decoded = tokenizer.decode(texts[i], skip_special_tokens=True)
                clean_text = " ".join(decoded.split())
                prob_strs = "\t".join(f"{float(probs[i, t]):.4f}" for t in range(num_tasks))
                f.write(f"{clean_text}\t{prob_strs}\n")
