import transformers
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import sys
import os

"""
nlprun -q jag -p standard -r 16G -c 1 -t 02:00:00 \
  -n modernbert_eval_CGEdit_AAE_fulltest_hf \
  -o slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation/Encoder-Only/BERT && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python eval.py \
     CGEdit \
     AAE \
     FullTest_Final \
     SociauxLing/answerdotai_ModernBERT-large_CGEdit_AAE_klcncozo"


nlprun -q jag -p standard -r 16G -c 1 -t 02:00:00 \
  -n modernbert_eval_CGEdit_AAE_fulltest \
  -o slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation/Encoder-Only/BERT && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python eval.py \
     CGEdit \
     AAE \
     FullTest_Final \
     /nlp/scr/mtano/Dissertation/Encoder-Only/Masis/models/final.pt"
"""
# args:
# 1: CGEdit or CGEdit-ManualGen
# 2: AAE or IndE
# 3: test set name (file will be ./data/<name>.csv or .txt)
# 4: HF model id (e.g. SociauxLing/modernbert-CGEdit-AAE-best)
if len(sys.argv) != 5:
    raise SystemExit(
        "Usage: python eval.py <gen_method> <lang> <test_set> <hf_model_id>"
    )

gen_method = sys.argv[1]
lang = sys.argv[2]
test_set = sys.argv[3]
MODEL_ID = sys.argv[4]   # fine-tuned model repo on Hugging Face

# output file name encodes model + dataset
model_tag = MODEL_ID.replace("/", "_")
out_dir = f"{model_tag}_{gen_method}_{lang}_{test_set}.tsv"

# try .csv first, fall back to .txt
csv_path = f"./data/{test_set}.csv"
txt_path = f"./data/{test_set}.txt"
if os.path.exists(csv_path):
    test_file = csv_path
elif os.path.exists(txt_path):
    test_file = txt_path
else:
    raise FileNotFoundError(f"Could not find ./data/{test_set}.csv or ./data/{test_set}.txt")


class MultitaskModel(transformers.PreTrainedModel):
    def __init__(self, encoder, taskmodels_dict, config):
        super().__init__(config)
        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    @classmethod
    def create(cls, model_name, head_type_list):
        config = transformers.AutoConfig.from_pretrained(model_name)
        encoder = transformers.AutoModel.from_pretrained(model_name, config=config)
        hidden_size = config.hidden_size

        taskmodels_dict = {}
        for task_name in head_type_list:
            taskmodels_dict[task_name] = nn.Linear(hidden_size, 2)

        return cls(encoder=encoder, taskmodels_dict=taskmodels_dict, config=config)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = out.last_hidden_state[:, 0, :]  # CLS
        logits = []
        for _, head in self.taskmodels_dict.items():
            logits.append(head(cls_repr))
        # [num_tasks * batch, 2]
        return torch.vstack(logits)


def eval_dataloader(eval_dataset, batch_size=64):
    sampler = SequentialSampler(eval_dataset)
    return DataLoader(eval_dataset, batch_size=batch_size, sampler=sampler)


class CustomDataset(Dataset):
    def __init__(self, input_ids, attention_mask, original_texts):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.original_texts = original_texts

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "text": self.original_texts[idx],
        }


def build_dataset(tokenizer, test_f, max_length=64):
    input_ids_list = []
    attn_list = []
    raw_texts = []

    with open(test_f) as r:
        for line in r:
            line = line.strip()
            if len(line.split()) < 2:
                continue

            enc = tokenizer(
                line,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids_list.append(enc["input_ids"].squeeze(0))
            attn_list.append(enc["attention_mask"].squeeze(0))
            raw_texts.append(line)

    input_ids = torch.stack(input_ids_list)
    attention_mask = torch.stack(attn_list)

    return CustomDataset(input_ids, attention_mask, raw_texts)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # same order as training!
    if lang == "AAE":
        head_type_list = [
            "zero-poss",
            "zero-copula",
            "double-tense",
            "be-construction",
            "resultant-done",
            "finna",
            "come",
            "double-modal",
            "multiple-neg",
            "neg-inversion",
            "n-inv-neg-concord",
            "aint",
            "zero-3sg-pres-s",
            "is-was-gen",
            "zero-pl-s",
            "double-object",
            "wh-qu",
        ]
    elif lang == "IndE":
        head_type_list = [
            "foc_self",
            "foc_only",
            "left_dis",
            "non_init_exis",
            "obj_front",
            "inv_tag",
            "cop_omis",
            "res_obj_pron",
            "res_sub_pron",
            "top_non_arg_con",
        ]
    else:
        raise ValueError("lang must be AAE or IndE")

    # Load tokenizer + fine-tuned model from Hugging Face
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)

    if MODEL_ID.endswith(".pt"):
        if lang == "AAE":
            BASE_MODEL = "bert-base-cased"
        elif lang == "IndE":
            BASE_MODEL = "bert-base-uncased"
        else:
            raise ValueError("lang must be AAE or IndE")

        model = MultitaskModel.create(BASE_MODEL, head_type_list)

        checkpoint = torch.load(MODEL_ID, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
    else:  # HF model ID
        model = MultitaskModel.from_pretrained(
            MODEL_ID, 
            trust_remote_code=True  # Required for custom models
        )

    model.to(device)
    model.eval()


    dataset = build_dataset(tokenizer, test_file, max_length=64)
    dataloader = eval_dataloader(dataset, batch_size=64)

    os.makedirs("./data/results", exist_ok=True)
    with open(f"./data/results/{out_dir}", "w") as f:
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            texts = batch["text"]

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            num_tasks = len(head_type_list)
            bsz = input_ids.size(0)
            outputs = outputs.view(num_tasks, bsz, 2)

            for i in range(bsz):
                text = texts[i]
                probs = []
                for t in range(num_tasks):
                    probs.append(str(float(outputs[t, i, 1].cpu())))
                f.write(text + "\t" + "\t".join(probs) + "\n")
