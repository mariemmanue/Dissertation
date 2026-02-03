import transformers
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import sys
import os
import argparse


"""
nlprun -q jag -p standard -r 24G -c 2 \
  -n repro_bert_full_pipeline \
  -o slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation/Encoder-Only/BERT && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python repro_train.py CGEdit AAE --wandb_project repro-bert"

"""

# --- ARGS ---
parser = argparse.ArgumentParser()
parser.add_argument("gen_method", type=str, help="CGEdit or CGEdit-ManualGen")
parser.add_argument("lang", type=str, help="AAE or IndE")
parser.add_argument("--wandb_project", type=str, default="repro-bert-cased")
args = parser.parse_args()

gen_method = args.gen_method
lang = args.lang

if lang == 'AAE':
    model_name = 'bert-base-cased'
elif lang == 'IndE':
    model_name = 'bert-base-uncased'

# Setup WandB
import wandb
wandb.init(project=args.wandb_project, name=f"{model_name}-{gen_method}")
run_name = wandb.run.name

# Local output dir
out_dir = f"models/{model_name}_{gen_method}_{lang}_{run_name}"

# --- MODEL DEFINITION ---
class MultitaskModel(transformers.PreTrainedModel):
    def __init__(self, encoder, taskmodels_dict):
        super().__init__(transformers.PretrainedConfig())
        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    @classmethod
    def create(cls, model_name, head_type_list):
        taskmodels_dict = {}
        # Load vanilla BERT
        shared_encoder = transformers.AutoModel.from_pretrained(model_name)
        
        # Create one head per feature
        for task_name in head_type_list:
            head = torch.nn.Linear(768, 2)
            taskmodels_dict[task_name] = head
        
        return cls(encoder=shared_encoder, taskmodels_dict=taskmodels_dict)

    def forward(self, input_ids, **kwargs):
        x = self.encoder(input_ids, **kwargs)
        cls_token = x.last_hidden_state[:, 0, :]
        out_list = []
        for task_name, head in self.taskmodels_dict.items():
            out_list.append(head(cls_token))
        return torch.stack(out_list, dim=1) # [B, T, 2]

# --- TRAINER ---
class MultitaskTrainer(transformers.Trainer):
    # Added **kwargs to handle 'num_items_in_batch' passed by newer Transformers
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        input_ids = inputs["input_ids"]
        outputs = model(input_ids) # [B, T, 2]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, 2), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# --- DATASET ---
class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

def trainM(tokenizer, train_f):
    input_ids = []
    labels = []
    print(f"Loading data from {train_f}...")
    with open(train_f, 'r', encoding='utf-8') as r:
        for i, line in enumerate(r):
            line = line.strip()
            if not line: continue
            parts = line.split("\t")
            if i==0 and not parts[1].isdigit(): continue 
            input_ids.append(parts[0])
            labels.append([int(x) for x in parts[1:]])

    print(f"Tokenizing {len(input_ids)} examples...")
    encodings = tokenizer(input_ids, truncation=True, padding=True, max_length=64, return_tensors='pt')
    dataset = CustomDataset(encodings, torch.tensor(labels))

    # REPRO HYPERPARAMS
    training_args = transformers.TrainingArguments(
        output_dir=out_dir,
        # overwrite_output_dir=False,
        report_to="wandb",
        learning_rate=1e-4,     # Authors' LR
        do_train=True,
        warmup_steps=300,       # Authors' Warmup
        num_train_epochs=500,   # Authors' Epochs
        per_device_train_batch_size=64,
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=50,
        push_to_hub=True,       # <--- ENABLE HUB PUSH
        hub_model_id=f"repro-{model_name}-{gen_method}-{lang}-{run_name}", # Unique Repo Name
        hub_strategy="every_save",
        remove_unused_columns=False
    )

    trainer = MultitaskTrainer(
        model=multitask_model,
        args=training_args,
        train_dataset=dataset,
    )
    
    print("Starting training...")
    trainer.train()
    
    # Save & Push Final
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    
    # Manually save the state_dict cleanly for eval.py to load
    # (Trainer saves safely, but let's be explicit for our custom class)
    torch.save(multitask_model.state_dict(), f"{out_dir}/pytorch_model.bin")
    
    trainer.push_to_hub()
    print(f"Pushed to HF Hub: {training_args.hub_model_id}")

if __name__ == "__main__":
    train_file = f"./{gen_method}/{lang}.tsv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    head_type_list=[
        "zero-poss", "zero-copula", "double-tense", "be-construction",
        "resultant-done", "finna", "come", "double-modal",
        "multiple-neg", "neg-inversion", "n-inv-neg-concord", "aint",
        "zero-3sg-pres-s", "is-was-gen", "zero-pl-s", "double-object", "wh-qu"
    ]
    
    print(f"Creating model: {model_name}")
    multitask_model = MultitaskModel.create(model_name, head_type_list)
    multitask_model.to(device)

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    trainM(tokenizer, train_file)
