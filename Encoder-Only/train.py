# import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch
import transformers
import torch.nn as nn
from torch.utils.data import Dataset
import sys
import os
import argparse
import shutil


"""
nlprun -q jag -p standard -r 40G -c 2 \
  -n modernbert_train \
  -o ModernBERT/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation/Encoder-Only && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python train.py CGEdit AAE"
"""

# --- ARGS ---
parser = argparse.ArgumentParser()
parser.add_argument("gen_method", type=str, help="CGEdit or CGEdit-ManualGen")
parser.add_argument("lang", type=str, help="AAE or IndE")
parser.add_argument("--wandb_project", type=str, default="modernbert")
parser.add_argument("--fix_vocab", action="store_true", help="Add AAE tokens to vocab")
args = parser.parse_args()

gen_method = args.gen_method
lang = args.lang
model_name = "answerdotai/ModernBERT-large"

# Setup WandB
import wandb
wandb.init(project=args.wandb_project, name=f"modernbert")
run_name = wandb.run.name

# Local output dir
out_dir = f"models/ModernBERT_{gen_method}_{lang}"

# --- MANUAL CLEANUP (Fixes v5.0.0 error) ---
if os.path.exists(out_dir):
    print(f"Cleaning existing directory: {out_dir}")
    shutil.rmtree(out_dir)
os.makedirs(out_dir, exist_ok=True)

# --- MODEL DEFINITION ---
class MultitaskModel(transformers.PreTrainedModel):
    def __init__(self, encoder, taskmodels_dict):
        super().__init__(transformers.PretrainedConfig())
        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    @classmethod
    def create(cls, model_name, head_type_list, fix_vocab=False):
        taskmodels_dict = {}
        
        # 1. Configure to disable compilation (CRITICAL FIX)
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        if hasattr(config, "reference_compile"):
            config.reference_compile = False

        # 2. Load Encoder
        shared_encoder = AutoModel.from_pretrained(
            model_name, 
            config=config, 
            trust_remote_code=True
        )
        
        # 3. Create Heads
        hidden_size = config.hidden_size
        for task_name in head_type_list:
            head = torch.nn.Linear(hidden_size, 2)
            taskmodels_dict[task_name] = head
        
        return cls(encoder=shared_encoder, taskmodels_dict=taskmodels_dict)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Pass through encoder
        x = self.encoder(input_ids, attention_mask=attention_mask, **kwargs)
        
        # ModernBERT CLS token is at index 0, same as BERT
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
        attention_mask = inputs.get("attention_mask", None)
        
        outputs = model(input_ids, attention_mask=attention_mask) # [B, T, 2]
        
        # Flatten logits and labels for CrossEntropy
        # logits shape: [B, T, 2] -> [B*T, 2]
        # labels shape: [B, T]    -> [B*T]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, 2), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Custom eval step that always returns (loss, logits, labels) with no grad,
        so Trainer can compute eval_loss and call compute_metrics.
        """
        # Shallow copy so we don't mutate original
        inputs = inputs.copy()

        # Extract labels
        labels = inputs.get("labels", None)

        # Eval mode, no grad
        model.eval()
        with torch.no_grad():
            # Force calculation of loss using our custom logic
            loss, logits = self.compute_loss(model, inputs, return_outputs=True)

        # Detach for numpy conversion
        if loss is not None:
            loss = loss.detach()
        if logits is not None:
            logits = logits.detach()
        if labels is not None and isinstance(labels, torch.Tensor):
            labels = labels.detach()

        if prediction_loss_only:
            return loss, None, None

        return loss, logits, labels


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
        for line in r:
            line = line.strip()
            if not line or not line.split("\t")[1].isdigit(): 
                continue
            parts = line.split("\t")
            input_ids.append(parts[0])
            labels.append([int(x) for x in parts[1:]])


    # SPLIT INTO TRAIN/DEV (80/20)
    from sklearn.model_selection import train_test_split
    train_texts, dev_texts, train_labels, dev_labels = train_test_split(
        input_ids, labels, test_size=0.2, random_state=42, stratify=None
    )
    
    print(f"Train: {len(train_texts)}, Dev: {len(dev_texts)}")
    
    # Tokenize both splits
    train_enc = tokenizer(train_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
    dev_enc = tokenizer(dev_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
    
    train_dataset = CustomDataset(train_enc, torch.tensor(train_labels))
    dev_dataset = CustomDataset(dev_enc, torch.tensor(dev_labels))

    training_args = transformers.TrainingArguments(
        output_dir=out_dir,
        report_to="wandb",
        
        # 1. LOWER LR for small dataset
        learning_rate=2e-5,
        
        # 2. STEP-BASED WARMUP
        warmup_ratio=0.06,
        
        # 3. AGGRESSIVE EARLY STOPPING
        num_train_epochs=50,
        
        # 4. ADD WEIGHT DECAY
        weight_decay=0.01,
        
        # 5. ENABLE EVALUATION & EARLY STOPPING
        eval_strategy="epoch",  # Correct argument name (older versions use evaluation_strategy)
        save_strategy="epoch",
        load_best_model_at_end=True,  # Now safe because prediction_step provides eval_loss
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        
        # 6. SMALLER BATCH
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        
        # 7. GRADIENT CLIPPING
        max_grad_norm=1.0,
        
        logging_steps=5,
        push_to_hub=True,
        hub_model_id=f"modernbert-{gen_method}-{lang}", # Fixed hub ID
        hub_strategy="end",
        remove_unused_columns=False,
    )

    trainer = MultitaskTrainer(
        model=multitask_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,  # Pass the dev set!
    )
    
    print("Starting training...")
    trainer.train()
    
    # Save & Push Final
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    torch.save(multitask_model.state_dict(), f"{out_dir}/pytorch_model.bin")
    
    trainer.push_to_hub()
    print(f"Pushed to HF Hub: {training_args.hub_model_id}")


if __name__ == "__main__":
    train_file = f"Combined_{lang}.tsv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    head_type_list=[
        "zero-poss", "zero-copula", "double-tense", "be-construction",
        "resultant-done", "finna", "come", "double-modal",
        "multiple-neg", "neg-inversion", "n-inv-neg-concord", "aint",
        "zero-3sg-pres-s", "is-was-gen", "zero-pl-s", "double-object", "wh-qu"
    ]
    
    print(f"Creating model: {model_name}")
    multitask_model = MultitaskModel.create(model_name, head_type_list)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # --- OPTIONAL: FIX VOCAB ---
    if args.fix_vocab:
        new_tokens = ["aint", "finna", "tryna", "ion", "talmbout"]
        num_added = tokenizer.add_tokens(new_tokens)
        if num_added > 0:
            print(f"Resizing embeddings: +{num_added} tokens")
            multitask_model.encoder.resize_token_embeddings(len(tokenizer))

    multitask_model.to(device)
    trainM(tokenizer, train_file)
