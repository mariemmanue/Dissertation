import transformers
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import sys
import os

gen_method = sys.argv[1]    # 'CGEdit' or 'CGEdit-ManualGen'
lang = sys.argv[2]          # 'AAE' or 'IndE'
test_set = sys.argv[3]

# pick model name by lang (same as before)
if lang == 'AAE':
    model_name = 'bert-base-cased'
elif lang == 'IndE':
    model_name = 'bert-base-uncased'
else:
    raise ValueError("lang must be 'AAE' or 'IndE'")

# this is the name train.py used
model_dir = model_name + "_" + gen_method + "_" + lang
out_dir = model_dir + "_" + test_set + ".tsv"


class MultitaskModel(transformers.PreTrainedModel):
    def __init__(self, encoder, taskmodels_dict):
        super().__init__(transformers.PretrainedConfig())
        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    @classmethod
    def create(cls, model_name, head_type_list):
        shared_encoder = transformers.AutoModel.from_pretrained(
            model_name,
            config=transformers.AutoConfig.from_pretrained(model_name)
        )
        taskmodels_dict = {}
        for task_name in head_type_list:
            head = torch.nn.Linear(768, 2)   # OK for BERT-base
            taskmodels_dict[task_name] = head
        return cls(encoder=shared_encoder, taskmodels_dict=taskmodels_dict)

    def forward(self, inputs, **kwargs):
        x = self.encoder(inputs)
        x = x.last_hidden_state[:, 0, :]
        out_list = []
        for task_name, head in self.taskmodels_dict.items():
            out_list.append(head(x))
        return torch.vstack(out_list)


def eval_dataloader(eval_dataset):
    sampler = SequentialSampler(eval_dataset)
    return DataLoader(eval_dataset, batch_size=64, sampler=sampler)


class CustomDataset(Dataset):
    def __init__(self, text):
        self.text = text

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return {"input_ids": self.text[idx]}


def testM(tokenizer, model, test_f):
    features_dict = {"input_ids": []}

    with open(test_f) as r:
        for line in r:
            if len(line.split()) < 2:
                continue
            tokenized = tokenizer.encode(
                line.strip(),
                max_length=64,
                padding="max_length",
                truncation=True,
            )
            features_dict["input_ids"].append(torch.LongTensor(tokenized))

    features_dict["input_ids"] = torch.stack(features_dict["input_ids"])
    dataset = CustomDataset(features_dict["input_ids"])

    dataloader = eval_dataloader(dataset)
    os.makedirs("./data/results", exist_ok=True)
    with open("./data/results/" + out_dir, 'w') as f:
        for _, inputs in enumerate(dataloader):
            for ex in inputs["input_ids"]:
                with torch.no_grad():
                    output = model(ex.unsqueeze(0).to(device))
                output = torch.nn.functional.softmax(output, dim=1)
                # take prob of class 1 for each head
                output = [str(float(x[1])) for x in output]
                sent = tokenizer.decode(ex).split()
                sent = [e for e in sent if e not in ['[PAD]', '[CLS]', '[SEP]']]
                f.write(" ".join(sent) + "\t" + "\t".join(output) + "\n")


if __name__ == "__main__":
    # allow csv or txt
    csv_path = "./data/" + test_set + ".csv"
    txt_path = "./data/" + test_set + ".txt"
    if os.path.exists(csv_path):
        test_file = csv_path
    elif os.path.exists(txt_path):
        test_file = txt_path
    else:
        raise FileNotFoundError(f"Could not find {csv_path} or {txt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if lang == 'AAE':
        head_type_list = [
            "zero-poss",
            "zero-copula",
            "double-tense",
            "be-construction", "resultant-done",
            "finna", "come", "double-modal",
            "multiple-neg", "neg-inversion", "n-inv-neg-concord", "aint",
            "zero-3sg-pres-s", "is-was-gen",
            "zero-pl-s",
            "double-object",
            "wh-qu"
        ]
    else:  # IndE
        head_type_list = [
            "foc_self", "foc_only", "left_dis", "non_init_exis",
            "obj_front", "inv_tag", "cop_omis", "res_obj_pron",
            "res_sub_pron", "top_non_arg_con"
        ]

    multitask_model = MultitaskModel.create(
        model_name=model_name,
        head_type_list=head_type_list
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    # 👇 THIS is the key change: load from ./models/...
    ckpt_path = f"./models/{model_dir}/final.pt"
    checkpoint = torch.load(ckpt_path, map_location=device)

    multitask_model.load_state_dict(checkpoint['model_state_dict'])
    multitask_model.to(device)
    multitask_model.eval()

    testM(tokenizer, multitask_model, test_file)
