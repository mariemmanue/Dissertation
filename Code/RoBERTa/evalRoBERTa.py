import torch
import pandas as pd
from transformers import RobertaTokenizer
from trainRoBERTa import MultitaskRoberta  # Make sure this class is accessible

FEATURE_COLUMNS = [
    "zero_poss", "zero_copula", "double_tense", "be_construction", "resultant_done", "finna", "come",
    "double_modal", "multiple_neg", "neg_inversion", "n_inv_neg_concord", "aint", "zero_3sg_pres_s",
    "is_was_gen", "zero_pl_s", "double_object", "wh_qu"
]

# Load model + tokenizer
model_path = "./models/multitask"
tokenizer = RobertaTokenizer.from_pretrained(model_path)

model = MultitaskRoberta(feature_columns=FEATURE_COLUMNS, model_name="roberta-base")
model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin", map_location=torch.device("cpu")))
model.eval()

# Prediction function
def predict(sentence):
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        max_length=64,
        truncation=True,
        padding="max_length"
    )

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

    logits = outputs["logits"].squeeze(0)
    probs = torch.sigmoid(logits)

    preds = {
        FEATURE_COLUMNS[i]: int(probs[i] > 0.5)
        for i in range(len(FEATURE_COLUMNS))
    }
    confidences = {
        FEATURE_COLUMNS[i]: round(probs[i].item(), 4)
        for i in range(len(FEATURE_COLUMNS))
    }

    return preds, confidences

# Evaluate and save to Excel
def evaluate_file(file_path="Eval.txt", output_path="Eval_Predictions.xlsx"):
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    data = []
    for i, raw_line in enumerate(lines):
        sentence = raw_line.strip()
        if not sentence or sentence.lower() == "nan":
            continue
        try:
            preds, probs = predict(sentence)
            row = {
                "sentence": sentence,
                **{f"{task}_pred": preds[task] for task in preds},
                **{f"{task}_conf": round(probs[task], 4) for task in probs},
            }
            data.append(row)
        except Exception as e:
            print(f"Skipping line {i+1} due to error: {e}")

    df = pd.DataFrame(data)
    df.to_excel(output_path, index=False)
    print(f"Predictions saved to: {output_path}")

if __name__ == "__main__":
    evaluate_file("Eval.txt", "Eval_Predictions.xlsx")