from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

def predict_sentence(sentence):
    tokenizer = RobertaTokenizer.from_pretrained("./models/Binary")
    model = RobertaForSequenceClassification.from_pretrained("./models/Binary")
    model.eval()

    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits).squeeze().tolist()
        preds = [int(p > 0.5) for p in probs]

    print(f"Sentence: {sentence}")
    print(f"Prediction: Habitual be: {preds[0]} | Negative concord: {preds[1]}")
    print(f"Probabilities: {probs}")

# Try it out
predict_sentence("She be working late.")
predict_sentence("She is going to be working late.")
predict_sentence("She is going to be late.")

predict_sentence("She ain't going to be late.")
predict_sentence("There ain't no way she is going to be late.")
predict_sentence("She ain't ever going to be late.")
predict_sentence("She ain't never going to be late.")
