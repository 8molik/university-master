import itertools
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
import re

url = "https://drive.google.com/uc?export=download&id=1Hc4PhNpnZVFpsZFmq-nB6i6ji-0AyfDV"

response = requests.get(url)
response.raise_for_status()

text = response.content.decode("utf-8")
sentences_bias = []

parts = re.split(r'\b(GOOD|BAD)', text)

label = None
for part in parts:
    part = part.strip()

    if part == "GOOD":
        label = True
    elif part == "BAD":
        label = False
    elif part and label is not None:
        sentences_bias.append((part, label))

MODEL_NAME = "flax-community/papuGaPT2"
DEVICE = "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()


@torch.no_grad()
def score_sentence(sentence):
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        add_special_tokens=True,
    )
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    logits = model(input_ids, attention_mask=attention_mask).logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    predictions = log_probs[0, :-1, :]
    targets = input_ids[0, 1:]
    chosen = predictions.gather(
        1, targets.unsqueeze(1)
    ).squeeze(1)
    return chosen.sum().item()

endings = {
    "To była opinia pozytywna.": True,
    "To była opinia negatywna.": False
  }

if __name__ == "__main__":
    positives = 0
    negatives = 0
    correct = 0

    for sentence, true_label in sentences_bias:
        scores = []

        for ending in endings:
            sentence_to_score = sentence + " " + ending
            scores.append(score_sentence(sentence_to_score))

        best_index = scores.index(max(scores))
        best_ending = list(endings.keys())[best_index]

        predicted_label = endings[best_ending]

        if predicted_label:
            positives += 1
        else:
            negatives += 1

        if predicted_label == true_label:
            correct += 1

    total = len(sentences_bias)

    print(f"Liczba opinii pozytywnych: {positives}")
    print(f"Liczba opinii negatywnych: {negatives}")
    print(f"Liczba wszystkich opinii: {total}")

    if total > 0:
        print(f"Poprawnie sklasyfikowane: {correct}")
        print(f"Accuracy: {correct / total:.2%}")