import itertools
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def rank_sentences(sentence, top_k=10):
    words = sentence.rstrip(".").lower().split()
    permutations_count = math.factorial(len(words))
    print(f"Liczba unikalnych permutacji: {permutations_count}")

    scored = []
    permutations = set(itertools.permutations(words))
    for permutation in permutations:
        text = " ".join(permutation).capitalize() + "."
        scored.append((text, score_sentence(text)))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    examples = [
        "Babuleńka miała dwa rogate koziołki.",
        "Wiewiórki w parku zaczepiają przechodniów.",
        "Student czyta ciekawą książkę.",
    ]

    for example in examples:
        print(f"\nZdanie: {example}")
        for number, (text, score) in enumerate(rank_sentences(example), 1):
            print(f"{number}. {score:.2f}: {text}")
