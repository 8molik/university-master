
import torch
from transformers import pipeline, set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "eryk-mazus/polka-1.1b"

set_seed(42)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0)
history = []

system_prompt = """
Jesteś Andrzejem Sapkowskim, czyli twórcą i znawcą uniwersum Wiedźmina.
Odpowiadasz krótko, naturalnie i konkretnie, zawsze na temat.
"""


def finish_at_last_period(response):
    response = " ".join(response.split()).strip()
    last_period = response.rfind(".")
    if last_period == -1:
        return ""
    return response[: last_period + 1]


def score_response(response, prompt):
    words = response.lower().split()
    score = 0
    if len(words) <= 30:
        score += 30 - len(words)

    prompt_tokens = set(tokenizer.encode(prompt, add_special_tokens=False))
    response_tokens = set(tokenizer.encode(response, add_special_tokens=False))
    if prompt_tokens.intersection(response_tokens):
        score += 10
    return score

def generate_response(user_input):
    conversation = system_prompt.strip() + "\n\n"
    for turn in history[-8:]:
        conversation += f"Uczestnik: {turn['user']}\n"
        conversation += f"Andrzej: {turn['assistant']}\n"
    conversation += (
        f"Uczestnik: {user_input}\n"
        "Andrzej:"
    )

    with torch.no_grad():
        responses = generator(
            conversation,
            max_new_tokens=50,
            temperature=0.6,
            num_return_sequences=3,
            repetition_penalty=1.2,
            top_p=0.9,
            return_full_text=False,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    candidates = [
        finish_at_last_period(item["generated_text"])
        for item in responses
    ]
    candidates = [candidate for candidate in candidates if candidate]
    if not candidates:
        return "Nie potrafię sformułować pełnej odpowiedzi."

    return max(candidates, key=lambda response: score_response(response, conversation))


def main():
    print("Witaj w chatbocie, który odpowiada na pytania dotyczące postaci z uniwersum Wiedźmina.")
    while True:
        user_input = input("Ty: ").strip()
        if user_input.lower() in ["exit", "quit", "koniec"]:
            break
        if not user_input:
            continue

        response = generate_response(user_input)
        print(f"Bot: {response}")
        history.append({"user": user_input, "assistant": response})


main()
