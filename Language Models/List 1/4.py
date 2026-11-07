import random
import requests
import unicodedata
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

MODEL_NAME = "eryk-mazus/polka-1.1b"

DEVICE = "cpu"

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

questions_url = "https://drive.google.com/uc?export=download&id=1MTAsgO8W3dnrhqNpCDZ-nml3l1epH0IJ"
answers_url = "https://drive.google.com/uc?export=download&id=1zovj8X8XVtyKLV19EZLnAX5Vz0W3Mt8c"

response_questions = requests.get(questions_url)
response_answers = requests.get(answers_url)
response_questions.raise_for_status()
response_answers.raise_for_status()

response_questions = response_questions.content.decode("utf-8")
response_answers = response_answers.content.decode("utf-8")
questions = []
answers = []
question_groups = ["yes_no", "fact", "number", "date", "person", "place", "other", "translate"]

for line in response_questions.splitlines():
    questions.append(line)

for line in response_answers.splitlines():
    answers.append(line)

def heuristic_func():
    from collections import Counter
    yes_no_answers = []

    for q, a in zip(questions, answers):
        if question_group(q) == "yes_no":
            yes_no_answers.extend(parse_answers(a))

    counts = Counter(normalize(a) for a in yes_no_answers)

    return counts.most_common(1)[0][0]


def normalize(text):
    return ''.join(
        c for c in unicodedata.normalize('NFD', text.lower())
        if unicodedata.category(c) != 'Mn'
    ).strip()


def question_group(q):
    q = normalize(q)

    if q.startswith(("ile ", "ilu ", "iloma ", "z ilu")):
        return "number"
    elif q.startswith(("w ktorym roku", "w ktorym wieku", "kiedy " )):
        return "date"
    elif q.startswith(("kto ", "czyje", "jakie imie", "jak nazywal")):
        return "person"
    elif q.startswith((
        "z ktore", "gdzie ", "przez ",
        "w ktorym kraju", "w ktorym miescie", "do ktore", "nad ktor", "na terenie", "stolica", "z ktorego panstwa", "w ktorych"
    )):
        return "place"
    elif q.startswith((
        "co oznacza", "co znaczy"
    )):
        return "translate"
    elif q.startswith(("czy ")) and q.count("czy") == 1:
        return "yes_no"
    elif q.startswith(("jak ", "ktor", "co ")):
        return "fact"
    else:
        return "other"


def parse_answers(answer):
    return [a.strip() for a in answer.split("\t") if a.strip()]


def create_few_shots(k):
    few_shot_examples = {}

    for g in question_groups:
        few_shots = []

        for q, a in zip(questions, answers):
            if question_group(q) == g:
                parsed = parse_answers(a)
                few_shots.append((q, parsed))

                if len(few_shots) >= k:
                    break

        few_shot_examples[g] = few_shots

    return few_shot_examples


def make_prompt(question, examples):
    prompt = (
        "Odpowiadaj na pytania faktograficzne "
        "w języku polskim. "
        "Podaj krótką odpowiedź, bez wyjaśnień.\n\n"
    )

    for q, variants in examples:
        prompt += (
            f"Pytanie: {q}\n"
            f"Odpowiedź: {random.choice(variants)}\n\n"
        )

    prompt += (
        f"Pytanie: {question}\n"
        "Odpowiedź:"
    )

    return prompt


@torch.no_grad()
def probability_func(question, examples, candidates):
    candidates = [c for c in set(candidates) if c.strip()]

    if not candidates:
        return ""

    prompt = make_prompt(question, examples)

    prefix = tokenizer(
        prompt, return_tensors="pt"
    )["input_ids"].to(DEVICE)

    best_answer = candidates[0]
    best_score = float("-inf")

    for answer in candidates:
        inputs = tokenizer(
            prompt + " " + answer,
            return_tensors="pt"
        ).to(DEVICE)

        logits = model(**inputs).logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        ids = inputs["input_ids"][0]
        start = prefix.shape[1] - 1

        tokens_log_prob = log_probs[0, start:-1].gather(
            1, ids[start+1:].unsqueeze(1)
        )

        score = tokens_log_prob.sum().item() / max(1, tokens_log_prob.shape[0])

        if score > best_score:
            best_score = score
            best_answer = answer

    return best_answer

@torch.no_grad()
def generate_answer(question, examples, num_answers=1):
    prompt = make_prompt(question, examples)

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(DEVICE)

    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=(num_answers > 1),
        temperature=0.7 if num_answers > 1 else 1.0,
        num_return_sequences=num_answers,
        pad_token_id=tokenizer.eos_token_id
    )

    input_length = inputs["input_ids"].shape[1]

    responses = []
    for seq in outputs:
        text = tokenizer.decode(seq[input_length:], skip_special_tokens=True)
        responses.append(text.strip().split("\n")[0].strip())

    if num_answers == 1:
        return responses[0]
    return responses


results = []
def main(k=150):
    few_shots = create_few_shots(3)
    hit, miss = 0, 0

    # sampled_indices = range(0, k)
    sampled_indices = random.sample(range(len(questions)), k)

    m = heuristic_func()

    for i  in sampled_indices:
        q = questions[i]
        q_type = question_group(q)
        print(i, q)
        if q_type == 'yes_no':
            generated_answer = m
        elif q_type == 'other':
            examples = few_shots[q_type]
            candidates = generate_answer(q, examples, num_answers=3)
            generated_answer = probability_func(q, examples, candidates)
        else:
            examples = few_shots[q_type]
            generated_answer = generate_answer(q, examples)

        results.append((q, generated_answer, answers[i]))
    print(results)
main()
