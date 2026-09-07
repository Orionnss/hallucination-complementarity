"""Prompt construction for the generator and the judge pool.

Kept in one module so the exact wording is versioned alongside the results — prompt
drift between runs would silently change the hallucination rate.
"""

from __future__ import annotations

from .datasets.base import QAItem

GENERATOR_SYSTEM = (
    "You are a helpful assistant. Answer the question directly and concisely. "
    "If you do not know the answer, say \"I don't know\"."
)


def generator_messages(item: QAItem) -> list[dict[str, str]]:
    """Chat messages for one QA item.

    The explicit "I don't know" affordance matters: without it the model is pushed to
    guess, which would inflate the hallucination rate and hollow out the INVALID class
    that the judges are supposed to detect.
    """
    if item.context:
        user = (
            f"Read the passage and answer the final question.\n\n"
            f"Passage:\n{item.context}\n\nQuestion: {item.question}"
        )
    else:
        user = f"Question: {item.question}"
    return [
        {"role": "system", "content": GENERATOR_SYSTEM},
        {"role": "user", "content": user},
    ]


JUDGE_SYSTEM = (
    "You are a strict evaluator deciding whether a model's answer to a question is "
    "factually correct, using the provided reference answers as ground truth."
)

JUDGE_TEMPLATE = """Question: {question}

Reference answer(s): {gold}

Model's answer: {answer}

Classify the model's answer into exactly one label:

- NOT_HALLUCINATED: the answer is factually consistent with a reference answer. Accept \
differences in phrasing, formatting, extra detail, or partial answers that are correct \
as far as they go.
- INVALID: the model declines to answer, says it does not know, or gives no substantive \
answer.
- HALLUCINATED: the answer asserts something that conflicts with the reference answers, \
or denies the premise of the question.

Answer with the label only, on a single line."""


def judge_messages(question: str, gold_answers: list[str], answer: str) -> list[dict[str, str]]:
    # Cap the alias list: TriviaQA questions can carry dozens of aliases, which would
    # bury the actual question and push the judge toward accepting anything.
    gold = "; ".join(gold_answers[:20]) if gold_answers else "(none provided)"
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": JUDGE_TEMPLATE.format(
                question=question, gold=gold, answer=answer.strip() or "(empty)"
            ),
        },
    ]


# --- base (non-instruction-tuned) generators ---------------------------------------
#
# A pretrained checkpoint has no chat template and does not follow instructions: given
# the instruct system prompt it continues the text rather than answering. Few-shot
# completion is the standard way to elicit QA from one, and the shots carry the format
# that the instruct prompt carries in words.
#
# The demonstrations are fixed and deliberately drawn from outside the four evaluation
# datasets, so no shot can leak an answer. One of them abstains: without it a base model
# effectively never declines, the INVALID class never fires, and the abstention findings
# would not transfer between the instruct and base runs.

BASE_HEADER = (
    "Answer each question. If you do not know the answer, say \"I don't know\".\n\n"
)

BASE_SHOTS = [
    ("What is the chemical symbol for gold?", "Au"),
    ("Which ocean lies between Africa and Australia?", "The Indian Ocean"),
    ("What was the middle name of the third person to summit K2?", "I don't know"),
    ("In what year did the Chernobyl disaster occur?", "1986"),
    ("What kind of animal is a Komodo dragon?", "A lizard"),
]

BASE_CTX_HEADER = (
    "Read each passage and answer the final question. If the passage does not say, "
    "answer \"I don't know\".\n\n"
)

#: Two short shots rather than five: CoQA passages run to ~1,400 characters, and five
#: demonstration passages would dominate the prompt and slow generation for no gain.
BASE_CTX_SHOTS = [
    ("The library opened in 1931. It was designed by Marta Rios and holds 40,000 books.",
     "Who designed it?", "Marta Rios"),
    ("Tom bought three apples and gave one to Ana.", "How many did he keep?", "Two"),
]

#: Generation stops at the first of these: a base model otherwise runs on and invents the
#: next question, and the trailing text would move the last-token position the probes read.
BASE_STOP_STRINGS = ["\nQ:", "\nPassage:", "\n\n"]


def base_prompt_text(item: QAItem) -> str:
    """Few-shot completion prompt for a generator with no chat template."""
    if item.context:
        parts = [BASE_CTX_HEADER]
        for ctx, q, a in BASE_CTX_SHOTS:
            parts.append(f"Passage: {ctx}\nQ: {q}\nA: {a}\n\n")
        parts.append(f"Passage: {item.context}\nQ: {item.question}\nA:")
        return "".join(parts)
    parts = [BASE_HEADER]
    for q, a in BASE_SHOTS:
        parts.append(f"Q: {q}\nA: {a}\n\n")
    parts.append(f"Q: {item.question}\nA:")
    return "".join(parts)
