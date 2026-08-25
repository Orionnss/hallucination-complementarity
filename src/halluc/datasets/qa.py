"""Concrete QA dataset adapters.

Context policy (DESIGN.md): closed-book unless the question is undefined without a
passage. TriviaQA, NQ-Open and SQuAD v2 therefore drop their contexts; CoQA keeps its
story because its questions are anaphoric and meaningless standalone.
"""

from __future__ import annotations

import hashlib

from datasets import load_dataset

from .base import DATASETS, QADataset, QAItem


@DATASETS.register("triviaqa")
class TriviaQA(QADataset):
    """Closed-book TriviaQA. `rc.nocontext` ships without evidence documents."""

    name = "triviaqa"

    def load(self) -> list[QAItem]:
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
        # rc.nocontext emits one row per evidence document, so questions repeat ~1.8x
        # (17,944 rows -> 9,960 questions). Deduplicating by question_id is essential:
        # a repeated question straddling a fold boundary is train/test leakage.
        items: dict[str, QAItem] = {}
        for row in ds:
            qid = row["question_id"]
            answer = row["answer"]
            # The canonical `value` must stay first. Aliases are Wikipedia-derived and
            # often obscure ("Culture of Devon" for Devon); sorting them alphabetically
            # can push the real answer past the cap applied in the judge prompt, so a
            # correct answer would be judged against nonsense.
            canonical = answer["value"]
            aliases = {g for g in answer.get("aliases", []) if g and g != canonical}
            if qid in items:
                aliases |= set(items[qid].gold_answers[1:])
            items[qid] = QAItem(
                item_id=f"triviaqa:{qid}",
                question=row["question"],
                gold_answers=[canonical, *sorted(aliases)],
            )
        return list(items.values())


@DATASETS.register("nq_open")
class NQOpen(QADataset):
    """Open-domain Natural Questions. Validation is only 3,610 rows — caps the pool."""

    name = "nq_open"

    def load(self) -> list[QAItem]:
        ds = load_dataset("google-research-datasets/nq_open", split="validation")
        return [
            QAItem(
                item_id=f"nq_open:{i}",
                question=row["question"],
                gold_answers=list(row["answer"]),
            )
            for i, row in enumerate(ds)
        ]


@DATASETS.register("squad_v2")
class SquadV2NoContext(QADataset):
    """SQuAD v2 with the passage removed, turning it into closed-book recall.

    Unanswerable rows are dropped: without the passage there is no premise to deny, so
    they would produce labels that mean something different from the other datasets.
    """

    name = "squad_v2"

    def load(self) -> list[QAItem]:
        ds = load_dataset("rajpurkar/squad_v2", split="validation")
        items = []
        for row in ds:
            golds = sorted(set(row["answers"]["text"]))
            if not golds:
                continue  # unanswerable
            items.append(
                QAItem(
                    item_id=f"squad_v2:{row['id']}",
                    question=row["question"],
                    gold_answers=golds,
                    # Questions sharing a paragraph probe overlapping facts. Grouping by
                    # article instead would leave only 35 groups — too coarse to split
                    # into 5 folds without wild fold-to-fold variance.
                    group_id="squad_v2:" + hashlib.sha1(
                        row["context"].encode()
                    ).hexdigest()[:16],
                    meta={"title": row["title"]},
                )
            )
        return items


@DATASETS.register("coqa")
class CoQA(QADataset):
    """Conversational QA — the only dataset that keeps its context.

    Rows are conversations; each turn becomes one item carrying the story plus the
    preceding turns, because CoQA questions reference earlier answers ("Where did he
    go?") and are unanswerable in isolation.
    """

    name = "coqa"

    def load(self) -> list[QAItem]:
        ds = load_dataset("stanfordnlp/coqa", split="validation")
        items = []
        for conv_idx, row in enumerate(ds):
            questions = row["questions"]
            answers = row["answers"]["input_text"]
            for turn, (question, answer) in enumerate(zip(questions, answers)):
                history = "\n".join(
                    f"Q: {questions[t]}\nA: {answers[t]}" for t in range(turn)
                )
                items.append(
                    QAItem(
                        item_id=f"coqa:{conv_idx}:{turn}",
                        question=question,
                        gold_answers=[answer],
                        context=row["story"] + (f"\n\n{history}" if history else ""),
                        # All turns of a conversation share the story; splitting them
                        # across folds would leak the passage into the test set.
                        group_id=f"coqa:{conv_idx}",
                        meta={"turn": turn, "conversation": conv_idx},
                    )
                )
        return items
