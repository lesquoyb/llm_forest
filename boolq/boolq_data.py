"""
Chargement et préparation du corpus BoolQ.

Un exemple BoolQ brut ({"passage", "question", "answer"}) est transformé en
`Example` : les ids du prompt, les ids de la réponse (" yes"/" no" + EOS) et
le label booléen.
"""

from dataclasses import dataclass

import torch
from datasets import load_dataset


@dataclass(frozen=True)
class Example:
    prompt_ids: list[int]
    answer_ids: list[int]  # " yes" ou " no", suivi de EOS
    label: bool  # True = yes


# ============================================================
# Chargement
# ============================================================

def load_boolq(dev_fraction=0.1, max_train=None, max_eval=None, seed=0):
    """
    Retourne (train, dev, test) bruts.

    - train / dev : découpage du split train officiel. Le dev sert à choisir
      la meilleure epoch, pour ne pas sélectionner sur le jeu de test.
    - test : le split validation officiel (le split test de BoolQ n'a pas
      de labels).
    """

    dataset = load_dataset("google/boolq")

    split = dataset["train"].train_test_split(test_size=dev_fraction, seed=seed)
    train, dev, test = split["train"], split["test"], dataset["validation"]

    def cap(ds, n):
        return ds if n is None else ds.select(range(min(n, len(ds))))

    return cap(train, max_train), cap(dev, max_eval), cap(test, max_eval)


# ============================================================
# Tokenisation
# ============================================================

def prepare_corpus(corpus, tokenizer, max_length=256):
    """
    Tokenise un corpus BoolQ brut en liste d'`Example`.

    Format du prompt :

        Passage:
        ...

        Question:
        ...

        Answer:

    Si c'est trop long, c'est le passage qui est tronqué : la question et
    "Answer:" sont toujours conservés, sinon le modèle ne peut pas répondre.

    Un corpus déjà préparé est renvoyé tel quel.
    """

    if isinstance(corpus, list) and (not corpus or isinstance(corpus[0], Example)):
        return corpus

    def encode(texts):
        return tokenizer(texts, add_special_tokens=False)["input_ids"]

    rows = list(corpus)

    head = encode("Passage:\n")
    passages = encode([row["passage"] for row in rows])
    tails = encode(
        ["\n\nQuestion:\n" + row["question"] + "\n\nAnswer:" for row in rows]
    )

    # Espace initial volontaire (tokenisation GPT-2).
    yes_ids = encode(" yes") + [tokenizer.eos_token_id]
    no_ids = encode(" no") + [tokenizer.eos_token_id]

    examples = []

    for row, passage, tail in zip(rows, passages, tails):
        answer_ids = yes_ids if row["answer"] else no_ids

        # Séquence d'entraînement = prompt + réponse sans son dernier token.
        max_prompt_length = max_length - (len(answer_ids) - 1)
        budget = max_prompt_length - len(head) - len(tail)

        if budget >= 0:
            prompt_ids = head + passage[:budget] + tail
        else:
            prompt_ids = (head + tail)[-max_prompt_length:]

        examples.append(Example(prompt_ids, answer_ids, bool(row["answer"])))

    return examples


def yes_no_ids(tokenizer):
    """Ids du premier token de " yes" et de " no"."""

    yes_id = tokenizer(" yes", add_special_tokens=False)["input_ids"][0]
    no_id = tokenizer(" no", add_special_tokens=False)["input_ids"][0]

    return yes_id, no_id


# ============================================================
# Collate
# ============================================================

def collate(batch, pad_token_id, with_answer=True):
    """
    Construit un batch paddé à droite.

    Convention (sans décalage caché) : labels[i, t] est le token que les
    logits à la position t doivent prédire. Les logits à la position
    len(prompt) - 1 prédisent donc le premier token de la réponse.

    with_answer=False : prompt seul, pour l'inférence.
    """

    sequences, labels = [], []

    for ex in batch:
        if with_answer:
            ids = ex.prompt_ids + ex.answer_ids[:-1]
            lab = [-100] * (len(ex.prompt_ids) - 1) + ex.answer_ids
        else:
            ids = ex.prompt_ids
            lab = [-100] * len(ids)

        sequences.append(ids)
        labels.append(lab)

    max_len = max(len(ids) for ids in sequences)

    def pad(rows, value):
        return torch.tensor(
            [row + [value] * (max_len - len(row)) for row in rows],
            dtype=torch.long,
        )

    return {
        "input_ids": pad(sequences, pad_token_id),
        "attention_mask": pad([[1] * len(ids) for ids in sequences], 0),
        "labels": pad(labels, -100),
        # Position qui prédit le premier token de la réponse.
        "answer_pos": torch.tensor([len(ex.prompt_ids) - 1 for ex in batch]),
        "is_yes": torch.tensor([ex.label for ex in batch]),
    }
