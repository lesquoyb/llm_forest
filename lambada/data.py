"""
Chargement et préparation du corpus LAMBADA.

Tâche : prédire le dernier mot d'un passage. Un passage brut ({"text"}) est
coupé en contexte + mot cible, puis tokenisé en `Example`.

Deux formats (load_lambada(fmt=...)) :
- "cimec" : cimec/lambada, texte en minuscules et pré-tokenisé (`do n't`,
  `` `` ``, `-lrb-`), avec un split validation qui sert d'entraînement ;
- "openai" : EleutherAI/lambada_openai, config "en" (le dépôt contient aussi
  des traductions de, es, fr, it), en texte propre. Ce sont les mêmes 5 153
  passages que le test de cimec, dans le même ordre, mais sans split
  validation : on s'entraîne sur la moitié "a" du test et on évalue sur la
  moitié "b" (les mêmes passages que la moitié "b" de cimec).
"""

import re
from dataclasses import dataclass

import torch
from datasets import load_dataset


@dataclass(frozen=True)
class Example:
    context_ids: list[int]
    target_ids: list[int]  # " mot", un ou plusieurs tokens
    target_word: str


# ============================================================
# Chargement
# ============================================================

def load_lambada(
    dev_fraction=0.1,
    max_train=None,
    max_eval=None,
    seed=0,
    test_half=None,
    fmt="cimec",
    openai_dev_size=300,
):
    """
    Retourne (train, dev, test) bruts.

    - train / dev : découpage du split validation officiel (4 869 passages).
      Le split train de LAMBADA contient des romans entiers, pas des
      paires passage / mot cible.
    - test : le split test officiel (5 153 passages).

    test_half : None = tout le test ; "a" ou "b" = une moitié fixe du test
    (même découpage quelle que soit `seed`). Le dev vient des mêmes romans que
    le train, ce qui le rend plus facile que le test : pour choisir un
    hyperparamètre, on utilise la moitié "a" et on garde la moitié "b" pour
    l'évaluation finale.

    fmt="openai" : train / dev = la moitié "a" du test en texte propre, dont
    `openai_dev_size` passages mis de côté pour choisir l'epoch ; test = la
    moitié "b" (test_half est alors ignoré, il vaut toujours "b").
    """

    def cap(ds, n):
        return ds if n is None else ds.select(range(min(n, len(ds))))

    if fmt == "openai":
        test = load_dataset("EleutherAI/lambada_openai", "en")["test"]
        halves = test.train_test_split(test_size=0.5, seed=0)
        split = halves["train"].train_test_split(test_size=openai_dev_size, seed=seed)
        return cap(split["train"], max_train), cap(split["test"], max_eval), cap(halves["test"], max_eval)

    if fmt != "cimec":
        raise ValueError(f"fmt doit valoir 'cimec' ou 'openai', pas {fmt!r}")

    dataset = load_dataset("cimec/lambada")

    split = dataset["validation"].train_test_split(test_size=dev_fraction, seed=seed)
    train, dev, test = split["train"], split["test"], dataset["test"]

    if test_half is not None:
        halves = test.train_test_split(test_size=0.5, seed=0)
        test = {"a": halves["train"], "b": halves["test"]}[test_half]

    return cap(train, max_train), cap(dev, max_eval), cap(test, max_eval)


# ============================================================
# Tokenisation
# ============================================================

def prepare_corpus(corpus, tokenizer, max_length=256):
    """
    Tokenise un corpus LAMBADA brut en liste d'`Example`.

    Si c'est trop long, on tronque le début du contexte : la fin du passage,
    juste avant le mot cible, est toujours conservée.

    Un corpus déjà préparé est renvoyé tel quel.
    """

    if isinstance(corpus, list) and (not corpus or isinstance(corpus[0], Example)):
        return corpus

    def encode(texts):
        return tokenizer(texts, add_special_tokens=False)["input_ids"]

    # Mot cible = après le dernier blanc, quel qu'il soit : en texte propre,
    # il est parfois précédé d'un retour à la ligne ("of\n\nPower").
    splits = [re.match(r"(.*\S)\s+(\S+)$", row["text"].strip(), re.S).groups() for row in corpus]
    contexts = encode([context for context, _ in splits])
    # Espace initial volontaire (tokenisation GPT-2).
    targets = encode([" " + word for _, word in splits])

    examples = []

    for (_, word), context_ids, target_ids in zip(splits, contexts, targets):
        # Séquence = contexte + cible sans son dernier token.
        max_context_length = max_length - (len(target_ids) - 1)
        examples.append(Example(context_ids[-max_context_length:], target_ids, word))

    return examples


# ============================================================
# Collate
# ============================================================

def collate(batch, pad_token_id):
    """
    Construit un batch paddé à droite.

    Convention (sans décalage caché) : labels[i, t] est le token que les
    logits à la position t doivent prédire. Seuls les tokens du mot cible
    ont un label.
    """

    sequences, labels = [], []

    for ex in batch:
        sequences.append(ex.context_ids + ex.target_ids[:-1])
        labels.append([-100] * (len(ex.context_ids) - 1) + ex.target_ids)

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
    }
