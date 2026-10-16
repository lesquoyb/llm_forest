"""
Apprentissage en contexte (few-shot) sur LAMBADA : au lieu de fine-tuner, on
place k passages d'entraînement complets (contexte + mot cible) devant la
question.

    <passage 1><sép><passage 2><sép>...<sép><contexte de la question> -> mot cible

Le prompt passe par les embeddings d'entrée comme d'habitude : le masque de
dimensions d'un membre (MaskedLM.masked_dims) s'applique à tout le prompt,
démonstrations comprises, et à la notation. Aucun poids n'est modifié.

Choix des démonstrations (IclConfig.strategy) :
- "fixed"   : k passages tirés au hasard une fois par membre, les mêmes pour
              toutes les questions (l'équivalent d'un petit jeu
              d'entraînement propre à chaque membre) ;
- "random"  : k passages tirés au hasard pour chaque question ;
- "similar" : les k passages les plus proches de la question (TF-IDF sur les
              tokens), le plus proche juste avant la question. Déterministe :
              les membres ne diffèrent alors que par leur masque.

Un ensemble ("forêt sans entraînement") = plusieurs membres qui partagent les
poids du modèle mais diffèrent par leurs démonstrations et/ou leur masque.

Usage :
    config = IclConfig(shots=4, strategy="random", mask_fraction=0.125)
    members, corpora, stats = make_icl_members(model, tokenizer, train, test, config, 10)
    scores = evaluate(members, None, member_corpora=corpora)
"""

import math
import random
from dataclasses import asdict, dataclass

import torch

from data import Example
from finetune import MaskedLM, random_masked_dims

STRATEGIES = ("fixed", "random", "similar")


@dataclass(frozen=True)
class IclConfig:
    shots: int = 4
    strategy: str = "random"
    mask_fraction: float = 0.0
    max_length: int = 512  # prompt complet ; TinyStories a été entraîné sur des textes courts
    separator: str = "\n\n"

    def __post_init__(self):
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategy doit être dans {STRATEGIES}, pas {self.strategy!r}")

    @property
    def diverse(self):
        """Des membres différents sont-ils possibles ? Sinon un ensemble ne sert à rien."""
        return self.mask_fraction > 0 or (self.shots > 0 and self.strategy != "similar")

    def as_dict(self):
        return asdict(self)


# ============================================================
# Choix des démonstrations
# ============================================================

def _tfidf(examples, vocab_size, idf=None, with_target=False):
    """Matrice TF-IDF creuse [n_exemples, vocab], lignes normalisées (sur les tokens)."""

    rows, cols = [], []
    for i, ex in enumerate(examples):
        ids = ex.context_ids + (ex.target_ids if with_target else [])
        rows.extend([i] * len(ids))
        cols.extend(ids)

    counts = torch.sparse_coo_tensor(
        torch.tensor([rows, cols]),
        torch.ones(len(rows)),
        (len(examples), vocab_size),
    ).coalesce()

    if idf is None:
        df = torch.bincount(counts.indices()[1], minlength=vocab_size).float()
        idf = torch.log((len(examples) + 1) / (df + 1)) + 1

    values = counts.values() * idf[counts.indices()[1]]
    norms = torch.zeros(len(examples)).index_add_(0, counts.indices()[0], values**2).sqrt()
    values = values / norms[counts.indices()[0]]

    return torch.sparse_coo_tensor(counts.indices(), values, counts.shape).coalesce(), idf


def similar_demos(queries, pool, k, vocab_size):
    """Indices des k passages de `pool` les plus proches de chaque question (cosinus TF-IDF)."""

    pool_vectors, idf = _tfidf(pool, vocab_size, with_target=True)
    query_vectors, _ = _tfidf(queries, vocab_size, idf=idf)

    similarity = torch.sparse.mm(query_vectors, pool_vectors.t()).to_dense()

    return similarity.topk(k, dim=1).indices.tolist()


def build_prompt(query, demos, separator_ids, max_length):
    """
    Prompt = démonstrations + contexte de la question. On ajoute les
    démonstrations dans l'ordre tant qu'elles tiennent dans `max_length` (la
    question n'est jamais tronquée). Retourne (Example, nombre de démos utilisées).
    """

    budget = max_length - len(query.context_ids) - (len(query.target_ids) - 1)
    prefix, used = [], 0

    for demo in demos:
        block = demo.context_ids + demo.target_ids + separator_ids
        if len(prefix) + len(block) > budget:
            break
        prefix += block
        used += 1

    return Example(prefix + query.context_ids, query.target_ids, query.target_word), used


# ============================================================
# Membres
# ============================================================

def make_icl_members(model, tokenizer, pool, queries, config, n_members=1, seed=0):
    """
    Construit `n_members` membres (MaskedLM qui partagent `model`) et, pour
    chacun, son corpus de prompts.

    pool / queries : listes d'`Example` (corpus préparés). Pour une même
    graine, les masques sont les mêmes quelles que soient les démonstrations.

    Retourne (members, corpora, stats) ; stats donne le nombre moyen de
    démonstrations réellement utilisées et la longueur moyenne des prompts.
    """

    separator_ids = tokenizer(config.separator, add_special_tokens=False)["input_ids"]
    hidden_size = model.config.hidden_size
    generator = torch.Generator().manual_seed(seed)

    neighbours = None
    if config.shots > 0 and config.strategy == "similar":
        # Le plus proche en dernier, juste avant la question.
        neighbours = [
            list(reversed(idx))
            for idx in similar_demos(queries, pool, config.shots, len(tokenizer))
        ]

    members, corpora, used, lengths = [], [], [], []

    for i in range(n_members):
        masked_dims = random_masked_dims(hidden_size, config.mask_fraction, generator)
        rng = random.Random(seed * 100_003 + i)

        if config.shots == 0:
            demo_lists = [[] for _ in queries]
        elif config.strategy == "fixed":
            fixed = rng.sample(range(len(pool)), config.shots)
            demo_lists = [fixed for _ in queries]
        elif config.strategy == "random":
            demo_lists = [rng.sample(range(len(pool)), config.shots) for _ in queries]
        else:
            demo_lists = neighbours

        corpus = []
        for query, idx in zip(queries, demo_lists):
            example, n_used = build_prompt(
                query, [pool[j] for j in idx], separator_ids, config.max_length
            )
            corpus.append(example)
            used.append(n_used)
            lengths.append(len(example.context_ids))

        members.append(MaskedLM(model, tokenizer, masked_dims))
        corpora.append(corpus)

    stats = {
        "shots_used_mean": sum(used) / len(used),
        "prompt_length_mean": sum(lengths) / len(lengths),
        "prompt_length_max": max(lengths),
    }

    return members, corpora, stats


def log_mean(values):
    """Moyenne géométrique (pour des perplexités)."""
    return math.exp(sum(math.log(v) for v in values) / len(values))
