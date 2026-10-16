"""
Fine-tuning d'un LM causal sur LAMBADA, avec neutralisation optionnelle de
dimensions de l'embedding.

Neutraliser une dimension d, c'est :

- en entrée : mettre à zéro la colonne d des embeddings de tokens ;
- en sortie ("notation") : mettre à zéro la dimension d du dernier état caché
  avant lm_head. Les logits ne dépendent donc plus de la colonne d de la
  matrice de sortie (qui est la même que wte, car les poids sont liés).

Le modèle lit et note les tokens uniquement dans le sous-espace des
dimensions conservées. Les couches internes restent pleine largeur.

Métriques :
- accuracy : le mot cible est juste si chacun de ses tokens est l'argmax
  du modèle (c'est exactement ce que produirait un décodage glouton) ;
- perplexité : exp(NLL moyenne du mot cible), par mot et non par token.

Usage :

    tuned = finetune(model, tokenizer, train_corpus, val_corpus)
    tuned = finetune(model, tokenizer, train_corpus, masked_dims=[3, 17, 42])
    scores = evaluate([tuned], test_corpus)
"""

import copy
import math
import time
from dataclasses import dataclass, field
from functools import partial

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import collate, prepare_corpus


# ============================================================
# Forward avec dimensions neutralisées
# ============================================================

def make_dim_mask(masked_dims, hidden_size, device):
    """Vecteur [hidden_size] : 0 sur les dimensions neutralisées, 1 ailleurs."""

    if not masked_dims:
        return None

    mask = torch.ones(hidden_size, device=device)
    mask[list(masked_dims)] = 0.0

    return mask


def random_masked_dims(hidden_size, mask_fraction, generator=None):
    """Tire au hasard round(mask_fraction * hidden_size) dimensions à neutraliser."""

    n_masked = round(mask_fraction * hidden_size)
    perm = torch.randperm(hidden_size, generator=generator)

    return tuple(sorted(perm[:n_masked].tolist()))


def masked_hidden_states(model, input_ids, attention_mask, dim_mask):
    """
    Embedding explicite -> transformer -> dernier état caché, avec le masque
    appliqué avant et après.
    """

    inputs_embeds = model.get_input_embeddings()(input_ids)

    if dim_mask is not None:
        inputs_embeds = inputs_embeds * dim_mask

    hidden = model.base_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
    ).last_hidden_state

    if dim_mask is not None:
        hidden = hidden * dim_mask

    return hidden


def target_logits(model, input_ids, attention_mask, labels, dim_mask):
    """
    Logits [n_tokens_cibles, vocab] aux seules positions qui ont un label :
    on évite de calculer [batch, seq, 50257] logits inutiles.
    """

    hidden = masked_hidden_states(model, input_ids, attention_mask, dim_mask)

    return model.get_output_embeddings()(hidden[labels != -100])


def clone_model(model):
    """
    Copie profonde qui partage les buffers au lieu de les dupliquer.

    Les buffers de GPT-Neo sont des masques d'attention causaux constants
    (8 x 4 Mo), plus lourds que les poids eux-mêmes (15 Mo) : les partager
    divise par trois la mémoire d'une forêt. À ne pas utiliser pour un modèle
    dont les buffers évoluent (ex. BatchNorm).
    """

    return copy.deepcopy(model, memo={id(b): b for b in model.buffers()})


def move_parameters(model, device):
    """
    Déplace les poids d'un modèle, mais pas ses buffers : ceux partagés par
    clone_model restent où ils sont (et partagés). Sert à garder des membres
    de forêt entraînés en mémoire vive, puis à les ramener sur le GPU.
    """

    for param in model.parameters():
        param.data = param.data.to(device)

    return model


def _device_of(model):
    return next(model.parameters()).device


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ============================================================
# Modèle (éventuellement) masqué
# ============================================================

@dataclass
class MaskedLM:
    """Un LM causal et les dimensions d'embedding qu'il ignore."""

    model: torch.nn.Module
    tokenizer: object
    masked_dims: tuple = ()
    history: dict = field(default_factory=dict)
    train_time_s: float = 0.0

    def dim_mask(self):
        return make_dim_mask(
            self.masked_dims, self.model.config.hidden_size, _device_of(self.model)
        )


# ============================================================
# Évaluation (un modèle seul ou un ensemble)
# ============================================================

def _correct_words(predicted, targets, word_index, n_words):
    """
    Nombre de mots justes pour chaque ligne de `predicted` [n_ensembles, n_tokens].
    Un mot est juste si aucun de ses tokens n'est faux.
    """

    wrong = torch.zeros(predicted.shape[0], n_words, device=targets.device)
    wrong.index_add_(1, word_index, (predicted != targets).float())

    return (wrong == 0).sum(1)


def _hard_votes(predictions, membership):
    """
    Vote à la majorité, token par token.

    predictions [n_membres, n_tokens] : argmax de chaque membre.
    membership [n_ensembles, n_membres] : 1 si le membre fait partie de l'ensemble.
    Retourne [n_ensembles, n_tokens]. En cas d'égalité, le plus petit id gagne.
    """

    same = (predictions[:, None, :] == predictions[None, :, :]).float()  # [i, j, token]
    votes = torch.einsum("ei,ijt->ejt", membership, same)  # voix pour la prédiction du membre j
    votes = votes.masked_fill(membership[:, :, None] == 0, -1)

    # Plus de voix d'abord, puis plus petit id (les ids tiennent sur 17 bits).
    score = votes.long() * (1 << 17) - predictions[None]
    winner = score.argmax(1)  # [n_ensembles, n_tokens]

    return predictions.gather(0, winner)


@torch.no_grad()
def evaluate(
    members,
    corpus,
    batch_size=32,
    max_length=256,
    subsets=None,
    chunk=32,
    member_corpora=None,
):
    """
    Évalue un ou plusieurs `MaskedLM` sur le même corpus, batch par batch.

    Retourne :
    - "members" : les scores de chaque modèle seul ;
    - "soft" : l'ensemble qui moyenne les distributions de probabilité des
      membres (pour un seul modèle, c'est le modèle lui-même) ;
    - "hard" : l'ensemble qui vote à la majorité, token par token, sur
      l'argmax de chaque membre (en cas d'égalité, le plus petit id gagne) ;
    - "subsets" (si `subsets` est une liste de listes d'indices de membres) :
      {"soft", "hard"} pour l'ensemble formé par chaque sous-ensemble.

    member_corpora : un corpus par membre, à la place de `corpus` (qui peut
    alors valoir None). Les corpus doivent avoir les mêmes mots cibles dans le
    même ordre ; seuls les contextes changent (ex. apprentissage en contexte,
    où chaque membre a ses propres démonstrations devant la question).

    Chaque membre ne tourne qu'une fois par batch. Les distributions moyennes
    de tous les ensembles sont obtenues d'un coup par un produit matriciel
    (poids des membres x probabilités), par paquets de `chunk` ensembles, et
    ne sont jamais stockées pour tout le corpus.
    """

    tokenizer = members[0].tokenizer
    device = _device_of(members[0].model)
    masks = [member.dim_mask() for member in members]
    n_members = len(members)

    for member in members:
        member.model.eval()

    def make_loader(c):
        return DataLoader(
            prepare_corpus(c, tokenizer, max_length),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=partial(collate, pad_token_id=tokenizer.pad_token_id),
        )

    if member_corpora is None:
        loaders = [make_loader(corpus)]
    else:
        assert len(member_corpora) == n_members
        first = member_corpora[0]
        assert all(
            len(c) == len(first) and all(a.target_ids == b.target_ids for a, b in zip(c, first))
            for c in member_corpora
        ), "les corpus des membres doivent avoir les mêmes mots cibles"
        loaders = [make_loader(c) for c in member_corpora]

    # Ensemble 0 = tous les membres, puis les sous-ensembles demandés.
    groups = [list(range(n_members))] + [list(idx) for idx in (subsets or [])]
    membership = torch.zeros(len(groups), n_members, device=device)
    for g, idx in enumerate(groups):
        membership[g, idx] = 1.0
    weights = membership / membership.sum(1, keepdim=True)
    log_weights = weights.log()  # -inf hors de l'ensemble

    member_correct = torch.zeros(n_members, device=device)
    member_nll = torch.zeros(n_members, dtype=torch.float64, device=device)
    soft_correct = torch.zeros(len(groups), device=device)
    soft_nll = torch.zeros(len(groups), dtype=torch.float64, device=device)
    hard_correct = torch.zeros(len(groups), device=device)
    n_words_total, n_tokens_total = 0, 0

    for batches in zip(*loaders):
        if len(batches) == 1:
            batches = batches * n_members
        labels = batches[0]["labels"].to(device)

        is_target = labels != -100
        targets = labels[is_target]
        word_index = is_target.nonzero()[:, 0]
        n_words, n_tokens = labels.shape[0], targets.numel()

        # Les tokens cibles sont les mêmes pour tous les membres, dans le même
        # ordre, même quand leurs contextes (et donc leurs positions) diffèrent.
        log_probs = torch.stack([
            target_logits(
                m.model,
                b["input_ids"].to(device),
                b["attention_mask"].to(device),
                b["labels"].to(device),
                mask,
            ).log_softmax(-1)
            for m, mask, b in zip(members, masks, batches)
        ])  # [n_membres, n_tokens, vocab]

        predictions = log_probs.argmax(-1)  # [n_membres, n_tokens]
        target_log_probs = log_probs.gather(
            2, targets.expand(n_members, -1)[..., None]
        ).squeeze(-1)  # [n_membres, n_tokens]

        # Membres seuls.
        member_correct += _correct_words(predictions, targets, word_index, n_words)
        member_nll -= target_log_probs.sum(1).double()

        # Vote soft : log(somme_i w_i p_i(cible)) et argmax de la distribution moyenne.
        mixed_target = (log_weights[:, :, None] + target_log_probs[None]).logsumexp(1)
        soft_nll -= mixed_target.sum(1).double()

        probs = log_probs.exp().reshape(n_members, -1)
        for start in range(0, len(groups), chunk):
            mixed = (weights[start:start + chunk] @ probs).view(-1, n_tokens, log_probs.shape[-1])
            soft_correct[start:start + chunk] += _correct_words(
                mixed.argmax(-1), targets, word_index, n_words
            )

        # Vote hard.
        hard_correct += _correct_words(
            _hard_votes(predictions, membership), targets, word_index, n_words
        )

        n_words_total += n_words
        n_tokens_total += n_tokens

    def scores(correct, nll=None):
        result = {"accuracy": correct.item() / n_words_total}
        if nll is not None:
            result["loss"] = nll.item() / n_tokens_total  # par token
            result["perplexity"] = math.exp(nll.item() / n_words_total)  # par mot
        return result

    results = {
        "members": [scores(c, n) for c, n in zip(member_correct, member_nll)],
        "soft": scores(soft_correct[0], soft_nll[0]),
        "hard": scores(hard_correct[0]),
    }

    if subsets:
        results["subsets"] = [
            {"soft": scores(soft_correct[g], soft_nll[g]), "hard": scores(hard_correct[g])}
            for g in range(1, len(groups))
        ]

    return results


# ============================================================
# Fine-tuning
# ============================================================

def train_step(model, optimizer, batch, dim_mask):
    """Un pas de descente de gradient ; retourne (loss, nombre de tokens cibles)."""

    device = _device_of(model)
    optimizer.zero_grad()

    labels = batch["labels"].to(device)
    logits = target_logits(
        model,
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        labels,
        dim_mask,
    )
    loss = F.cross_entropy(logits, labels[labels != -100])

    loss.backward()
    optimizer.step()

    return loss.item(), logits.shape[0]


def finetune(
    model,
    tokenizer,
    corpus,
    val_corpus=None,
    *,
    masked_dims=(),
    epochs=5,
    lr=5e-5,
    batch_size=16,
    max_length=256,
    seed=0,
    copy_model=True,
    eval_every=None,
    verbose=True,
    desc="Fine-tuning",
):
    """
    Fine-tune `model` sur `corpus` (LAMBADA brut ou liste d'`Example`).

    La loss ne porte que sur les tokens du mot cible.

    - masked_dims : dimensions de l'embedding à neutraliser (entrée + notation).
    - val_corpus : si fourni, on garde les poids de l'évaluation qui a la
      plus faible loss de validation.
    - eval_every : nombre de pas entre deux évaluations sur val_corpus.
      None = une évaluation à la fin de chaque epoch.
    - copy_model : entraîne une copie, le modèle passé en argument n'est pas
      modifié.

    Retourne un `MaskedLM` (modèle entraîné, masque, historique, temps
    d'entraînement en secondes, évaluations de validation comprises).

    history["val_step"] et history["val_compute_s"] donnent, pour chaque
    évaluation, le nombre de pas effectués et le temps de calcul
    d'entraînement écoulé (évaluations exclues).
    """

    torch.manual_seed(seed)

    if copy_model:
        model = clone_model(model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    device = _device_of(model)
    tuned = MaskedLM(model, tokenizer, tuple(masked_dims))
    dim_mask = tuned.dim_mask()

    train_data = prepare_corpus(corpus, tokenizer, max_length)
    val_data = (
        prepare_corpus(val_corpus, tokenizer, max_length)
        if val_corpus is not None
        else None
    )

    loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate, pad_token_id=tokenizer.pad_token_id),
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = AdamW(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_step": [],
        "val_compute_s": [],
    }
    best_val_loss, best_state = float("inf"), None

    step = 0
    compute_s = 0.0  # temps d'entraînement hors évaluations

    _sync(device)
    start = last_mark = time.perf_counter()

    def validate():
        nonlocal best_val_loss, best_state, compute_s, last_mark

        _sync(device)
        compute_s += time.perf_counter() - last_mark

        val = evaluate([tuned], val_data, max_length=max_length)["soft"]
        history["val_loss"].append(val["loss"])
        history["val_acc"].append(val["accuracy"])
        history["val_step"].append(step)
        history["val_compute_s"].append(compute_s)

        message = f" | val loss {val['loss']:.4f} | val acc {val['accuracy'] * 100:.2f}%"

        if val["loss"] < best_val_loss:
            best_val_loss = val["loss"]
            best_state = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }
            message += " *"

        model.train()
        last_mark = time.perf_counter()

        return message

    for epoch in range(epochs):
        model.train()
        running_loss, running_tokens = 0.0, 0

        progress = tqdm(
            loader,
            desc=f"{desc} {epoch + 1}/{epochs}",
            disable=not verbose,
            leave=False,
        )

        for batch in progress:
            loss, n_tokens = train_step(model, optimizer, batch, dim_mask)

            step += 1
            running_loss += loss * n_tokens
            running_tokens += n_tokens
            progress.set_postfix(loss=f"{loss:.4f}")

            if val_data is not None and eval_every and step % eval_every == 0:
                message = validate()
                if verbose:
                    tqdm.write(f"{desc} pas {step}{message}")

        history["train_loss"].append(running_loss / running_tokens)
        message = f"{desc} {epoch + 1}/{epochs} | train loss {history['train_loss'][-1]:.4f}"

        if val_data is not None and not eval_every:
            message += validate()

        if verbose:
            print(message)

    if best_state is not None:
        model.load_state_dict(best_state)

    # Les gradients du dernier pas restent sinon attachés aux poids : ils pèsent
    # autant que le modèle, pour chaque membre d'une forêt.
    optimizer.zero_grad(set_to_none=True)

    _sync(device)
    tuned.train_time_s = time.perf_counter() - start
    tuned.history = history

    model.eval()

    return tuned
