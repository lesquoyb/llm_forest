"""
Fine-tuning d'un LM causal sur BoolQ, avec neutralisation optionnelle de
dimensions de l'embedding.

Neutraliser une dimension d, c'est :

- en entrée : mettre à zéro la colonne d des embeddings de tokens
  (model.transformer.wte(input_ids)[..., d] = 0) ;
- en sortie ("notation") : mettre à zéro la dimension d du dernier état caché
  avant lm_head. Les logits ne dépendent donc plus de la colonne d de la
  matrice de sortie (qui est la même que wte, car les poids sont liés).

Le modèle lit et note les tokens uniquement dans le sous-espace des
dimensions conservées. Les couches internes restent pleine largeur.

Usage :

    tuned = finetune(model, tokenizer, train_corpus, val_corpus)
    tuned = finetune(model, tokenizer, train_corpus, masked_dims=[3, 17, 42])
    p_yes = tuned.predict_proba(test_corpus)
"""

import copy
import time
from dataclasses import dataclass, field
from functools import partial

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from boolq_data import collate, prepare_corpus, yes_no_ids


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


def answer_loss(model, hidden, labels):
    """
    Cross-entropy (vocabulaire complet) sur les tokens de la réponse.

    lm_head n'est appliqué qu'aux positions qui ont un label, pas à toute la
    séquence : on évite de calculer [batch, seq, 50257] logits inutiles.
    """

    targets = labels != -100
    logits = model.get_output_embeddings()(hidden[targets])

    return F.cross_entropy(logits, labels[targets])


def yes_no_logits(model, hidden, answer_pos, yes_id, no_id):
    """Logits [batch, 2] de (" yes", " no") à la position qui prédit la réponse."""

    h = hidden[torch.arange(hidden.shape[0], device=hidden.device), answer_pos]
    head = model.get_output_embeddings()
    ids = [yes_id, no_id]
    bias = head.bias[ids] if getattr(head, "bias", None) is not None else None

    return F.linear(h, head.weight[ids], bias)


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

    @torch.no_grad()
    def predict_proba(self, corpus, batch_size=64, max_length=256):
        """
        P(yes) pour chaque exemple, avec P(yes) = softmax(logit_yes, logit_no).

        Seul le prompt est passé au modèle. Les exemples sont triés par
        longueur pour limiter le padding, puis remis dans l'ordre d'origine.
        """

        data = prepare_corpus(corpus, self.tokenizer, max_length)
        device = _device_of(self.model)
        dim_mask = make_dim_mask(
            self.masked_dims, self.model.config.hidden_size, device
        )
        yes_id, no_id = yes_no_ids(self.tokenizer)

        self.model.eval()

        order = sorted(range(len(data)), key=lambda i: len(data[i].prompt_ids))
        p_yes = torch.empty(len(data))

        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            batch = collate(
                [data[i] for i in idx],
                self.tokenizer.pad_token_id,
                with_answer=False,
            )

            hidden = masked_hidden_states(
                self.model,
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                dim_mask,
            )
            logits = yes_no_logits(
                self.model, hidden, batch["answer_pos"].to(device), yes_id, no_id
            )

            p_yes[idx] = logits.softmax(-1)[:, 0].float().cpu()

        return p_yes


# ============================================================
# Évaluation (loss + accuracy)
# ============================================================

@torch.no_grad()
def evaluate(model, tokenizer, data, dim_mask=None, batch_size=64):
    """Loss moyenne sur la réponse et accuracy yes/no."""

    device = _device_of(model)
    yes_id, no_id = yes_no_ids(tokenizer)
    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate, pad_token_id=tokenizer.pad_token_id),
    )

    was_training = model.training
    model.eval()

    total_loss, correct = 0.0, 0

    for batch in loader:
        labels = batch["labels"].to(device)
        hidden = masked_hidden_states(
            model,
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            dim_mask,
        )

        n = labels.shape[0]
        total_loss += answer_loss(model, hidden, labels).item() * n

        logits = yes_no_logits(
            model, hidden, batch["answer_pos"].to(device), yes_id, no_id
        )
        predicted_yes = (logits[:, 0] > logits[:, 1]).cpu()
        correct += (predicted_yes == batch["is_yes"]).sum().item()

    model.train(was_training)

    return total_loss / len(data), correct / len(data)


# ============================================================
# Fine-tuning
# ============================================================

def finetune(
    model,
    tokenizer,
    corpus,
    val_corpus=None,
    *,
    masked_dims=(),
    epochs=3,
    lr=5e-5,
    batch_size=16,
    max_length=256,
    seed=0,
    copy_model=True,
    verbose=True,
    desc="Fine-tuning",
):
    """
    Fine-tune `model` sur `corpus` (BoolQ brut ou liste d'`Example`).

    - masked_dims : dimensions de l'embedding à neutraliser (entrée + notation).
    - val_corpus : si fourni, on garde les poids de l'epoch qui a la plus
      faible loss de validation.
    - copy_model : entraîne une copie, le modèle passé en argument n'est pas
      modifié.

    Retourne un `MaskedLM` (modèle entraîné, masque, historique, temps
    d'entraînement en secondes, évaluations de validation comprises).
    """

    torch.manual_seed(seed)

    if copy_model:
        model = copy.deepcopy(model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    device = _device_of(model)
    dim_mask = make_dim_mask(masked_dims, model.config.hidden_size, device)

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

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss, best_state = float("inf"), None

    _sync(device)
    start = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        progress = tqdm(
            loader,
            desc=f"{desc} {epoch + 1}/{epochs}",
            disable=not verbose,
            leave=False,
        )

        for batch in progress:
            optimizer.zero_grad()

            labels = batch["labels"].to(device)
            hidden = masked_hidden_states(
                model,
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                dim_mask,
            )
            loss = answer_loss(model, hidden, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.shape[0]
            progress.set_postfix(loss=f"{loss.item():.4f}")

        history["train_loss"].append(running_loss / len(train_data))
        message = f"{desc} {epoch + 1}/{epochs} | train loss {history['train_loss'][-1]:.4f}"

        if val_data is not None:
            val_loss, val_acc = evaluate(model, tokenizer, val_data, dim_mask)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            message += f" | val loss {val_loss:.4f} | val acc {val_acc * 100:.2f}%"

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    k: v.detach().clone() for k, v in model.state_dict().items()
                }
                message += " *"

        if verbose:
            print(message)

    if best_state is not None:
        model.load_state_dict(best_state)

    _sync(device)
    train_time_s = time.perf_counter() - start

    model.eval()

    return MaskedLM(
        model=model,
        tokenizer=tokenizer,
        masked_dims=tuple(masked_dims),
        history=history,
        train_time_s=train_time_s,
    )
