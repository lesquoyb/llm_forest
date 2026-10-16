"""
Random LLM Forest sur LAMBADA : l'équivalent d'un random forest, avec un
petit LM fine-tuné à la place de chaque arbre.

Chaque membre :
- part du même modèle de base (comme chaque arbre part de zéro) ;
- est entraîné sur un échantillon bootstrap du corpus (bagging) ;
- ignore un sous-ensemble aléatoire de dimensions de l'embedding, en entrée
  et en notation. C'est l'équivalent du tirage aléatoire de features ;
- choisit sa meilleure epoch sur ses exemples out-of-bag, sans toucher au test.

Prédiction du mot cible :
- vote "soft" : moyenne des distributions de probabilité des membres ;
- vote "hard" : vote à la majorité, token par token, sur l'argmax des membres.
"""

import random

import torch

from data import prepare_corpus
from finetune import evaluate, finetune, move_parameters, random_masked_dims


class RandomLLMForest:

    def __init__(
        self,
        base_model,
        tokenizer,
        n_estimators=5,
        mask_fraction=0.25,
        bootstrap=True,
        early_stopping=True,
        offload=False,
        max_oob_samples=1000,
        max_length=256,
        seed=0,
        verbose=True,
        **finetune_kwargs,
    ):
        """
        - mask_fraction : proportion des dimensions de l'embedding neutralisées
          pour chaque membre (0.25 -> 16 dimensions sur 64).
        - max_oob_samples : nombre maximal d'exemples out-of-bag utilisés pour
          choisir l'epoch de chaque membre (limite le coût de validation).
        - bootstrap : False = chaque membre voit tout le corpus (seule la graine
          change l'ordre des données).
        - early_stopping : chaque membre garde sa meilleure epoch (loss sur
          ses exemples out-of-bag, ou sur val_corpus sans bootstrap). False =
          tous les membres font exactement `epochs` epochs. Pour la forêt,
          False est meilleur : un membre un peu surentraîné reste utile dans
          l'ensemble (voir forest_epochs.py).
        - offload : garde les membres déjà entraînés en mémoire vive (CPU)
          pendant l'entraînement des suivants, pour les gros modèles.
          Appeler `to(device)` avant d'évaluer.
        - mask_fraction=0 : aucun masque.
        - finetune_kwargs : transmis à `finetune` (epochs, lr, batch_size...).
        """

        self.base_model = base_model
        self.tokenizer = tokenizer
        self.n_estimators = n_estimators
        self.mask_fraction = mask_fraction
        self.bootstrap = bootstrap
        self.early_stopping = early_stopping
        self.offload = offload
        self.max_oob_samples = max_oob_samples
        self.max_length = max_length
        self.seed = seed
        self.verbose = verbose
        self.finetune_kwargs = finetune_kwargs

        self.members = []

    @property
    def train_time_s(self):
        return sum(member.train_time_s for member in self.members)

    def fit(self, corpus, val_corpus=None):
        """
        val_corpus : utilisé pour choisir l'epoch quand bootstrap=False
        (avec bootstrap, ce sont les exemples out-of-bag).
        """

        # Tokenisé une seule fois, partagé par tous les membres.
        data = prepare_corpus(corpus, self.tokenizer, self.max_length)
        hidden_size = self.base_model.config.hidden_size

        rng = random.Random(self.seed)
        generator = torch.Generator().manual_seed(self.seed)

        self.members = []

        for i in range(self.n_estimators):
            masked_dims = random_masked_dims(hidden_size, self.mask_fraction, generator)

            if self.bootstrap:
                indices = [rng.randrange(len(data)) for _ in range(len(data))]
                oob = sorted(set(range(len(data))) - set(indices))
                rng.shuffle(oob)
                train_data = [data[j] for j in indices]
                val_data = [data[j] for j in oob[:self.max_oob_samples]] or None
            else:
                train_data = data
                val_data = (
                    prepare_corpus(val_corpus, self.tokenizer, self.max_length)
                    if val_corpus is not None
                    else None
                )

            if not self.early_stopping:
                val_data = None

            if self.verbose:
                print(
                    f"\n[Membre {i + 1}/{self.n_estimators}] "
                    f"{len(masked_dims)}/{hidden_size} dimensions neutralisées : "
                    f"{list(masked_dims)}"
                )

            member = finetune(
                self.base_model,
                self.tokenizer,
                train_data,
                val_data,
                masked_dims=masked_dims,
                max_length=self.max_length,
                seed=self.seed + i,
                copy_model=True,
                verbose=self.verbose,
                desc=f"Membre {i + 1}",
                **self.finetune_kwargs,
            )
            if self.offload:
                move_parameters(member.model, "cpu")
            self.members.append(member)

        return self

    def to(self, device):
        """Déplace les poids de tous les membres (voir `offload`)."""

        for member in self.members:
            move_parameters(member.model, device)

        return self

    def evaluate(self, corpus, batch_size=32):
        """Scores de chaque membre et des votes soft / hard (voir finetune.evaluate)."""

        return evaluate(self.members, corpus, batch_size, self.max_length)
