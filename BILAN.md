# Random LLM Forest — bilan des expériences

16 octobre 2026 · Baptiste

Version en ligne (avec les graphiques interactifs) : https://claude.ai/code/artifact/d948cf71-3c79-4420-ad44-9d1d9f181c2c

## Résumé

Sur LAMBADA, une forêt de 20 TinyStories-28M fine-tunés et masqués à 12,5 % atteint 38,7 % d'accuracy, contre 27,8 % pour un 28M seul. C'est le meilleur résultat obtenu.

- **L'effet d'ensemble domine.** Il fait passer le 1M de 13,0 % à 24,5 % (100 membres) et le 28M de 27,8 % à 37,6 % (20 membres, sans masque).
- **Le masque aide un peu, jusqu'à 12,5 %.** Il apporte +0,3 à +0,5 point au 1M et +1,0 point au 28M. À 25 %, il fait perdre de la qualité.
- **La largeur compte plus que le nombre.** Un seul 28M bat 100 membres de 1M.
- **Ce qui ne marche pas :** masquer pour raccourcir le fine-tuning, l'apprentissage en contexte (des exemples dans le prompt), et BoolQ.
- **Limite principale :** une seule graine par expérience. Les écarts dus au masque restent proches du bruit (environ ±0,8 point).

## Méthode

**Fine-tuning avec masque.** Neutraliser une dimension, c'est mettre sa colonne à zéro dans les embeddings d'entrée et dans le dernier état caché, avant le calcul des scores. Le score ne dépend alors plus du tout de cette dimension. Les couches internes restent pleine largeur, et le calcul coûte le même temps.

**Random LLM Forest.** Chaque membre part du même modèle de base. Il s'entraîne sur un tirage aléatoire avec remise du corpus (bootstrap) et ignore un sous-ensemble aléatoire de dimensions. La prédiction est la moyenne des probabilités des membres (vote soft), ou leur vote à la majorité (vote hard). « Bagging » désigne la même forêt sans masque.

**Protocole LAMBADA.** La tâche est de prédire le dernier mot d'un passage de roman.

- **Données :** `cimec/lambada`. Entraînement sur 4 382 passages tirés du split validation officiel, et test officiel coupé en deux moitiés fixes.
- **Choix des réglages** (nombre d'epochs, notamment) **sur la moitié A du test.** Le dev vient des mêmes romans que l'entraînement, ce qui le rend plus facile.
- **Scores finaux sur la moitié B** (2 577 passages), qui n'a servi à aucun choix.
- **Un mot n'est compté juste que si tous ses tokens sont prédits.** La perplexité est calculée par mot cible.
- **Une seule graine par configuration.** Les courbes par taille de forêt sont moyennées sur 20 sous-ensembles tirés parmi les membres entraînés.

## Résultats LAMBADA

La meilleure configuration est une forêt de 20 TinyStories-28M masqués à 12,5 % : 38,7 % d'accuracy et une perplexité de 186, contre 1 362 pour le 28M seul.

Accuracy sur la moitié B du test (2 577 passages), une graine :

| Modèle | Accuracy | Entraînement |
| --- | --- | --- |
| Forêt 28M ×20, masque 12,5 % | 38,7 % | 37 min |
| Forêt 28M ×20, sans masque | 37,6 % | 37 min |
| 28M fine-tuné seul | 27,8 % | 3,7 min |
| 33M fine-tuné seul | 27,8 % | 3,9 min |
| Forêt 1M ×100, masque 12,5 % | 24,5 % | 1 h 49 |
| Forêt 1M ×100, sans masque | 24,1 % | 2 h 11 |
| 1M fine-tuné seul | 13,0 % | 1,5 min |
| Contexte : 28M, 10 masques, 0 exemple | 7,1 % | aucun |
| 28M sans fine-tuning | 5,8 % | aucun |
| 33M sans fine-tuning | 4,7 % | aucun |
| 1M sans fine-tuning | 1,8 % | aucun |

Graphiques : `lambada/outputs/single_models/comparison.png`, `lambada/outputs/forest_grid_100m/grid.png`, `lambada/outputs/forest_grid_28M/grid.png`.

Le coût suit la taille de la forêt. Vingt membres de 28M représentent environ 1 milliard de paramètres au total et 10 fois le temps d'entraînement d'un modèle seul. Les temps d'inférence mesurés varient du simple au double d'un run à l'autre, à cause du GPU du portable.

Le gain d'une forêt est fort au début, puis s'épuise : celles de 1M plafonnent vers 60 à 80 membres. Les forêts de 28M progressaient encore à 20 membres, surtout avec le masque. Les forêts de 28M dépassent le 28M seul dès 2 membres ; celles de 1M ne l'atteignent jamais.

**Apprentissage en contexte.** Placer 1 à 4 passages d'entraînement dans le prompt fait toujours baisser le score, pour les trois modèles. Pour le 28M, on passe de 5,8 % sans exemple à 2,4 à 3,1 % avec. Le seul gain vient d'un ensemble de 10 membres masqués sans exemple (7,1 %), très loin du fine-tuning.

## Ce qui marche, ce qui ne marche pas

**Ce qui marche**

- **Le bagging de petits modèles.** C'est lui qui apporte l'essentiel du gain, avant même le masque.
- **Un masque de 12,5 %.** Il donne des membres plus faibles seuls, mais plus différents entre eux, et la forêt dépasse le bagging au-delà de 7 à 10 membres. La perplexité baisse aussi, de 8 % (28M) à 18 % (1M).
- **Pas d'early stopping par membre, et plus d'epochs.** Un membre un peu surentraîné est moins bon seul mais meilleur dans l'ensemble : +3 à 4 points pour une forêt de 10 membres de 1M. Le bon nombre d'epochs augmente avec la taille de la forêt.

**Ce qui ne marche pas**

- **Masquer pour aller plus vite.** Un pas coûte le même temps, et la convergence est plus lente : à 50 % de masque, il faut 4 fois plus d'epochs pour atteindre la même perte.
- **Un masque de 25 % ou plus.** La forêt reste sous le bagging à toutes les tailles, et à 75 % elle est inutilisable.
- **Le vote hard.** Il est toujours moins bon que la moyenne des probabilités, et s'effondre à 2 membres, où chaque désaccord se départage au hasard.
- **L'apprentissage en contexte, sans fine-tuning.**

**Ce qui est intéressant**

- **La largeur compte plus que le nombre de membres.** Un seul 28M (512 dimensions) bat 100 membres de 1M (64 dimensions).
- **Le masque se comporte comme un dropout figé.** Chaque membre coupe toujours les mêmes dimensions. La diversité que ça apporte rapporte davantage sur le modèle large (+1,0 point contre +0,3 à +0,5).
- **Sans entraînement, le 28M supporte le masque, le 1M non.** Un ensemble de 10 membres masqués du 28M gagne +1,3 point sans aucun fine-tuning, alors que le masque fait tomber le 1M à 0 %.

## BoolQ

Sur BoolQ, aucun modèle ne comprend la tâche. Répondre toujours « yes » donne déjà 62,2 % sur le split validation officiel (3 270 questions).

| Modèle (TinyStories-1M) | Accuracy | Balanced accuracy | Réponses « yes » |
| --- | --- | --- | --- |
| Base, sans fine-tuning | 37,9 % | 50,0 % | 0 % |
| Fine-tuné | 62,2 % | 50,0 % | 99,9 % |
| Forêt de 5, masque 25 % | 65,7 % | 56,3 % | 90 % |
| Fine-tuné sur LAMBADA, évalué sur BoolQ | 38,8 % | 48,4 % | 10 % |

Le modèle fine-tuné a seulement appris à répondre la classe majoritaire. Les 3,5 points de la forêt sont un vrai signal, mais faible. Deux bugs du script d'origine ont été corrigés : les labels décalés deux fois et la troncature qui coupait la question. Ces résultats datent d'avant ce qu'on a appris sur LAMBADA (pas d'early stopping, plus d'epochs) et n'ont pas été refaits depuis.

## Questions ouvertes et prochaines expériences

La question centrale est de savoir si la forêt bat un modèle unique de même coût. Or tout ce qu'on a comparé jusqu'ici diffère à la fois par la forme, les données d'entraînement et le coût. Les temps ci-dessous sont des estimations pour le GPU du portable (RTX 5060, 8 Go).

| Expérience | Ce qu'elle tranche | Temps estimé | Faisable ici |
| --- | --- | --- | --- |
| Réplication avec une 2ᵉ graine | L'avantage du masque (+0,3 à +1 point) est-il réel ? | ~1 h 30 (forêts 28M) | Oui |
| Sensibilité de l'embedding | Quelles dimensions comptent, et comment choisir les masques | ~5 min (1M), ~45 min (28M) | Oui |
| Gamme Pythia fine-tunée (14M à 410M) | Où se place la forêt sur une courbe taille → qualité propre | ~2 à 3 h | Oui jusqu'à 160M ; 410M avec des astuces mémoire |
| Famille préentraînée nous-mêmes, seule la largeur varie | La comparaison à mémoire et embedding égaux | ~1 à 6 h par modèle jusqu'à ~50M | Oui jusqu'à ~50M ; non pour ~400M |

**Analyse de sensibilité de l'embedding : oui, et elle coûte peu.** On masque chaque dimension une à une et on mesure la hausse de la perte : 64 évaluations pour le 1M, 512 pour le 28M. On peut la compléter par une analyse en composantes principales de la matrice d'embedding (combien de directions portent l'essentiel de la variance). Elle servirait à trois choses :

1. **Comprendre le seuil.** Pourquoi 12,5 % passe et 25 % casse : le masque touche-t-il des dimensions critiques ?
2. **Choisir les masques au lieu de les tirer au hasard.** Par exemple, ne jamais masquer les dimensions critiques, ou répartir celles-ci entre les membres pour maximiser la diversité.
3. **Masquer dans une autre base.** Les axes bruts de l'embedding sont peut-être arbitraires. Masquer des directions principales plutôt que des coordonnées pourrait marcher mieux.

**Une gamme de modèles faite pour comparer les tailles : Pythia (EleutherAI).** De mémoire, à vérifier : 14M, 31M, 70M, 160M, 410M et au-delà, tous entraînés sur les mêmes données (The Pile), dans le même ordre, avec un même tokenizer. La largeur va de 128 dimensions (14M) à 1 024 (410M). Contrairement à TinyStories, dont les modèles changent aussi de nombre de couches, la taille y est la variable contrôlée. The Pile contient des livres, donc ces modèles partent bien mieux sur LAMBADA que TinyStories.

**Comparer la meilleure forêt de 1M à un modèle de mémoire équivalente.** « Équivalent » peut se mesurer de trois façons, qui ne désignent pas le même modèle :

- **En mémoire totale :** 100 × 3,75M = 375M paramètres (1,5 Go), comparable à Pythia-410M. Mais 85 % de chaque membre est la table d'embedding (3,2M sur 3,75M), recopiée 100 fois.
- **En paramètres hors embedding :** environ 100 × 0,5M = 50M, entre Pythia-70M et Pythia-160M.
- **En calcul par prédiction :** 100 passages d'un petit modèle, soit du même ordre qu'un passage de Pythia-160M à 410M.

On sait déjà qu'un seul 28M (52M paramètres) bat la forêt de 1M à 375M. La forêt de petits modèles perd donc nettement à mémoire égale. La question intéressante devient : la forêt de 28M (1 milliard de paramètres au total) bat-elle un Pythia-410M fine-tuné ? Fine-tuner Pythia-410M sur 8 Go demande une précision réduite (bf16), de geler l'embedding ou d'utiliser LoRA. Pythia-160M passe sans astuce.

**Préentraîner notre propre famille : envisageable jusqu'à ~50M, pas pour un modèle de 400M.** C'est la seule façon de faire varier uniquement la largeur, avec les mêmes couches, les mêmes données et le même embedding. Ordre de grandeur : 6 × paramètres × tokens opérations, à environ 10 000 milliards d'opérations par seconde utiles sur ce GPU, pour un corpus de ~500M tokens (la taille approximative de TinyStories).

- **Largeur 64 (taille 1M) :** ~1e16 opérations, de 20 minutes à 1 heure.
- **Largeur 512 (taille 28M) :** ~1,5e17 opérations, ~4 heures.
- **Un modèle de ~400M :** ~1e18 opérations, ~30 heures, et l'optimiseur seul dépasse les 8 Go de mémoire. Pas raisonnable sur le portable ; quelques heures sur un GPU loué.

Partir d'un embedding plus petit aide beaucoup. Un tokenizer à 8 000 mots au lieu de 50 257 divise la table d'embedding par 6, et le coût redevient dominé par les couches. Deux choix de données possibles : le split train de LAMBADA (des romans entiers, faits pour entraîner un modèle de langage), ou TinyStories. LAMBADA colle mieux à la tâche.

**Autres questions laissées ouvertes**

- **Le 28M seul est peut-être sous-estimé.** Son epoch est choisie sur la perte, alors que son accuracy culmine plus tard (32 % sur la moitié A à l'epoch 5).
- **Les forêts de 28M progressaient encore.** Elles montaient à 6 epochs et à 20 membres.
- **Pas de forêt de 33M.**
- **BoolQ n'a pas été refait** avec les réglages appris sur LAMBADA.
- **Les poids des forêts ne sont pas sauvegardés.** On peut les reproduire au bit près, mais il faut réentraîner.

## Analyse de sensibilité de l'embedding (en cours)

L'objectif est de savoir quelles dimensions de l'embedding du TinyStories-1M comptent vraiment, pour choisir les masques des forêts, ou réduire l'embedding. L'analyse porte sur le modèle avant fine-tuning, en texte propre (`lambada_openai`), sur 512 passages fixes de la moitié A.

**Pourquoi en texte propre.** Sur les mêmes 5 153 passages, la version `cimec` (minuscules, pré-tokenisée) divise les scores des modèles de base par 2,4 à 2,6, et multiplie leur perplexité par 150 à 600. Une analyse sur ce format mesurerait surtout la façon dont le modèle subit un texte inhabituel.

**Corrélations mesurées entre dimensions** (430 000 tokens de LAMBADA, pondérés par fréquence) :

- **Corrélations par paires faibles à modérées.** La corrélation médiane, en valeur absolue, vaut 0,15, la plus forte 0,51, et 9 % des paires dépassent 0,3.
- **Information concentrée dans peu de directions.** 9 composantes principales portent 50 % de la variance, 27 en portent 80 %, et le rang effectif est d'environ 19 sur 64.
- **Corrélation créée par le texte.** Sur le vocabulaire non pondéré, la corrélation médiane tombe à 0,06 : ce sont les tokens fréquents de LAMBADA qui l'induisent.

Carte des corrélations : `lambada/outputs/openai/embedding_correlations_1M.png`.

**Méthodes écartées et pourquoi**

- **Une dimension à la fois, seule.** Elle mesure l'effet autour du modèle complet, sans aucune interaction. Or les membres masquent plusieurs dimensions à la fois. Elle est gardée uniquement comme point de comparaison.
- **Sobol sur le masque aléatoire à 12,5 %.** Ce taux est un réglage trouvé sur le mauvais format, et ce plan mesure la sensibilité au masque, pas celle du modèle.
- **Gradients locaux (DGSM).** Ils sont rigoureux et gratuits, mais mesurent l'effet d'une variation infinitésimale. Supprimer une dimension est un grand changement, que le gradient prédit mal dans un modèle non linéaire.
- **Sobol sur les coordonnées brutes.** Il suppose des entrées indépendantes, ce que les corrélations ci-dessus ne permettent qu'approximativement. Il est aussi trop coûteux avec l'outil actuel (plus d'une journée).

**Méthode retenue : Morris binaire, centré sur la suppression.** Chaque trajectoire part d'un état où chaque dimension est gardée avec une probabilité q, puis supprime ou rétablit chaque dimension une fois, dans un ordre aléatoire. L'effet élémentaire d'une dimension est la hausse de la perte quand on la supprime, dans ce contexte. On en tire deux mesures :

- **µ\*** : l'importance moyenne de la dimension, avec un intervalle de confiance par bootstrap ;
- **σ** : la part d'interactions, c'est-à-dire à quel point l'effet dépend des autres dimensions supprimées.

**Réglage** : 50 trajectoires, soit 3 250 évaluations par régime, dans deux régimes : q = 0,9 (environ 10 % de dimensions déjà supprimées, proche du modèle complet) et q = 0,5 (la moitié). Si le classement des dimensions reste stable d'un régime à l'autre, les interactions comptent peu. Morris est validé sur des fonctions de test : il retrouve exactement les effets d'une fonction additive, et un σ positif uniquement sur deux dimensions en interaction.

**Outil** : l'évaluation des masques passe par paquets sur le GPU. Elle redonne exactement les pertes de `evaluate`. Les gros paquets saturaient la mémoire, car les matrices d'attention sont trop lourdes. Les passages sont maintenant triés par longueur, et la taille des paquets est choisie par une mesure de vitesse avant le run.

**Pistes suivantes**

1. **Refaire Morris dans la base des composantes principales**, pour savoir si masquer des directions principales a plus de sens que masquer des coordonnées brutes. Le rang effectif d'environ 19 le suggère.
2. **Calculer les indices de Sobol dans cette même base**, où les coordonnées sont décorrélées, pour quantifier les interactions si Morris en révèle.
3. **Refaire l'analyse sur les modèles fine-tunés**, plus tard : un membre entraîné avec son masque s'adapte, ce que l'analyse du modèle de base ne voit pas.
4. **Effets de Shapley**, seulement si une réponse rigoureuse dans la base d'origine est nécessaire. C'est coûteux : plusieurs heures pour le 1M.

## Code et fichiers

Le projet est dans `finetuningboolq/`, en deux dossiers indépendants (`boolq/` et `lambada/`). Il n'est pas versionné : un `git init` serait utile avant la suite.

| Fichier (`lambada/`) | Rôle | Résultats dans `outputs/` |
| --- | --- | --- |
| `finetune.py` | Fine-tuning avec masque, évaluation groupée de sous-ensembles | – |
| `forest.py` | Random LLM Forest (bootstrap, masque, early stopping, offload) | – |
| `main.py` | Modèle de base, modèle fine-tuné et forêt, comparés | `finetuned/` |
| `convergence.py` | Vitesse de convergence selon la taille du masque | `convergence/` |
| `forest_epochs.py` | Bon nombre d'epochs pour une forêt | `forest_epochs/`, `forest_epochs_28M/` |
| `forest_grid.py` | Forêts selon le nombre de membres et le masque (`--resume`, `--offload`) | `forest_grid_100m/`, `forest_grid_28M/` |
| `single_models.py` | Modèles seuls (28M, 33M) et tableau comparatif global | `single_models/` |
| `icl.py`, `icl_bench.py` | Apprentissage en contexte | `icl/` |
| `ensembles.py` | Forêt face à des ensembles par graines ou par bagging (test rapide seulement, jamais lancé en entier) | – |
| `sensitivity.py` | Analyse de sensibilité de l'embedding (Morris, Sobol) | `openai/sensitivity_*/` |

En route, deux fuites de mémoire GPU ont été corrigées. Les masques d'attention constants étaient dupliqués dans chaque copie du modèle, et chaque membre gardait ses gradients après l'entraînement. Elles divisaient la vitesse par 10 à 20 sur les grosses forêts.
