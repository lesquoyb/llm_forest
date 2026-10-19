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

## Reprise en texte propre (octobre)

Les résultats ci-dessous en format `cimec` sont tirés vers le bas par le format du texte. En texte propre (`lambada_openai`), le 28M fine-tuné seul atteint 36,2 %, et une forêt de 50 TinyStories-1M masqués à 6,25 % atteint 27,2 %.

**Le format pèse lourd.** `cimec` est en minuscules et pré-tokenisé (`do n't`, guillemets remplacés par des accents graves). Sur les mêmes 5 153 passages, les modèles de base font 2,4 à 2,6 fois mieux en texte propre, et leur perplexité est divisée par 150 à 600. Une raison : sans majuscules, les noms propres se découpent en plus de tokens (45 % des mots cibles font plusieurs tokens, contre 22 %).

| Modèle, sans fine-tuning | `cimec` | Texte propre |
| --- | --- | --- |
| TinyStories-1M | 1,7 % | 4,0 % |
| TinyStories-28M | 5,7 % | 14,6 % |
| TinyStories-33M | 4,6 % | 11,7 % |

**Protocole.** `lambada_openai` ne contient que les 5 153 passages du test. On les coupe avec le même découpage qu'avant :

- **moitié A** : 2 076 passages d'entraînement et 500 de sélection ;
- **moitié B** : test final, avec les mêmes passages que la moitié B de `cimec`.

Le dépôt contient aussi des traductions (de, es, fr, it) ; on utilise la configuration anglaise, vérifiée passage par passage.

**Réglages refaits sur la sélection** (TinyStories-1M) :

- **Epochs** : à 10 membres, l'accuracy plafonne vers 12 à 15 epochs, pour tous les masques. On garde 15 epochs, sans early stopping par membre. Avec early stopping, la forêt reste bloquée vers 17 %.
- **Masque × taille** : plus le masque est grand, plus il faut de membres pour qu'il paie. Le masque de 6,25 % dépasse le bagging vers 20 membres, celui de 12,5 % vers 30 à 40. Le bagging plafonne dès 30 membres, alors que les forêts masquées montaient encore à 50.

| Membres (sélection, 500 passages) | 1 | 5 | 10 | 20 | 30 | 50 |
| --- | --- | --- | --- | --- | --- | --- |
| Sans masque (bagging) | 14,6 % | 22,1 % | 23,9 % | 24,6 % | 25,0 % | 24,8 % |
| Masque 6,25 % | 13,4 % | 21,1 % | 23,3 % | 24,9 % | 25,5 % | 26,6 % |
| Masque 12,5 % | 13,0 % | 19,8 % | 22,5 % | 23,9 % | 25,0 % | 26,2 % |
| Masque 25 % | 11,1 % | 18,5 % | 21,4 % | 23,5 % | 24,2 % | 25,2 % |

**Test final, moitié B** (configuration choisie sur la sélection) :

| Modèle | Accuracy | Perplexité |
| --- | --- | --- |
| 1M fine-tuné seul | 15,8 % | 11 297 |
| Forêt 1M ×50, sans masque | 26,9 % | 1 964 |
| **Forêt 1M ×50, masque 6,25 %** | **27,2 %** | **1 711** |
| Forêt 1M ×50, masque 12,5 % | 26,8 % | 1 714 |
| Forêt 1M ×50, masque 25 % | 26,5 % | 1 986 |
| 28M fine-tuné seul (pour comparaison) | 36,2 % | 632 |

Sur le test, l'avantage du masque en accuracy fond (+0,2 point au lieu de +1,8 sur la sélection) et reste dans le bruit. En perplexité, il est net : −13 % pour les masques de 6,25 % et 12,5 %. Les forêts masquées sont mieux calibrées, sans trouver plus souvent le bon mot.

Résultats : `lambada/outputs/openai/` (`forest_epochs_1M/`, `forest_grid_1M_v2/`, `single_models/`).

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

## Analyse de sensibilité de l'embedding : méthode

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

## Analyse de sensibilité : résultats

L'importance d'une dimension dépend fortement du nombre de dimensions déjà supprimées. Seule la dimension 18 compte dans tous les régimes. Tout porte sur TinyStories-1M avant fine-tuning, en texte propre, avec 512 passages fixes.

**Correction de méthode.** Le premier Morris basculait chaque dimension une fois le long d'une trajectoire. En binaire, une telle trajectoire part de q dimensions gardées et finit à 1 − q : elle parcourt tout l'espace, et les deux régimes visés se confondaient. Il est refait avec le plan radial (Campolongo et al. 2011), où l'on bascule une dimension à la fois puis on revient au point de départ. Les états restent proches du point de départ.

**Morris radial (50 points de départ par régime)**

| Régime | Dimensions les plus influentes | Accord avec « une à la fois » |
| --- | --- | --- |
| Proche du modèle complet (~10 % supprimées) | 56, 10, 57, 45, 52, 46, 20, 49, 33, 18 | 0,80 |
| Moitié supprimée | 18, 36, 16, 40, 27, 0, 26, 55, 56, 11 | 0,00 |

- **Les deux classements n'ont presque rien en commun** (corrélation de rang de 0,16).
- **Les dimensions 36, 16 et 0 servent de secours.** Elles comptent peu tant que le modèle est presque complet (rangs 40 à 50), et deviennent critiques quand la moitié des dimensions manque (rangs 2, 3 et 6).
- **Aucune dimension n'est négligeable.** Près du modèle complet, la moins influente pèse encore 12 % de la plus influente, et la moitié de l'effet est portée par 18 dimensions.

**Sobol sur les 16 dimensions retenues par le criblage** (les 48 autres restent gardées, N = 1 024, environ 18 000 évaluations par plan) :

| Dimension | 18 | 57 | 10 | 20 | 33 | 46 | 49 | 40 | 55 | 47 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1, effet propre (plan binaire) | 0,26 | 0,12 | 0,09 | 0,08 | 0,07 | 0,05 | 0,04 | 0,03 | 0,03 | 0,02 |
| ST, avec interactions (plan binaire) | 0,29 | 0,17 | 0,12 | 0,10 | 0,10 | 0,09 | 0,06 | 0,05 | 0,03 | 0,06 |

- **Les effets sont largement additifs.** Les effets propres expliquent 83 % de la variance de la perte, et les interactions environ 17 %.
- **La dimension 18 explique à elle seule 26 % de la variance.**
- **Le classement tient d'un plan à l'autre** : corrélation de rang de 0,91 entre suppression franche (plan binaire) et atténuation progressive (plan continu).
- **Estimateurs vérifiés.** Le premier estimateur de S1 n'était pas centré et donnait des valeurs inutilisables (somme de 1,41). Il est corrigé, et complété par la différence des moyennes, spécifique aux entrées binaires. Les deux concordent, et les deux retrouvent les vrais indices sur une fonction test.

Résultats : `lambada/outputs/openai/sensitivity_TinyStories-1M*/`.

**Pour les forêts.** Les masques de 6,25 à 12,5 % se situent dans le régime proche du modèle complet : c'est ce classement qui compte pour eux. Expérience naturelle : tirer les masques parmi les dimensions les moins influentes de ce régime, et protéger la 18.

**Masques guidés (1M, 50 membres, même graine que les masques aléatoires, donc comparaison appariée)**

Deux stratégies : « low » tire les dimensions masquées parmi les 32 moins influentes ; « weighted » tire avec une probabilité inversement proportionnelle à mu* (Morris, régime q = 0,9). Low a été abandonné après 6,25 %.

| Masque | Stratégie | 10 membres | 30 membres | 50 membres | Perplexité à 50 |
| --- | --- | --- | --- | --- | --- |
| 6,25 % | aléatoire | – | – | 27,16 % | 1 711 |
| 6,25 % | weighted | – | – | 27,55 % | 1 664 |
| 6,25 % | low | – | – | 26,81 % | 1 697 |
| 12,5 % | aléatoire | 23,66 % | 26,24 % | 26,81 % | 1 714 |
| 12,5 % | weighted | 24,14 % | 26,47 % | 27,20 % | 1 683 |

- **Weighted fait un peu mieux qu'aléatoire, régulièrement** : environ +0,4 point à toutes les tailles, et une perplexité 2 à 4 % plus basse, sur le test comme sur la sélection.
- **La deuxième graine ne confirme pas le gain en précision.** À 6,25 % et 50 membres, weighted fait 26,97 % contre 27,12 % pour aléatoire (graine 1), après 27,55 % contre 27,16 % (graine 0). Seule la perplexité reste un peu plus basse avec weighted pour les deux graines (−1 à −3 %). Conclusion : **le guidage par la sensibilité pré-fine-tuning n'apporte rien de mesurable**.
- **Low ne marche pas** : se limiter aux 32 dimensions les moins influentes réduit la diversité des masques, ce qui coûte plus que la protection des dimensions importantes.
- **Un membre seul est aussi bon quel que soit le guidage** (~13,3 %) : le fine-tuning compense le masque. Le gain éventuel de weighted vient de la forêt, pas des membres.

**Masque de 6,25 % jusqu'à 100 membres (1M, graine 0)**

| Membres | 10 | 20 | 40 | 60 | 80 | 100 |
| --- | --- | --- | --- | --- | --- | --- |
| Précision (test B) | 24,3 % | 26,1 % | 27,2 % | 27,6 % | 27,9 % | 27,9 % |
| Perplexité | 2 938 | 2 149 | 1 813 | 1 676 | 1 604 | 1 561 |

- **Plateau vers 80 membres, autour de 27,9 %.** La perplexité, elle, continue de baisser doucement.
- **Le plafond des forêts de 1M reste loin du 28M** : 100 membres de 1M (environ 375 M de paramètres) font 27,9 %, contre 36,2 % pour un seul 28M fine-tuné et 42,7 % pour une forêt de 20 × 28M. La forêt ne remplace pas la taille du modèle.

**Deuxième graine (1M, masques et bootstraps retirés)**

| 50 membres | Graine 0 | Graine 1 |
| --- | --- | --- |
| Bagging seul | 26,93 % (ppl 1 964) | 26,66 % (ppl 1 961) |
| Masque 6,25 % aléatoire | 27,16 % (ppl 1 711) | 27,12 % (ppl 1 692) |
| Masque 6,25 % weighted | 27,55 % (ppl 1 664) | 26,97 % (ppl 1 678) |

- **Le masque de 6,25 % bat le bagging pour les deux graines** : +0,2 et +0,5 point, et surtout une perplexité 13 % plus basse les deux fois. Cet effet sur la perplexité est solide ; celui sur la précision est petit mais de même signe.
- **La variation entre graines est d'environ 0,3 point à 50 membres**, du même ordre que les écarts entre stratégies de masque.

**Forêts de 28M en texte propre (20 membres, 6 époques, test B)**

| Membres | Sans masque | Masque 6,25 % | Masque 12,5 % | Perplexité (0 / 6,25 / 12,5 %) |
| --- | --- | --- | --- | --- |
| 1 | 32,7 % | 31,9 % | 30,8 % | 1 093 / 1 159 / 1 311 |
| 2 | 36,8 % | 36,3 % | 35,8 % | 424 / 424 / 449 |
| 5 | 40,4 % | 40,2 % | 40,1 % | 212 / 203 / 212 |
| 10 | 41,9 % | 41,9 % | 41,9 % | 163 / 151 / 155 |
| 15 | 42,4 % | 42,5 % | 42,9 % | 147 / 133 / 137 |
| 20 | 42,7 % | 42,8 % | **43,3 %** | 138 / 124 / 128 |

Référence : le 28M fine-tuné seul (10 époques, données complètes) fait 36,2 % pour une perplexité de 632.

- **La forêt marche aussi sur le gros modèle** : +6,5 points sur le 28M fine-tuné seul, et une perplexité divisée par 4,5. Deux membres suffisent déjà à le rattraper.
- **Le masque croise le bagging plus tôt** qu'avec le 1M, vers 10 membres au lieu de 20. Sa perplexité est 10 % plus basse à 20 membres, et il est meilleur sur la sélection (42,8 % contre 41,2 %). En précision sur le test, l'écart reste dans le bruit.
- **Le masque de 12,5 % devient le meilleur à partir de 15 membres** : 43,3 % à 20 membres, et 44,2 % sur la sélection (contre 41,2 % sans masque). Plus le masque est grand, plus il perd sur un membre seul et plus il gagne sur une grande forêt, comme supposé.
- **La courbe s'aplatit sans masque** (+0,8 point entre 10 et 20 membres), alors que le masque de 12,5 % monte encore (+1,4 point).

**Jusqu'à 30 membres (28M, nouvelle série, même graine)**

| Membres | Sans masque | Masque 12,5 % | Masque 25 % |
| --- | --- | --- | --- |
| 1 | 32,8 % | 31,0 % | 28,6 % |
| 10 | 41,8 % | 42,0 % | 41,0 % |
| 20 | 42,7 % | 43,3 % | 42,4 % |
| 30 | 43,0 % (ppl 127) | **44,0 %** (ppl 119) | 43,0 % (ppl 136) |
| 30, sélection | 41,0 % | 43,6 % | 43,4 % |

- **Le masque de 12,5 % est le bon réglage pour le 28M** : +1 point sur le test à 30 membres, +2,6 points sur la sélection, et une perplexité plus basse. L'écart grandit avec la taille de la forêt.
- **Le masque de 25 % est trop fort** : il perd trop sur chaque membre (28,6 % seul) pour que la diversité compense à 30 membres, sauf sur la sélection.
- **Sans masque, la forêt plafonne** : +0,3 point entre 20 et 30 membres. Le masque de 12,5 % gagne encore +0,7 point.
- **Deuxième graine (forêt et bootstraps retirés) : l'écart tient.** À 30 membres, 44,35 % avec le masque de 12,5 % contre 42,80 % sans (perplexité 116 contre 127), et 42,8 % contre 41,0 % sur la sélection. Sur les deux graines, le masque de 12,5 % apporte +1,0 et +1,6 point : bien au-delà de la variation entre graines (~0,3 point).
- **18,75 % (graine 0, 30 membres)** : 43,7 % (perplexité 122), entre 12,5 % (44,0 %) et 25 % (43,0 %). **L'optimum à 30 membres est autour de 12,5 %** ; il glissera sans doute vers des masques plus grands avec des forêts plus grandes, mais 40 membres de 28M dépassent la mémoire vive disponible (membres gardés sur CPU).
- **Coût** : environ 54 s d'entraînement par membre, soit 18 min pour 20 membres.

Résultats : `lambada/outputs/openai/forest_grid_28M/`.

**Prochaines étapes**

1. ~~Pousser le masque de 6,25 % jusqu'à 100 membres~~ : fait, plateau vers 80 membres.
2. ~~Masques guidés par la sensibilité~~ : fait, voir ci-dessus.
3. ~~Forêts de 28M en texte propre~~ : fait, voir ci-dessus.
4. ~~Deuxième graine~~ : fait, voir ci-dessus.

## Ensembles de masques à l'inférence (un seul modèle)

Question : le gain du masque dans la forêt vient-il seulement de la diversité des votes ? Test : UN modèle, X masques aléatoires appliqués à l'inférence (entrée + notation), probabilités moyennées. Un seul jeu de poids en mémoire, un seul entraînement (ou aucun). Script : `lambada/mask_ensemble.py`. Trois variantes : modèle de base ; fine-tuné une fois sans masque ; fine-tuné une fois avec un nouveau masque aléatoire de 12,5 % à chaque pas (« dropout de dimensions »).

**28M (test B, 30 masques)**

| Variante | Seul, sans masque | Meilleur ensemble de masques | Entraînement |
| --- | --- | --- | --- |
| Base (pas de fine-tuning) | 15,1 % (ppl 10 039) | 16,3 % (12,5 %, 20 masques ; ppl 1 960) | 0 |
| Fine-tuné une fois | 35,7 % (ppl 696) | 38,7 % (18,75–25 % ; ppl 267) | 52 s |
| Fine-tuné avec dropout de dimensions | 37,3 % (ppl 721) | **40,75 %** (18,75 % ; ppl 213) | 54 s |
| *Forêt 30 × 28M, bagging sans masque* | | *43,0 % (ppl 127)* | *27 min* |
| *Forêt 30 × 28M, masque 12,5 %* | | *44,0 % (ppl 119)* | *27 min* |

**1M (15 époques, 30 masques)** : fine-tuné une fois 18,1 % → 21,5 % avec 30 masques de 6,25 % ; avec dropout de dimensions 17,7 % → 20,3 %. Le modèle de base 1M s'effondre dès qu'on le masque (3,8 % → moins de 1 %). Forêt 1M de 30 membres : environ 26,8 %.

- **Les masques à l'inférence apportent un vrai gain sans aucun entraînement supplémentaire** : +3 points sur le 28M fine-tuné, +3,4 sur le 1M, perplexité divisée par 2,5.
- **Entraîner avec un masque différent à chaque pas aide deux fois** : le modèle seul gagne déjà +1,6 point (régularisation), et il supporte des masques plus forts à l'inférence. Meilleur réglage : 40,75 %, pour le coût d'un seul fine-tuning.
- **Mais cela ne remplace pas la forêt.** La forêt sans masque (43,0 %) bat le meilleur ensemble de masques (40,75 %), avec une perplexité bien plus basse (127 contre 213). La diversité qui vient d'entraînements différents (bootstrap, graines) vaut plus que celle des masques d'inférence. L'hypothèse « le gain n'est qu'un vote aléatoire » est donc vraie pour une partie seulement : environ 5 points sur 8 de la forêt s'obtiennent avec un seul modèle.
- **Sans fine-tuning, presque rien** : +1,2 point sur le 28M, et le 1M s'effondre. Le modèle doit avoir vu des masques ou au moins être adapté à la tâche.

**Quel filtre ?** (`lambada/filter_analysis.py`, 28M fine-tuné une fois, masque de 12,5 %, test B)

| Traitement du modèle seul | Précision | Perplexité |
| --- | --- | --- |
| Rien | 35,7 % | 696 |
| Meilleure température (T = 1,4, choisie sur la sélection) | 35,7 % | 445 |
| Sortie multipliée par p = 0,875 (= température 1,14) | 35,7 % | 507 |
| 30 masques en sortie seule | 35,7 % | 503 |
| Entrée multipliée par p | 34,8 % | 745 |
| 30 masques en entrée seule | 38,0 % | 286 |
| 30 masques entrée + sortie | 38,6 % | 258 |

- **Côté sortie, le masque n'est qu'une température.** 30 masques en sortie donnent la même chose que la multiplication par p, c'est-à-dire une température 1/p : même précision, même perplexité (503 contre 507). Le terme de variance est négligeable. Une température bien réglée fait même mieux (445).
- **Côté entrée, ce n'est pas un filtre, c'est un vrai ensemble.** Remplacer les masques par leur espérance dégrade le modèle (34,8 %), alors que 30 masques tirés au hasard donnent +2,3 points et une perplexité de 286, bien meilleure qu'avec n'importe quelle température. Le masque traverse le réseau, non linéaire : chaque masque donne un sous-réseau différent et c'est leur désaccord qui sert, comme le MC dropout.
- **Même schéma sur le 1M et sur la variante dropout** : précision gagnée uniquement par les masques d'entrée tirés au hasard ; environ 45 % du gain de log-perplexité s'obtiennent par une simple température.
- **Le 1M de base ne supporte pas les masques d'entrée** (3,8 % → 0,8 %), alors que le 28M de base les tolère (15,1 % → 15,6 %).

Conséquence pratique : le masque de sortie est inutile (une température fait mieux, en une passe). Seul le masque d'entrée apporte de la diversité.

Piste suivante : combiner, c'est-à-dire quelques membres entraînés avec dropout de dimensions, chacun vu à travers plusieurs masques (par exemple 5 membres × 6 masques).

## Ce que dit la littérature

Je n'ai pas trouvé d'analyse de sensibilité formelle (Morris ou Sobol) appliquée aux dimensions de l'embedding d'un modèle de langage. La recherche est rapide et non exhaustive. Les travaux voisins pointent trois choses : quelques dimensions dominent souvent, leur variance ne dit pas leur importance, et les axes ne sont pas arbitraires quand le modèle est entraîné avec Adam.

| Travail | Méthode | Ce qu'ils ont trouvé | Conséquence pour nous |
| --- | --- | --- | --- |
| [Kovaleva et al. 2021, BERT Busters](https://arxiv.org/pdf/2105.06990) | Mise à zéro de 1 ou 2 dimensions repérées à leur amplitude | Jusqu'à −44 points sur BERT et ses variantes, moins sur GPT-2 | S'attendre à quelques dimensions dominantes |
| [Puccetti et al. 2022](https://arxiv.org/abs/2205.11380v3) | Lien entre ces dimensions et les données d'entraînement | Leur amplitude suit la fréquence des tokens | Cohérent avec nos corrélations, créées par les tokens fréquents |
| [Sun et al. 2024, Massive Activations](https://arxiv.org/pdf/2402.17762) | Repérage puis mise à zéro d'activations géantes | 2 dimensions fixes chez LLaMA2, qui servent de biais ; les supprimer casse le modèle | Même attente |
| [Timkey & van Schijndel 2021, Rogue Dimensions](https://aclanthology.org/2021.emnlp-main.372) | Comparaison entre géométrie et comportement | 1 à 3 dimensions dominent la similarité, sans être celles qui comptent pour le comportement | Mesurer l'effet sur la perte, pas la variance |
| [Ethayarajh 2019](https://ar5iv.arxiv.org/html/1909.00512) | Géométrie des représentations | Elles occupent un cône étroit (anisotropie) | Cohérent avec notre rang effectif d'environ 19 sur 64 |
| [Mu & Viswanath 2018, All-but-the-Top](https://arxiv.org/pdf/1702.01417) | Retrait des directions principales dominantes | Les embeddings de mots s'améliorent | Piste pour masquer dans la base des composantes principales |
| [Elhage et al. 2023, Privileged Bases](https://transformer-circuits.pub/2023/privileged-basis/index.html) et [Caples & Neuhaus](https://www.lesswrong.com/posts/yrhu6MeFddnGRSLtQ/adam-optimizer-causes-privileged-basis-in-transformer-lm) | Comparaison Adam contre SGD | Adam rend certaines coordonnées privilégiées, reproduit sur un TinyStories de 12M (non relu par des pairs) | La base brute a du sens pour TinyStories |
| [Fel et al. 2021](https://arxiv.org/pdf/2111.04138) | Sobol avec des masques de perturbation, sur des images | Attributions efficaces, interactions comprises | Précédent pour utiliser le masque comme entrée |
| [Ghorbani & Zou 2020, Neuron Shapley](https://proceedings.neurips.cc/paper/2020/hash/41c542dfe6e4fc3deb251d64cf6ed2e4-Abstract.html) | Valeurs de Shapley par ablation, accélérées par un bandit | Retirer 30 filtres détruit l'accuracy d'Inception-v3 | Recours si Morris montre beaucoup d'interactions, et pour le 28M |
| [Iooss & Prieur](https://arxiv.org/pdf/1707.01334v6), dans la lignée d'Owen 2014 et de Song et al. 2016 | Effets de Shapley | La référence quand les entrées sont corrélées | Si une réponse rigoureuse en base brute est nécessaire |

La pratique habituelle avec Morris est de cribler d'abord, puis de faire un Sobol sur les entrées retenues ([UQpy](https://uqpyproject.readthedocs.io/en/latest/sensitivity/morris.html)). C'est le plan suivi ici.

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
