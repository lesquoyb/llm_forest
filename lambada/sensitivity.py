"""
LAMBADA : analyse de sensibilité des dimensions de l'embedding (Morris, Sobol).

Morris (--method morris, par défaut), version binaire centrée sur la
suppression. Chaque point de départ garde chaque dimension avec la
probabilité q (--keep-probs). L'effet élémentaire EE_i = hausse de la perte
quand on supprime la dimension i, dans ce contexte. Deux plans (--morris-design) :
- "radial" (par défaut, Campolongo et al. 2011) : depuis chaque point de
  départ, on bascule une seule dimension à la fois et on revient au point de
  départ. Les états restent proches du point de départ : q fixe vraiment le
  régime (q = 0,9 : environ 10 % de dimensions supprimées) ;
- "trajectory" (Morris 1991) : on bascule les dimensions l'une après l'autre
  sans revenir. En binaire, la trajectoire finit avec la proportion inverse
  de dimensions gardées (de q à 1 - q) : criblage global de tout l'espace,
  quel que soit q. Par dimension :
mu* = moyenne de |EE_i| (importance), mu = moyenne de EE_i (signe),
sigma = écart-type de EE_i (interactions : l'effet dépend des autres
dimensions supprimées). Comparer plusieurs q montre si le classement tient
selon le nombre de dimensions déjà supprimées. Coût : r x (d + 1) masques.

Sobol (--method sobol) :

Entrées : un facteur x_i par dimension de l'embedding, qui multiplie la
dimension i à l'entrée et à la notation, exactement comme le masque des
forêts (x_i = 0 : dimension neutralisée). Deux plans d'expérience, sans
réglage arbitraire :
- "binary"  : x_i dans {0, 1}, tiré avec P(1) = 1/2 ; tous les masques sont
              équiprobables ;
- "uniform" : x_i ~ U[0, 1], atténuation continue.

Sortie : la perte moyenne par token sur le mot cible (et l'accuracy) sur un
jeu fixe de passages. Indices :
- premier ordre S_i : part de la variance de la sortie due à x_i seul
  (estimateur de Saltelli 2010) ;
- total ST_i : part due à x_i, interactions comprises (estimateur de Jansen).
Intervalles de confiance à 95 % par bootstrap. 1 - somme(S_i) = part des
interactions. En complément : l'effet de chaque dimension masquée seule
("une à la fois"), autour du modèle complet.

Coût : N x (d + 2) évaluations de masque. Les masques sont évalués par
paquets dans un même passage GPU (les passages sont recopiés une fois par
masque du paquet).

Exemples :
    python lambada/sensitivity.py --format openai
    python lambada/sensitivity.py --design uniform --n 2048
    python lambada/sensitivity.py --checkpoint lambada/outputs/openai/finetuned_1M
    python lambada/sensitivity.py --quick
"""

import argparse
import json
import math
import time
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import collate, load_lambada, prepare_corpus


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="roneneldan/TinyStories-1M")
    parser.add_argument(
        "--checkpoint", default=None,
        help="dossier d'un modèle sauvegardé (ex. fine-tuné) à la place de --model",
    )
    parser.add_argument("--format", choices=["cimec", "openai"], default="openai")
    parser.add_argument("--method", choices=["morris", "sobol"], default="morris")
    parser.add_argument("--keep-probs", type=float, nargs="+", default=[0.9, 0.5],
                        help="Morris : probabilité de garder chaque dimension au départ des trajectoires")
    parser.add_argument("--trajectories", type=int, default=50,
                        help="Morris : nombre de points de départ (trajectoires) r")
    parser.add_argument("--morris-design", choices=["radial", "trajectory"], default="radial")
    parser.add_argument("--designs", nargs="+", choices=["binary", "uniform"], default=["binary", "uniform"])
    parser.add_argument(
        "--sobol-dims", default="all",
        help="Sobol : 'all', 'topK' (les K plus influentes selon Morris, lu dans results.json) "
             "ou une liste '3,17,42'. Les autres dimensions restent gardées (facteur 1).",
    )
    parser.add_argument("--keep-prob", type=float, default=0.5,
                        help="Sobol, plan binaire : probabilité de garder chaque dimension")
    parser.add_argument("--morris-q", default=None,
                        help="régime Morris (q) utilisé pour 'topK' ; par défaut le premier")
    parser.add_argument("--n", type=int, default=1024, help="taille de base N de Saltelli")
    parser.add_argument("--eval-samples", type=int, default=512)
    # Masques x passages = séquences traitées d'un coup : au-delà de quelques
    # centaines, les matrices d'attention saturent la mémoire du GPU.
    parser.add_argument("--mask-batch", type=int, default=8, help="masques évalués ensemble")
    parser.add_argument("--passage-batch", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--quick", action="store_true", help="petit test (N=32, 64 passages)")
    args = parser.parse_args()

    if args.quick:
        args.n = 32
        args.trajectories = 4
        args.eval_samples = 64
        args.bootstrap = 50

    if args.output_dir is None:
        name = Path(args.checkpoint).name if args.checkpoint else args.model.split("/")[-1]
        args.output_dir = str(Path(__file__).parent / "outputs" / args.format / f"sensitivity_{name}")

    return args


# ============================================================
# Évaluation d'un paquet de masques
# ============================================================

class MaskEvaluator:
    """Perte par token et accuracy par mot pour un paquet de vecteurs de masque [K, H]."""

    def __init__(self, model, tokenizer, data, passage_batch, device):
        self.model = model
        self.embed = model.get_input_embeddings()
        self.head = model.get_output_embeddings()
        self.device = device
        self.chunks = []
        # Trié par longueur : chaque paquet est paddé au plus long de ses passages.
        data = sorted(data, key=lambda e: len(e.context_ids) + len(e.target_ids))

        for start in range(0, len(data), passage_batch):
            batch = collate(data[start:start + passage_batch], tokenizer.pad_token_id)
            # Paquet complété jusqu'au multiple de 32 supérieur : peu de formes de
            # tenseurs différentes. Sinon PyTorch garde un bloc de mémoire en
            # réserve par forme rencontrée, et la réserve dépasse la mémoire du GPU.
            length = -(-batch["input_ids"].shape[1] // 32) * 32
            pad = length - batch["input_ids"].shape[1]
            input_ids = torch.nn.functional.pad(batch["input_ids"], (0, pad), value=tokenizer.pad_token_id)
            attention_mask = torch.nn.functional.pad(batch["attention_mask"], (0, pad), value=0)
            labels = torch.nn.functional.pad(batch["labels"], (0, pad), value=-100).to(device)
            is_target = labels != -100
            self.chunks.append({
                "input_ids": input_ids.to(device),
                "attention_mask": attention_mask.to(device),
                "is_target": is_target,
                "targets": labels[is_target],
                "word_index": is_target.nonzero()[:, 0],
                "n_words": labels.shape[0],
            })

        self.n_tokens = sum(c["targets"].numel() for c in self.chunks)
        self.n_words = sum(c["n_words"] for c in self.chunks)

    @torch.no_grad()
    def __call__(self, masks):
        masks = masks.to(self.device)
        k, hidden = masks.shape
        nll = torch.zeros(k, dtype=torch.float64, device=self.device)
        correct = torch.zeros(k, device=self.device)

        for c in self.chunks:
            p, length = c["input_ids"].shape
            embeds = self.embed(c["input_ids"])  # [P, L, H]
            x = (embeds[None] * masks[:, None, None, :]).reshape(k * p, length, hidden)
            out = self.model.base_model(
                inputs_embeds=x, attention_mask=c["attention_mask"].repeat(k, 1)
            ).last_hidden_state.view(k, p, length, hidden) * masks[:, None, None, :]

            logits = self.head(out[:, c["is_target"]])  # [K, n_tokens, V]
            log_probs = logits.log_softmax(-1)
            targets = c["targets"].expand(k, -1)
            nll -= log_probs.gather(-1, targets[..., None]).squeeze(-1).sum(1).double()

            wrong = torch.zeros(k, c["n_words"], device=self.device)
            wrong.index_add_(1, c["word_index"], (logits.argmax(-1) != targets).float())
            correct += (wrong == 0).sum(1)

        return (nll / self.n_tokens).float().cpu(), (correct / self.n_words).cpu()

    def many(self, masks, batch, progress=None):
        losses, accs = [], []
        for n, start in enumerate(range(0, len(masks), batch)):
            loss, acc = self(masks[start:start + batch])
            losses.append(loss)
            accs.append(acc)
            if n % 50 == 49 and self.device.type == "cuda":
                torch.cuda.empty_cache()
            if progress:
                progress(start + batch)
        return torch.cat(losses), torch.cat(accs)


# ============================================================
# Sobol
# ============================================================

def sample(design, n, d, generator, keep_prob=0.5):
    if design == "binary":
        return (torch.rand(n, d, generator=generator) < keep_prob).float()
    return torch.rand(n, d, generator=generator)


def sobol_indices(f_a, f_b, f_ab):
    """f_a, f_b : [N] ; f_ab : [d, N]. Retourne (S1, ST) de taille d."""

    both = torch.cat([f_a, f_b])
    var = both.var()
    # Saltelli 2010, sorties centrées : sans centrage, la moyenne de la sortie
    # (une perte de ~11) gonfle énormément la variance de l'estimateur.
    f_b_c = f_b - both.mean()
    s1 = (f_b_c[None] * (f_ab - f_a[None])).mean(1) / var
    st = 0.5 * ((f_a[None] - f_ab) ** 2).mean(1) / var  # Jansen
    return s1, st


def binary_first_order(masks, y, active):
    """
    S1 pour des entrées binaires indépendantes, par différence des moyennes :
    S1_i = p(1-p) (E[Y | x_i = 1] - E[Y | x_i = 0])^2 / Var(Y).
    Toutes les lignes du plan de Saltelli (A, B et les AB_i) sont des tirages
    de la loi d'entrée, donc toutes servent : estimateur bien plus précis.
    """

    x = masks[:, active]
    p = x.mean(0)
    mean_kept = (x * y[:, None]).sum(0) / x.sum(0)
    mean_removed = ((1 - x) * y[:, None]).sum(0) / (1 - x).sum(0)
    return p * (1 - p) * (mean_kept - mean_removed) ** 2 / y.var()


def bootstrap(f_a, f_b, f_ab, n_boot, generator):
    n = len(f_a)
    s1s, sts = [], []
    for _ in range(n_boot):
        idx = torch.randint(n, (n,), generator=generator)
        s1, st = sobol_indices(f_a[idx], f_b[idx], f_ab[:, idx])
        s1s.append(s1)
        sts.append(st)
    s1s, sts = torch.stack(s1s), torch.stack(sts)
    q = torch.tensor([0.025, 0.975])
    return torch.quantile(s1s, q, dim=0), torch.quantile(sts, q, dim=0)


def run_design(design, evaluator, d_full, args, active=None):
    """Sobol sur les dimensions `active` (toutes par défaut) ; les autres restent à 1."""

    active = list(range(d_full)) if active is None else list(active)
    d = len(active)
    generator = torch.Generator().manual_seed(args.seed)
    a = sample(design, args.n, d, generator, args.keep_prob)
    b = sample(design, args.n, d, generator, args.keep_prob)
    ab = a[None].repeat(d, 1, 1)  # [d, N, d]
    for i in range(d):
        ab[i, :, i] = b[:, i]

    small = torch.cat([a, b, ab.reshape(d * args.n, d)])
    masks = torch.ones(len(small), d_full)
    masks[:, active] = small
    total = len(masks)
    start = time.perf_counter()

    def progress(done):
        if done == args.mask_batch * 10 or done % (args.mask_batch * 200) == 0:
            elapsed = time.perf_counter() - start
            eta = elapsed / done * (total - done)
            print(f"  {design} : {min(done, total)}/{total} masques | reste ~{eta / 60:.0f} min", flush=True)

    loss, acc = evaluator.many(masks, args.mask_batch, progress)
    elapsed = time.perf_counter() - start

    results = {"design": design, "n": args.n, "evaluations": total, "time_s": elapsed,
               "active_dims": active, "keep_prob": args.keep_prob if design == "binary" else None}
    boot_gen = torch.Generator().manual_seed(args.seed + 1)

    # Sorties brutes gardées : les indices peuvent être recalculés sans relancer.
    torch.save({"masks": masks, "loss": loss, "accuracy": acc, "active": active},
               Path(args.output_dir) / f"sobol_{design}_raw.pt")

    for name, y in (("loss", loss), ("accuracy", acc)):
        f_a, f_b = y[:args.n], y[args.n:2 * args.n]
        f_ab = y[2 * args.n:].view(d, args.n)
        s1, st = sobol_indices(f_a, f_b, f_ab)
        (s1_lo, s1_hi), (st_lo, st_hi) = bootstrap(f_a, f_b, f_ab, args.bootstrap, boot_gen)
        if design == "binary":
            s1_dm = binary_first_order(masks, y, active)
            boots = torch.stack([
                binary_first_order(masks[idx], y[idx], active)
                for idx in (torch.randint(len(y), (len(y),), generator=boot_gen)
                            for _ in range(args.bootstrap))
            ])
            dm_lo, dm_hi = torch.quantile(boots, torch.tensor([0.025, 0.975]), dim=0)
        results[name] = {
            "mean": torch.cat([f_a, f_b]).mean().item(),
            "std": torch.cat([f_a, f_b]).std().item(),
            "S1": s1.tolist(), "S1_low": s1_lo.tolist(), "S1_high": s1_hi.tolist(),
            "ST": st.tolist(), "ST_low": st_lo.tolist(), "ST_high": st_hi.tolist(),
            "sum_S1": s1.sum().item(), "sum_ST": st.sum().item(),
        }
        if design == "binary":
            results[name].update({
                "S1_diff_means": s1_dm.tolist(),
                "S1_diff_means_low": dm_lo.tolist(),
                "S1_diff_means_high": dm_hi.tolist(),
                "sum_S1_diff_means": s1_dm.sum().item(),
            })

    return results


# ============================================================
# Morris
# ============================================================

def morris_trajectories(d, r, keep_prob, generator):
    """r trajectoires binaires de d + 1 points ; retourne (points [r, d+1, d], ordres [r, d])."""

    points = torch.empty(r, d + 1, d)
    orders = torch.empty(r, d, dtype=torch.long)

    for t in range(r):
        x = (torch.rand(d, generator=generator) < keep_prob).float()
        order = torch.randperm(d, generator=generator)
        points[t, 0] = x
        for step, i in enumerate(order.tolist()):
            x = x.clone()
            x[i] = 1 - x[i]
            points[t, step + 1] = x
        orders[t] = order

    return points, orders


def morris_radial(d, r, keep_prob, generator):
    """r points de départ, chacun suivi des d états où une seule dimension est basculée."""

    points = torch.empty(r, d + 1, d)
    orders = torch.arange(d).repeat(r, 1)
    for t in range(r):
        x = (torch.rand(d, generator=generator) < keep_prob).float()
        points[t, 0] = x
        flipped = x.repeat(d, 1)
        flipped[torch.arange(d), torch.arange(d)] = 1 - x
        points[t, 1:] = flipped
    return points, orders


def morris_radial_effects(f, points):
    """Effets de la suppression [r, d] pour le plan radial : comparaison au point de départ."""

    base = points[:, 0]  # [r, d]
    delta = f[:, 1:] - f[:, :1]  # [r, d] : état basculé - point de départ
    # Dimension gardée au départ -> le basculement la supprime : l'effet est delta.
    return torch.where(base == 1, delta, -delta)


def morris_effects(f, points, orders):
    """Effets élémentaires de la suppression [r, d] à partir des sorties f [r, d+1]."""

    r, d = orders.shape
    effects = torch.empty(r, d)
    for t in range(r):
        for step, i in enumerate(orders[t].tolist()):
            delta = f[t, step + 1] - f[t, step]
            # Dimension gardée avant le pas -> supprimée : l'effet est delta.
            effects[t, i] = delta if points[t, step, i] == 1 else -delta
    return effects


def run_morris(keep_prob, evaluator, d, args):
    generator = torch.Generator().manual_seed(args.seed)
    if args.morris_design == "radial":
        points, orders = morris_radial(d, args.trajectories, keep_prob, generator)
    else:
        points, orders = morris_trajectories(d, args.trajectories, keep_prob, generator)
    masks = points.reshape(-1, d)
    total = len(masks)
    start = time.perf_counter()

    def progress(done):
        if done == args.mask_batch * 10 or done % (args.mask_batch * 100) == 0:
            elapsed = time.perf_counter() - start
            print(f"  q = {keep_prob:g} : {min(done, total)}/{total} masques | "
                  f"reste ~{elapsed / done * (total - done) / 60:.0f} min", flush=True)

    loss, acc = evaluator.many(masks, args.mask_batch, progress)
    results = {"keep_prob": keep_prob, "design": args.morris_design,
               "trajectories": args.trajectories, "evaluations": total,
               "time_s": time.perf_counter() - start}
    boot_gen = torch.Generator().manual_seed(args.seed + 1)

    for name, y in (("loss", loss), ("accuracy", acc)):
        f = y.view(args.trajectories, d + 1)
        effects = (morris_radial_effects(f, points) if args.morris_design == "radial"
                   else morris_effects(f, points, orders))
        mu_star = effects.abs().mean(0)
        boots = torch.stack([
            effects[torch.randint(args.trajectories, (args.trajectories,), generator=boot_gen)].abs().mean(0)
            for _ in range(args.bootstrap)
        ])
        low, high = torch.quantile(boots, torch.tensor([0.025, 0.975]), dim=0)
        results[name] = {
            "mean_output": y.mean().item(),
            "mu": effects.mean(0).tolist(),
            "mu_star": mu_star.tolist(),
            "mu_star_low": low.tolist(),
            "mu_star_high": high.tolist(),
            "sigma": effects.std(0).tolist(),
        }

    return results


def resolve_dims(spec, output_dir, d, morris_q=None):
    if spec == "all":
        return list(range(d))
    if spec.startswith("top"):
        k = int(spec[3:])
        with open(Path(output_dir) / "results.json", encoding="utf-8") as f:
            morris = json.load(f)["morris"]
        key = morris_q if morris_q is not None else next(iter(morris))
        mu_star = morris[key]["loss"]["mu_star"]
        return sorted(sorted(range(d), key=lambda i: -mu_star[i])[:k])
    return sorted(int(x) for x in spec.split(","))


def spearman(a, b):
    ra = torch.tensor(a).argsort().argsort().float()
    rb = torch.tensor(b).argsort().argsort().float()
    return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source = args.checkpoint or args.model
    tokenizer = AutoTokenizer.from_pretrained(source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(source).to(device).eval()

    # Jeu fixe : début de l'entraînement (moitié a en format openai), jamais le test.
    train_raw, _, _ = load_lambada(max_train=args.eval_samples, seed=args.seed, fmt=args.format)
    data = prepare_corpus(train_raw, tokenizer, args.max_length)
    d = model.config.hidden_size

    evaluator = MaskEvaluator(model, tokenizer, data, args.passage_batch, device)
    print(
        f"{source} | {d} dimensions | {len(data)} passages, {evaluator.n_tokens} tokens cibles | "
        f"N = {args.n} -> {args.n * (d + 2)} masques par plan",
        flush=True,
    )

    full_loss, full_acc = evaluator(torch.ones(1, d))
    one_off = torch.ones(d, d) - torch.eye(d)  # une dimension masquée à la fois
    oat_loss, oat_acc = evaluator.many(one_off, args.mask_batch)
    print(f"modèle complet : perte {full_loss.item():.3f} | accuracy {full_acc.item() * 100:.2f}%", flush=True)

    results = {
        "config": vars(args),
        "dimensions": d,
        "n_passages": len(data),
        "full": {"loss": full_loss.item(), "accuracy": full_acc.item()},
        "one_at_a_time": {
            "loss_increase": (oat_loss - full_loss).tolist(),
            "accuracy_change": (oat_acc - full_acc).tolist(),
        },
        "designs": {},
        "morris": {},
    }

    if args.method == "morris":
        oat = results["one_at_a_time"]["loss_increase"]
        for keep_prob in args.keep_probs:
            r = run_morris(keep_prob, evaluator, d, args)
            results["morris"][str(keep_prob)] = r
            m = r["loss"]
            ranked = sorted(range(d), key=lambda i: -m["mu_star"][i])
            print(
                f"q = {keep_prob:g} : perte moyenne {m['mean_output']:.3f} | "
                f"plus influentes (mu*) : {ranked[:8]} | moins influentes : {ranked[-8:]} | "
                f"rang corrélé à une-à-la-fois : {spearman(m['mu_star'], oat):.2f} | "
                f"sigma/mu* médian : {torch.tensor(m['sigma']).div(torch.tensor(m['mu_star'])).median():.2f} | "
                f"{r['time_s'] / 60:.1f} min",
                flush=True,
            )
            with open(output_dir / "results.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
        keys = list(results["morris"])
        if len(keys) > 1:
            a, b = (results["morris"][k]["loss"]["mu_star"] for k in keys[:2])
            print(f"corrélation de rang des mu* entre q = {keys[0]} et q = {keys[1]} : {spearman(a, b):.2f}")
        plot_morris(results, output_dir / "morris.png")
        print(f"\nRésultats et graphique enregistrés dans {output_dir}/")
        return

    active = resolve_dims(args.sobol_dims, output_dir, d, args.morris_q)
    if args.method == "sobol":
        # Garde les résultats Morris déjà présents dans le même dossier.
        previous = output_dir / "results.json"
        if previous.exists():
            with open(previous, encoding="utf-8") as f:
                results["morris"] = json.load(f).get("morris", {})
    print(f"Sobol sur {len(active)} dimensions : {active}", flush=True)

    for design in args.designs:
        r = run_design(design, evaluator, d, args, active)
        results["designs"][design] = r
        loss = r["loss"]
        top = [active[j] for j in sorted(range(len(active)), key=lambda j: -loss["ST"][j])[:8]]
        print(
            f"{design} : perte moyenne {loss['mean']:.3f} (écart-type {loss['std']:.3f}) | "
            f"somme S1 {loss['sum_S1']:.2f}, somme ST {loss['sum_ST']:.2f} | "
            f"dimensions les plus influentes (ST) : {top} | {r['time_s'] / 60:.1f} min",
            flush=True,
        )
        with open(output_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    plot(results, output_dir / "sobol.png")
    print(f"\nRésultats et graphique enregistrés dans {output_dir}/")


# ============================================================
# Graphique
# ============================================================

S1_COLOR = "#2a78d6"
ST_COLOR = "#eb6834"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def plot_morris(results, path):
    regimes = list(results["morris"])
    d = results["dimensions"]
    first = results["morris"][regimes[0]]["loss"]
    order = sorted(range(d), key=lambda i: -first["mu_star"][i])
    oat = results["one_at_a_time"]["loss_increase"]

    fig, axes = plt.subplots(len(regimes) + 1, 1, figsize=(15, 3.4 * (len(regimes) + 1)), facecolor=SURFACE)
    for ax, key in zip(axes, regimes):
        m = results["morris"][key]["loss"]
        mu_star = [m["mu_star"][i] for i in order]
        err = [[m["mu_star"][i] - m["mu_star_low"][i] for i in order],
               [m["mu_star_high"][i] - m["mu_star"][i] for i in order]]
        ax.bar([i - 0.2 for i in range(d)], mu_star, width=0.4, color=S1_COLOR, yerr=err,
               error_kw={"elinewidth": 0.6, "ecolor": TEXT_SECONDARY}, label="mu* (importance)")
        ax.bar([i + 0.2 for i in range(d)], [m["sigma"][i] for i in order], width=0.4,
               color=ST_COLOR, label="sigma (interactions)")
        ax.set_title(f"Morris, q = {key} : chaque dimension gardée avec la probabilité {key} au départ",
                     color=TEXT, fontsize=11, loc="left")
        ax.legend(frameon=False, fontsize=9)
    axes[-1].bar(range(d), [oat[i] for i in order], color="#9a9890")
    axes[-1].set_title("Une dimension supprimée à la fois, autour du modèle complet", color=TEXT, fontsize=11, loc="left")

    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.set_xticks(range(d))
        ax.set_xticklabels([str(i) for i in order], fontsize=6, rotation=90)
        ax.set_xlim(-1, d)
        ax.set_ylabel("hausse de la perte", color=TEXT_SECONDARY, fontsize=9)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
        ax.grid(True, axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[-1].set_xlabel(f"Dimension de l'embedding (triées par mu* à q = {regimes[0]})", color=TEXT_SECONDARY)

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot(results, path):
    designs = list(results["designs"])
    # Positions j = 0..k-1 sur les dimensions étudiées (toutes, ou un sous-ensemble).
    active = results["designs"][designs[0]].get("active_dims") or list(range(results["dimensions"]))
    d = len(active)
    fig, axes = plt.subplots(
        len(designs) + 1, 1, figsize=(15, 3.6 * (len(designs) + 1)), facecolor=SURFACE
    )
    axes = list(axes) if len(designs) else [axes]

    order = None
    for ax, design in zip(axes, designs):
        r = results["designs"][design]["loss"]
        if order is None:
            order = sorted(range(d), key=lambda i: -r["ST"][i])
        x = range(d)
        st = [r["ST"][i] for i in order]
        s1 = [r["S1"][i] for i in order]
        st_err = [[r["ST"][i] - r["ST_low"][i] for i in order], [r["ST_high"][i] - r["ST"][i] for i in order]]
        s1_err = [[r["S1"][i] - r["S1_low"][i] for i in order], [r["S1_high"][i] - r["S1"][i] for i in order]]
        ax.bar([i - 0.2 for i in x], st, width=0.4, color=ST_COLOR, label="total (ST)", yerr=st_err,
               error_kw={"elinewidth": 0.6, "ecolor": TEXT_SECONDARY})
        ax.bar([i + 0.2 for i in x], s1, width=0.4, color=S1_COLOR, label="premier ordre (S1)", yerr=s1_err,
               error_kw={"elinewidth": 0.6, "ecolor": TEXT_SECONDARY})
        ax.set_title(
            f"Plan {design} : somme S1 = {r['sum_S1']:.2f}, somme ST = {r['sum_ST']:.2f} "
            f"(perte moyenne {r['mean']:.2f}, écart-type {r['std']:.2f})",
            color=TEXT, fontsize=11, loc="left",
        )
        ax.legend(frameon=False, fontsize=9)

    oat = results["one_at_a_time"]["loss_increase"]
    axes[-1].bar(range(d), [oat[active[i]] for i in order], color="#9a9890")
    axes[-1].set_title(
        "Une dimension masquée à la fois : hausse de la perte autour du modèle complet",
        color=TEXT, fontsize=11, loc="left",
    )

    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.set_xticks(range(d))
        ax.set_xticklabels([str(active[i]) for i in order], fontsize=6 if d > 20 else 9, rotation=90)
        ax.set_xlim(-1, d)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
        ax.grid(True, axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[-1].set_xlabel("Dimension de l'embedding (triées par indice total du premier plan)", color=TEXT_SECONDARY)

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
