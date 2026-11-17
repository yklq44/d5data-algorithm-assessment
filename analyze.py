"""Evaluate the auto-labels against CIFAR-10 ground truth and the CIFAR-10N human labels.

Inputs, all written by demo_code.py:
  outputs/labels_train.csv (+ .json)   final labels for the 50k train images
  outputs/probs/test_*.npy             dev-split scores of every prompt variant that was run
  outputs/run_info.json                timing of every model run
Outputs:
  outputs/metrics.json                 every number quoted in the report
  outputs/figures/*.png
  outputs/examples.csv                 the samples shown in figures/examples.png
  outputs/suspected_label_errors.csv   images where all models and the crowd majority agree (vote confidence
                                       >= 0.9) on a label that differs from CIFAR-10
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score

from demo_code import CLASSES, HERE, MODELS, fit_temperature, load_split, softmax

OUT = HERE / "outputs"
FIG = OUT / "figures"
SIDE_INFO_URL = "https://raw.githubusercontent.com/UCSC-REAL/cifar-10-100n/main/data/side_info_cifar10N.csv"
WORKERS = ["human_label1", "human_label2", "human_label3"]
DISPLAY = {"single_crowd_label": "Single crowd label (mean of 3)", "human_majority": "Crowd majority of 3",
           "clip": "CLIP ViT-L/14", "siglip": "SigLIP2 so400m", "qwen": "Qwen3-VL-4B", "vote": "Model vote (3 models)"}
TO_INT = {c: i for i, c in enumerate(CLASSES)}
SHORT = {"airplane": "plane", "automobile": "car"}
THRESHOLDS = [0.0, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.98, 0.99]


def r4(x):
    return round(float(x), 4)


def accuracy(pred, truth):
    return r4((pred == truth).mean()) if len(truth) else None


def ece(conf, correct, bins=15):
    """Expected calibration error with equal-width confidence bins."""
    b = np.minimum((conf * bins).astype(int), bins - 1)
    return r4(sum((b == i).mean() * abs(correct[b == i].mean() - conf[b == i].mean())
                  for i in range(bins) if (b == i).any()))


def prob_ece(probs, labels):
    return ece(probs.max(1), probs.argmax(1) == labels)


def fleiss_kappa(ratings):
    n, r = ratings.shape
    counts = np.stack([(ratings == c).sum(1) for c in range(len(CLASSES))], axis=1)
    p_chance = ((counts.sum(0) / (n * r)) ** 2).sum()
    p_observed = (((counts ** 2).sum(1) - r) / (r * (r - 1))).mean()
    return r4((p_observed - p_chance) / (1 - p_chance))


def top_confusions(pred, truth, k=5):
    pairs = pd.Series([f"{CLASSES[t]}->{CLASSES[p]}" for t, p in zip(truth, pred) if t != p])
    return {name: int(n) for name, n in pairs.value_counts().head(k).items()}


def accept_mask(conf, n_agree, n_models, threshold):
    return (n_agree == n_models) & (conf >= threshold)


# ---------------------------------------------------------------- dev split


def dev_analysis(prompts):
    """Prompt ablation, calibration and the coverage/precision trade-off on the dev split."""
    _, y, _ = load_split(HERE / "data", "test")
    ablation = {}
    for m in MODELS:
        for v in ("simple", "templates", "guided"):
            path = OUT / "probs" / f"test_{m}_{v}.npy"
            if path.exists():
                ablation.setdefault(m, {})[v] = accuracy(np.load(path).argmax(1), y)

    calibration, probs = {}, {}
    for m, v in prompts.items():
        scores = np.load(OUT / "probs" / f"test_{m}_{v}.npy")
        t = fit_temperature(scores, y)
        probs[m] = softmax(scores / t)
        calibration[m] = {"temperature": round(t, 3), "mean_conf_before": r4(softmax(scores).max(1).mean()),
                          "mean_conf_after": r4(probs[m].max(1).mean()), "accuracy": accuracy(probs[m].argmax(1), y),
                          "ece_before": prob_ece(softmax(scores), y), "ece_after": prob_ece(probs[m], y)}

    mean = np.mean(list(probs.values()), axis=0)
    vote, conf = mean.argmax(1), mean.max(1)
    n_agree = np.sum([p.argmax(1) == vote for p in probs.values()], axis=0)
    sweep = []
    for t in THRESHOLDS:
        ok = accept_mask(conf, n_agree, len(probs), t)
        sweep.append({"threshold": t, "coverage": r4(ok.mean()), "accepted_accuracy": accuracy(vote[ok], y[ok])})
    return {"prompt_ablation": ablation, "calibration": calibration, "vote_accuracy": accuracy(vote, y),
            "review_threshold_sweep": sweep}


# ---------------------------------------------------------------- train split (human labels available)


def human_side_info():
    """Labeling time from CIFAR-10N's side information (seconds each worker spent per batch of 10).

    The random_label1..3 columns cannot be matched to the Worker1..3 ids (per-worker accuracy is
    indistinguishable from a shuffled-id baseline), so only time and worker counts are used.
    """
    path = HERE / "data" / "side_info_cifar10N.csv"
    if not path.exists():
        torch.hub.download_url_to_file(SIDE_INFO_URL, str(path))
    side = pd.read_csv(path)
    start = side["Image-batch"].str.split("--").str[0].astype(int).to_numpy()
    assert (start == np.arange(len(side)) * 10).all(), "expected consecutive batches of 10 images"
    times = side[[f"Work-time-in-seconds-{k}" for k in (1, 2, 3)]].to_numpy()
    sec = times.ravel() / 10
    return {
        "seconds_per_image_mean": r4(sec.mean()), "seconds_per_image_median": r4(np.median(sec)),
        "worker_hours_for_3_labels_per_image": round(times.sum() / 3600, 1),
        "distinct_workers": int(pd.unique(side[[f"Worker{k}-id" for k in (1, 2, 3)]].to_numpy().ravel()).size),
    }


def train_analysis(df, cfg, thresholds, tag):
    models = list(cfg["prompts"])
    y = df.true_label.map(TO_INT).to_numpy()
    pred = {c: df[c].map(TO_INT).to_numpy() for c in WORKERS + ["human_majority"]}
    pred |= {m: df[f"{m}_label"].map(TO_INT).to_numpy() for m in models}
    pred["vote"] = df.vote_label.map(TO_INT).to_numpy()
    conf = {m: df[f"{m}_conf"].to_numpy() for m in models} | {"vote": df.vote_conf.to_numpy()}
    majority = pred["human_majority"]
    crowd = np.concatenate([pred[w] for w in WORKERS])
    y3 = np.tile(y, 3)

    labelers = {}
    for k, p in pred.items():
        labelers[k] = {"accuracy": accuracy(p, y), "agreement_with_crowd_majority": accuracy(p, majority),
                       "kappa_with_crowd_majority": r4(cohen_kappa_score(p, majority))}
    labelers["single_crowd_label"] = {"accuracy": accuracy(crowd, y3),
                                      "agreement_with_crowd_majority": accuracy(crowd, np.tile(majority, 3)),
                                      "kappa_with_crowd_majority": r4(cohen_kappa_score(crowd, np.tile(majority, 3)))}

    per_class = {}
    for k, (p, t) in {"single_crowd_label": (crowd, y3), "human_majority": (majority, y),
                      **{m: (pred[m], y) for m in models + ["vote"]}}.items():
        per_class[k] = {c: accuracy(p[t == i], t[t == i]) for i, c in enumerate(CLASSES)}

    n_crowd_right = sum(pred[w] == y for w in WORKERS)
    by_crowd = {}
    for n in range(4):
        m = n_crowd_right == n
        by_crowd[f"{n}_of_3_crowd_labels_correct"] = {"images": int(m.sum()),
                                                      **{k: accuracy(pred[k][m], y[m]) for k in models + ["vote"]}}

    curves = {}
    for k, c in conf.items():
        order = np.argsort(-c, kind="stable")
        cum = np.cumsum(pred[k][order] == y[order]) / np.arange(1, len(y) + 1)
        curves[k] = {f"{cov:.2f}": r4(cum[int(cov * len(y)) - 1]) for cov in (0.5, 0.7, 0.8, 0.9, 0.95, 1.0)}

    n_agree = df.n_agree.to_numpy()
    by_agreement = {f"{n}_of_{len(models)}_models_agree_with_vote": {
        "share": r4((n_agree == n).mean()), "vote_accuracy": accuracy(pred["vote"][n_agree == n], y[n_agree == n])}
        for n in range(1, len(models) + 1)}
    calibration = {k: ece(c, pred[k] == y) for k, c in conf.items()}

    # Hard majority vote for comparison: 2-of-3 wins, otherwise the most confident model.
    stacked = np.stack([pred[m] for m in models], 1)
    most_confident = stacked[np.arange(len(y)), np.argmax(np.stack([conf[m] for m in models], 1), 1)]
    counts = np.stack([(stacked == c).sum(1) for c in range(len(CLASSES))], 1)
    hard_vote = np.where(counts.max(1) >= 2, counts.argmax(1), most_confident)

    hitl = []
    for t in thresholds:
        ok = accept_mask(conf["vote"], n_agree, len(models), t)
        row = {"threshold": t, "auto_labeled": r4(ok.mean()), "auto_label_accuracy": accuracy(pred["vote"][ok], y[ok]),
               "on_routed_images": {"model_vote": accuracy(pred["vote"][~ok], y[~ok]),
                                    "one_crowd_label": accuracy(pred["human_label1"][~ok], y[~ok]),
                                    "crowd_majority": accuracy(majority[~ok], y[~ok])}}
        for name, fallback, cost in [("one_crowd_label", pred["human_label1"], 1), ("crowd_majority", majority, 3),
                                     ("expert", y, 1)]:
            row[f"final_accuracy_route_to_{name}"] = accuracy(np.where(ok, pred["vote"], fallback), y)
            row[f"human_labels_per_image_{name}"] = r4((~ok).mean() * cost)
        hitl.append(row)

    # Model vote as a detector of wrong crowd labels: flag a crowd label when the vote disagrees with it.
    detection = {}
    for name, labels in [("crowd_majority", majority), ("one_crowd_label", pred["human_label1"])]:
        wrong = labels != y
        detection[name] = {"error_rate": r4(wrong.mean())}
        for rule, flag in [("vote_disagrees", pred["vote"] != labels),
                           ("vote_disagrees_conf_0.9", (pred["vote"] != labels) & (conf["vote"] >= 0.9))]:
            detection[name][rule] = {"flagged": r4(flag.mean()), "precision": r4(wrong[flag].mean()),
                                     "recall": r4(flag[wrong].mean())}

    unanimous_models = n_agree == len(models)
    suspect = unanimous_models & (conf["vote"] >= 0.9) & (pred["vote"] == majority) & (pred["vote"] != y)
    all_six = suspect & np.all([pred[w] == pred["vote"] for w in WORKERS], axis=0)

    ratings_h = np.stack([pred[w] for w in WORKERS], 1)
    ratings_m = np.stack([pred[m] for m in models], 1)
    consistency = {
        "crowd_all_3_agree": r4((ratings_h == ratings_h[:, :1]).all(1).mean()), "crowd_fleiss_kappa": fleiss_kappa(ratings_h),
        "models_all_agree": r4(unanimous_models.mean()),
        "models_fleiss_kappa": fleiss_kappa(ratings_m) if len(models) > 1 else None,
    }
    freq = {k: [r4(f) for f in np.bincount(p, minlength=10) / len(p)] for k, p in
            {"single_crowd_label": crowd, "human_majority": majority, **{m: pred[m] for m in models}, "vote": pred["vote"]}.items()}

    extra = {}
    if "qwen" in models:
        mass = np.exp(np.load(OUT / "probs" / f"{tag}_qwen_{cfg['prompts']['qwen']}.npy")).sum(1)
        extra["qwen_label_token_mass"] = {"mean": r4(mass.mean()), "min": r4(mass.min()), "share_below_0.9": r4((mass < 0.9).mean())}

    return {
        "labelers": labelers, "per_class_accuracy": per_class,
        "soft_vote_accuracy": accuracy(pred["vote"], y), "hard_vote_accuracy": accuracy(hard_vote, y),
        "accuracy_by_model_agreement": by_agreement, "ece_calibrated": calibration,
        "top_confusions": {"single_crowd_label": top_confusions(crowd, y3), **{k: top_confusions(pred[k], y) for k in models + ["vote"]}},
        "accuracy_by_crowd_correctness": by_crowd, "accuracy_at_coverage": curves, "human_in_the_loop": hitl,
        "crowd_label_error_detection": detection,
        "consistency": consistency, "predicted_class_frequency": freq,
        "suspected_ground_truth_errors": {"models_unanimous_and_crowd_majority_agree": int(suspect.sum()),
                                          "all_3_models_and_all_3_crowd_labels_agree": int(all_six.sum())},
        **extra,
    }, pred, conf, y, suspect


# ---------------------------------------------------------------- figures


def short(label):
    return SHORT.get(label, label)


def caption(df, i, models):
    r = df.iloc[i]
    crowd = "/".join(short(r[w]) for w in WORKERS)
    preds = "/".join(short(r[f"{m}_label"]) for m in models)
    return f"GT: {short(r.true_label)}\ncrowd: {crowd}\nmodels: {preds}\nvote: {short(r.vote_label)} {r.vote_conf:.2f}"


def example_groups(df, pred, y, suspect, rng, n_models, per_group=3):
    majority, vote = pred["human_majority"], pred["vote"]
    all_crowd_right = np.all([pred[w] == y for w in WORKERS], axis=0)
    groups = {
        "Everyone agrees": np.flatnonzero((vote == y) & all_crowd_right & (df.n_agree.to_numpy() == n_models)),
        "Models right, crowd majority wrong": np.flatnonzero((vote == y) & (majority != y)),
        "Crowd majority right, models wrong": np.flatnonzero((vote != y) & (majority == y)),
        "Models + crowd agree, CIFAR-10 label differs": np.flatnonzero(suspect),
    }
    return {name: rng.choice(ix, min(per_group, len(ix)), replace=False) for name, ix in groups.items()}


def plot_examples(images, df, groups, models):
    """The four groups as a 2 x 2 grid, so the figure fits the page width at a readable size."""
    per = max(len(ix) for ix in groups.values())
    fig = plt.figure(figsize=(2 * per * 1.5, 2 * 1.9))
    for sub, (name, ix) in zip(fig.subfigures(2, 2, wspace=0.03).flat, groups.items()):
        sub.suptitle(name, fontsize=7.5, weight="bold", y=0.985)
        axes = np.atleast_1d(sub.subplots(1, per))
        for ax, i in zip(axes, list(ix) + [None] * (per - len(ix))):
            ax.axis("off")
            if i is not None:
                ax.imshow(images[df["index"].iloc[i]], interpolation="bicubic")
                ax.set_title(caption(df, i, models), fontsize=6, loc="left", family="monospace")
        sub.subplots_adjust(left=0.01, right=0.95, top=0.63, bottom=0.01, wspace=0.15)
    fig.savefig(FIG / "examples.png", dpi=200)
    plt.close(fig)


def plot_coverage(pred, conf, y, n_agree, n_models, metrics):
    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    for k, c in conf.items():
        order = np.argsort(-c, kind="stable")
        err = 1 - np.cumsum(pred[k][order] == y[order]) / np.arange(1, len(y) + 1)
        cov = np.arange(1, len(y) + 1) / len(y)
        ax.plot(cov, 100 * np.maximum(err, 1e-4), label=DISPLAY[k], lw=2.2 if k == "vote" else 1.3)
    for t, marker in [(0.0, "o"), (0.9, "s")]:
        ok = accept_mask(conf["vote"], n_agree, n_models, t)
        rule = "all models agree" + (f" & conf >= {t}" if t else "")
        ax.plot(ok.mean(), 100 * (1 - (pred["vote"][ok] == y[ok]).mean()), marker, color="black", ms=5, label=rule)
    for k, style in [("single_crowd_label", ":"), ("human_majority", "--")]:
        err = 100 * (1 - metrics["labelers"][k]["accuracy"])
        ax.axhline(err, color="gray", ls=style, lw=1.2)
        ax.text(0.505, err * 1.1, f"{DISPLAY[k]}, all images: {err:.1f}%", fontsize=6.3, color="dimgray")
    ax.set_yscale("log")
    ax.set_yticks([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 20], ["0.01", "0.03", "0.1", "0.3", "1", "3", "10", "20"])
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.01, 25)
    ax.set_xlabel("coverage (share of images auto-labeled, most confident first)", fontsize=8)
    ax.set_ylabel("error rate vs CIFAR-10 labels (%)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6.3, loc="upper left", bbox_to_anchor=(0.0, 0.8), ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "coverage.png", dpi=170)
    plt.close(fig)


def plot_confusions(pred, y):
    crowd = np.concatenate([pred[w] for w in WORKERS])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, (title, p, t) in zip(axes, [("Single crowd label", crowd, np.tile(y, 3)), ("Model vote", pred["vote"], y)]):
        cm = pd.crosstab(t, p).reindex(index=range(10), columns=range(10), fill_value=0).to_numpy()
        cm = cm / cm.sum(1, keepdims=True)
        off = np.where(np.eye(10, dtype=bool), np.nan, cm)
        ax.imshow(off, cmap="Reds", vmin=0, vmax=0.08)
        for i in range(10):
            for j in range(10):
                if i != j and cm[i, j] >= 0.005:
                    ax.text(j, i, f"{100 * cm[i, j]:.1f}", ha="center", va="center", fontsize=6,
                            color="white" if cm[i, j] > 0.05 else "black")
        ax.set_xticks(range(10), CLASSES, rotation=60, fontsize=7)
        ax.set_yticks(range(10), CLASSES, fontsize=7)
        ax.set_title(f"{title}: off-diagonal errors (% of true class)", fontsize=8)
        ax.set_xlabel("predicted", fontsize=7)
        ax.set_ylabel("CIFAR-10 label", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "confusions.png", dpi=170)
    plt.close(fig)


def plot_suspects(images, df, suspect, rng, n=40):
    ix = np.flatnonzero(suspect)
    ix = np.sort(rng.choice(ix, min(n, len(ix)), replace=False))
    if len(ix) == 0:
        return
    cols = 8
    rows = int(np.ceil(len(ix) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.25, rows * 1.45))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), ix):
        r = df.iloc[i]
        ax.imshow(images[r["index"]], interpolation="bicubic")
        ax.set_title(f"#{r['index']}\nGT {short(r.true_label)}\nall: {short(r.vote_label)}", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIG / "suspected_label_errors.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Evaluate demo_code.py labels on the train split.")
    p.add_argument("--tag", default="train", help="evaluate outputs/labels_<tag>.csv, e.g. train or train_n2000")
    tag = p.parse_args().tag

    FIG.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((OUT / f"labels_{tag}.json").read_text())
    df = pd.read_csv(OUT / f"labels_{tag}.csv")
    images, _, _ = load_split(HERE / "data", "train")
    rng = np.random.default_rng(0)

    metrics = {"config": cfg, "dev": dev_analysis(cfg["prompts"])}
    train, pred, conf, y, suspect = train_analysis(df, cfg, THRESHOLDS, tag)
    metrics["train"] = train
    metrics["human_effort"] = human_side_info()
    run_info = json.loads((OUT / "run_info.json").read_text())
    metrics["model_throughput"] = {m: run_info.get(f"{tag}_{m}_{v}") for m, v in cfg["prompts"].items()}
    (OUT / ("metrics.json" if tag == "train" else f"metrics_{tag}.json")).write_text(json.dumps(metrics, indent=2))

    models = list(cfg["prompts"])
    groups = example_groups(df, pred, y, suspect, rng, len(models))
    plot_examples(images, df, groups, models)
    plot_coverage(pred, conf, y, df.n_agree.to_numpy(), len(models), train)
    plot_confusions(pred, y)
    plot_suspects(images, df, suspect, rng)
    examples = pd.concat([df.iloc[ix].assign(group=name) for name, ix in groups.items()])
    examples.to_csv(OUT / "examples.csv", index=False)
    df[suspect].to_csv(OUT / "suspected_label_errors.csv", index=False)

    print(json.dumps({k: metrics[k] for k in ("dev", "human_effort", "model_throughput")}, indent=1))
    print(json.dumps(train["labelers"], indent=1))
    print(f"wrote metrics and figures to {OUT}")


if __name__ == "__main__":
    main()
