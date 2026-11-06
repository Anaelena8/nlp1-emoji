"""
How much does the U+FE0F label leak inflate results?

2 x 2 design, for one system (encoder + model), 3 seeds:
    train on RAW tweets  (leak present)   x   evaluate on RAW test   (leak present)    -> what a naive pipeline reports
    train on RAW tweets                   x   evaluate on CLEAN test (leak removed)    -> does the model depend on the shortcut?
    train on CLEAN tweets                 x   evaluate on RAW test
    train on CLEAN tweets                 x   evaluate on CLEAN test                   -> honest performance
Each model is trained on all 45k training tweets. The test set is used for ANALYSIS here, not for choosing
anything, so this does not bias the main results.

Usage:  .venv/bin/python -m project.leakage --encoder bow+bigrams --model logistic
Writes  results/project/leakage/<encoder>__<model>.json  and  results/project/figures/leakage__<encoder>__<model>.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

from project.lib import MODEL_FACTORIES, clean_leak, load_tweeteval_emoji, metrics
from project.run_experiments import DEFAULT_VOCAB, get_tokenizer, make_encoders

OUT = Path("results/project")
HEART, SUN = 0, 12   # label ids of ❤ and ☀ (the two classes whose emoji left U+FE0F behind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="bow+bigrams")
    ap.add_argument("--model", default="logistic")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    raw_train, _, raw_test, _ = load_tweeteval_emoji(clean=False)
    y_train = np.array([l for _, l in raw_train])
    y_test = np.array([l for _, l in raw_test])
    test_sets = {"raw test": [t for t, _ in raw_test],
                 "clean test": [clean_leak(t) for t, _ in raw_test]}

    rows = []
    for train_cond, clean in [("trained on raw", False), ("trained on clean", True)]:
        texts = [clean_leak(t) if clean else t for t, _ in raw_train]
        tok = get_tokenizer(texts, DEFAULT_VOCAB, clean)
        enc = make_encoders(tok, texts)[args.encoder]()
        X = enc.transform(texts)
        X_tests = {name: enc.transform(tt) for name, tt in test_sets.items()}
        for seed in args.seeds:
            model = MODEL_FACTORIES[args.model](seed).fit(X, y_train)
            for test_cond, Xt in X_tests.items():
                pred = model.predict_proba(Xt)
                per_class = f1_score(y_test, pred.argmax(1), average=None, zero_division=0)
                rows.append({"train": train_cond, "test": test_cond, "seed": seed,
                             **metrics(y_test, pred),
                             "f1_red_heart": float(per_class[HEART]), "f1_sun": float(per_class[SUN]),
                             "f1_other_18": float(np.delete(per_class, [HEART, SUN]).mean())})
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)

    # summary: mean over seeds
    summary = {}
    for tr in ["trained on raw", "trained on clean"]:
        for te in ["raw test", "clean test"]:
            sel = [r for r in rows if r["train"] == tr and r["test"] == te]
            summary[f"{tr} | {te}"] = {k: round(float(np.mean([r[k] for r in sel])), 4)
                                       for k in ["macro_f1", "accuracy", "f1_red_heart", "f1_sun", "f1_other_18"]}
    name = f"{args.encoder}__{args.model}"
    (OUT / "leakage").mkdir(parents=True, exist_ok=True)
    (OUT / "leakage" / f"{name}.json").write_text(json.dumps({"system": name, "summary": summary, "runs": rows}, indent=1))
    print(json.dumps(summary, indent=1))

    # figure: grouped bars of macro-F1, F1(red_heart), F1(sun), mean F1 of the other 18 classes
    keys = ["macro_f1", "f1_red_heart", "f1_sun", "f1_other_18"]
    labels = ["macro-F1\n(all 20)", "F1 red_heart", "F1 sun", "mean F1\nother 18"]
    conds = list(summary)
    colors = ["#C44E52", "#E8A0A2", "#8FAADC", "#4C72B0"]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    w = 0.2
    for i, c in enumerate(conds):
        vals = [100 * summary[c][k] for k in keys]
        bars = ax.bar(np.arange(len(keys)) + (i - 1.5) * w, vals, w, label=c, color=colors[i])
        ax.bar_label(bars, fmt="%.0f", fontsize=7)
    ax.set_xticks(np.arange(len(keys)), labels)
    ax.set_ylabel("F1 (%) on test, mean of seeds")
    ax.set_title(f"Effect of the U+FE0F leak ({args.model} | {args.encoder})")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(OUT / "figures" / f"leakage__{name}.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
