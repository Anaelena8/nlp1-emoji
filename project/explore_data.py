"""
Explore TweetEval emoji and compute trivial baselines.

Usage:  .venv/bin/python -m project.explore_data
Writes  results/project/figures/*.png  and  results/project/data_stats.json
"""
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from project.lib import LEAK_CHAR, load_tweeteval_emoji

OUT = Path("results/project")
FIG = OUT / "figures"

# Text names for plots (matplotlib cannot render colour emoji). Same order as the dataset's label ids.
EMOJI_NAMES = ["red_heart", "heart_eyes", "joy_tears", "two_hearts", "fire", "smiling", "sunglasses",
               "sparkles", "blue_heart", "kiss", "camera", "US_flag", "sun", "purple_heart", "wink",
               "hundred", "grin", "xmas_tree", "camera_flash", "wink_tongue"]


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    train, val, test, emojis = load_tweeteval_emoji()
    K = len(emojis)
    labels = [f"{e} {n}" for e, n in zip(emojis, EMOJI_NAMES)]
    stats = {"sizes": {"train": len(train), "validation": len(val), "test": len(test)}, "classes": labels}

    # --- class distribution -------------------------------------------------------------------------
    dist = {}
    for name, split in [("train", train), ("validation", val), ("test", test)]:
        c = Counter(l for _, l in split)
        dist[name] = [c[k] / len(split) for k in range(K)]
    stats["class_share"] = {s: {labels[k]: round(v, 4) for k, v in enumerate(d)} for s, d in dist.items()}

    order = np.argsort(dist["train"])[::-1]
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(K)
    ax.bar(x - 0.2, np.array(dist["train"])[order] * 100, 0.4, label="train", color="#4C72B0")
    ax.bar(x + 0.2, np.array(dist["test"])[order] * 100, 0.4, label="test", color="#DD8452")
    ax.axhline(100 / K, ls="--", c="gray", lw=1, label="balanced (5%)")
    ax.set_xticks(x, [EMOJI_NAMES[k] for k in order], rotation=60, ha="right")
    ax.set_ylabel("% of tweets"); ax.set_title("Class distribution: heavily imbalanced (red_heart = 21%)")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "class_distribution.png", dpi=150); plt.close(fig)

    # --- tweet length --------------------------------------------------------------------------------
    lengths = np.array([len(t.split()) for t, _ in train])
    stats["words_per_tweet"] = {"mean": round(float(lengths.mean()), 2), "median": float(np.median(lengths)),
                                "p95": float(np.percentile(lengths, 95)), "max": int(lengths.max())}
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(lengths, bins=range(0, 40), color="#4C72B0")
    ax.set_xlabel("words per tweet"); ax.set_ylabel("tweets"); ax.set_title(f"Tweet length (median {np.median(lengths):.0f} words)")
    fig.tight_layout(); fig.savefig(FIG / "tweet_length.png", dpi=150); plt.close(fig)

    # --- surface patterns per class: the leak, location tags, hashtags, @user ---------------------------
    def share(pred):
        out = []
        for k in range(K):
            texts = [t for t, l in train if l == k]
            out.append(sum(pred(t) for t in texts) / len(texts))
        return out
    patterns = {
        "contains U+FE0F (leak)": lambda t: LEAK_CHAR in t,
        "location tag ' @ Place'": lambda t: " @ " in t,
        "hashtag": lambda t: "#" in t,
        "@user mention": lambda t: "@user" in t,
    }
    stats["pattern_share_overall"] = {p: round(float(np.mean([f(t) for t, _ in train])), 4) for p, f in patterns.items()}
    stats["pattern_share_per_class"] = {p: {labels[k]: round(v, 4) for k, v in enumerate(share(f))}
                                        for p, f in patterns.items()}
    leak_texts = [l for t, l in train if LEAK_CHAR in t]
    stats["leak"] = {"tweets_with_leak": len(leak_texts),
                     "share_of_train": round(len(leak_texts) / len(train), 4),
                     "label_share_among_leak_tweets": {labels[k]: round(v / len(leak_texts), 4)
                                                       for k, v in Counter(leak_texts).most_common(5)}}

    leak = np.array(share(patterns["contains U+FE0F (leak)"])) * 100
    fig, ax = plt.subplots(figsize=(10, 3.6))
    colors = ["#C44E52" if v > 50 else "#8C8C8C" for v in leak[order]]
    ax.bar(x, leak[order], color=colors)
    ax.set_xticks(x, [EMOJI_NAMES[k] for k in order], rotation=60, ha="right")
    ax.set_ylabel("% of tweets"); ax.set_ylim(0, 105)
    ax.set_title("Tweets still containing the invisible character U+FE0F, per label")
    for i, v in enumerate(leak[order]):
        if v > 1:
            ax.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "leak_by_class.png", dpi=150); plt.close(fig)

    # --- 1.3 trivial baselines (on validation and test) ---------------------------------------------
    rng = np.random.default_rng(0)
    y_train = np.array([l for _, l in train])
    majority = Counter(y_train).most_common(1)[0][0]
    prior = np.bincount(y_train, minlength=K) / len(y_train)
    stats["baselines"] = {}
    for name, split in [("validation", val), ("test", test)]:
        y = np.array([l for _, l in split])
        preds = {"always red_heart (majority)": np.full_like(y, majority),
                 "random, train class proportions": rng.choice(K, size=len(y), p=prior),
                 "random, uniform": rng.integers(0, K, size=len(y))}
        stats["baselines"][name] = {
            b: {"macro_f1": round(float(f1_score(y, p, average="macro", zero_division=0)), 4),
                "accuracy": round(float(accuracy_score(y, p)), 4)} for b, p in preds.items()}

    (OUT / "data_stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    print(json.dumps({k: stats[k] for k in ["sizes", "words_per_tweet", "pattern_share_overall", "leak", "baselines"]},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
