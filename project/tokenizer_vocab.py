"""
Choose the BPE vocabulary size.

Protocol: for each vocab size, train BPE on the 45k training tweets (text only), encode with bag-of-words,
train logistic regression on the full training split, and score macro-F1 on the official 5k VALIDATION
split. The test set is not used, and neither are the CV folds, so this choice cannot leak into later results.

Usage:  .venv/bin/python -m project.tokenizer_vocab
Writes  results/project/tokenizer_vocab.json  and  results/project/figures/vocab_size.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from project.lib import BagOfWords, LogLinear, load_tweeteval_emoji, metrics, train_bpe

OUT = Path("results/project")
SIZES = [500, 1000, 2000, 4000, 8000, 16000, 32000]


def main():
    # clean=True: the validation split was (mostly) released WITHOUT the U+FE0F leak while train/test have it,
    # so a fair comparison removes the leak everywhere (see project/explore_data.py).
    train, val, _, _ = load_tweeteval_emoji(clean=True)
    texts, y = [t for t, _ in train], np.array([l for _, l in train])
    vtexts, vy = [t for t, _ in val], np.array([l for _, l in val])
    rows = []
    for V in SIZES:
        tok = train_bpe(texts, vocab_size=V)
        enc = BagOfWords(tok)
        X, Xv = enc.transform(texts), enc.transform(vtexts)
        tokens_per_tweet = X.sum() / X.shape[0]
        used = int((X.sum(axis=0) > 0).sum())
        rare = int(((X > 0).sum(axis=0) < 5).sum())          # vocabulary entries seen in < 5 tweets
        m = metrics(vy, LogLinear(seed=0).fit(X, y).predict_proba(Xv))
        rows.append({"vocab_size": V, "actual_vocab": tok.get_vocab_size(), "tokens_per_tweet": round(float(tokens_per_tweet), 2),
                     "features_seen_in_<5_tweets": rare, **{k: round(v, 4) for k, v in m.items()}})
        print(rows[-1], flush=True)

    (OUT / "tokenizer_vocab.json").write_text(json.dumps(rows, indent=1))
    fig, ax1 = plt.subplots(figsize=(6, 3.4))
    xs = [r["vocab_size"] for r in rows]
    ax1.plot(xs, [100 * r["macro_f1"] for r in rows], "o-", c="#4C72B0", label="validation macro-F1")
    ax1.set_xscale("log"); ax1.set_xticks(xs, [str(v) for v in xs]); ax1.minorticks_off()
    ax1.set_xlabel("BPE vocabulary size"); ax1.set_ylabel("macro-F1 (%)", color="#4C72B0")
    ax2 = ax1.twinx()
    ax2.plot(xs, [r["tokens_per_tweet"] for r in rows], "s--", c="#DD8452", label="tokens per tweet")
    ax2.set_ylabel("tokens per tweet", color="#DD8452")
    ax1.set_title("Vocabulary size (logistic + BoW, leak removed, validation split)")
    fig.tight_layout(); fig.savefig(OUT / "figures" / "vocab_size.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
