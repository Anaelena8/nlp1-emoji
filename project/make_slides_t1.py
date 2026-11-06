"""
Build backup slides 1-3 as a 16:9 PDF.
    Slide 1  Dataset    (Get to work #1): the data, imbalance, metric consequence, the U+FE0F leak
    Slide 2  Tokenizer  (Get to work #2): BPE, which data, vocabulary size
    Slide 3  Features   (Get to work #3): token indicator, bag-of-words, BoW + bigrams
Numbers are read from the result files, so rerunning experiments keeps the slides in sync.

Usage:  .venv/bin/python -m project.make_slides_t1 [--leak_system bow+bigrams__logistic]
Writes  results/project/slides_teammate1.pdf
"""
import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT = Path("results/project")
FIG = OUT / "figures"
W, H = 13.33, 7.5           # 16:9 inches
DARK, ACCENT, MUTED = "#1F2937", "#C44E52", "#6B7280"
FOOTER = "NLP1 Project 1 · TweetEval emoji · code: project/  (lib.py, explore_data.py, tokenizer_vocab.py, leakage.py)"


def slide(pdf, number, title, bullets, images=(), right_text=None):
    """Two columns: wrapped bullets on the left (x < 0.47); figures or a monospace text box on the right."""
    fig = plt.figure(figsize=(W, H))
    fig.text(0.04, 0.92, f"{number}  {title}", fontsize=26, weight="bold", color=DARK)
    fig.add_artist(plt.Line2D([0.04, 0.96], [0.895, 0.895], color=ACCENT, lw=2))
    y = 0.855
    for b in bullets:
        indent = b.startswith("  ")
        lines = textwrap.wrap(b.strip(), 64 if indent else 58)
        text = ("– " if indent else "• ") + "\n   ".join(lines)
        fig.text(0.06 if indent else 0.04, y, text, fontsize=11 if indent else 12.5,
                 color=MUTED if indent else DARK, va="top", linespacing=1.22)
        y -= (0.0325 if indent else 0.036) * len(lines) + (0.009 if indent else 0.018)
    for path, box in images:
        ax = fig.add_axes(box)
        ax.imshow(mpimg.imread(path)); ax.axis("off")
    if right_text:
        fig.text(0.50, 0.85, right_text, fontsize=10.5, family="monospace", va="top", color=DARK,
                 bbox=dict(boxstyle="round,pad=0.8", fc="#F3F4F6", ec="#D1D5DB"))
    fig.text(0.04, 0.03, FOOTER, fontsize=9, color=MUTED)
    pdf.savefig(fig); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leak_system", default="bow+bigrams__logistic")
    args = ap.parse_args()

    stats = json.loads((OUT / "data_stats.json").read_text())
    vocab = {r["vocab_size"]: r for r in json.loads((OUT / "tokenizer_vocab.json").read_text())}
    leak = json.loads((OUT / "leakage" / f"{args.leak_system}.json").read_text())["summary"]
    maj = stats["baselines"]["test"]["always red_heart (majority)"]
    rr, cc, rc = (leak["trained on raw | raw test"], leak["trained on clean | clean test"],
                  leak["trained on raw | clean test"])
    pct = lambda v: f"{100 * v:.1f}"

    with PdfPages(OUT / "slides_teammate1.pdf") as pdf:
        # ---------------------------------------------------------------- slide 1: dataset
        slide(pdf, "1", "Dataset: TweetEval emoji (20 classes)", [
            "Task: guess which of 20 emojis a tweet contained (emoji deleted from the text)",
            f"  {stats['sizes']['train'] // 1000}k train (CV) · {stats['sizes']['validation'] // 1000}k validation (choices) · {stats['sizes']['test'] // 1000}k test (final); median {stats['words_per_tweet']['median']:.0f} words",
            "Very imbalanced: red_heart = 21% of tweets, hundred / US_flag ≈ 2%",
            f"  'always red_heart': {pct(maj['accuracy'])}% accuracy, {pct(maj['macro_f1'])}% macro-F1 → we use macro-F1",
            "Label leak: deleting ❤️/☀️ left an invisible character (U+FE0F)",
            f"  in {100 * stats['leak']['share_of_train']:.0f}% of train tweets ({100 * list(stats['leak']['label_share_among_leak_tweets'].values())[0]:.0f}% red_heart); in train + test, not validation",
            f"  macro-F1 {pct(rr['macro_f1'])} with leak vs {pct(cc['macro_f1'])} without (about 6 fake points)",
            f"  trained with leak, tested without: red_heart F1 {100 * rc['f1_red_heart']:.0f} → shortcut learning",
            "Decision: remove U+FE0F from all splits",
        ], images=[(FIG / "class_distribution.png", [0.49, 0.50, 0.49, 0.38]),
                   (FIG / f"leakage__{args.leak_system}.png", [0.49, 0.07, 0.49, 0.42])])

        # ---------------------------------------------------------------- slide 2: tokenizer
        v16, v32, v500 = vocab[16000], vocab[32000], vocab[500]
        slide(pdf, "2", "Tokenizer: BPE trained on our tweets", [
            "New tokenizer? Yes: tweets (hashtags, slang, @user) differ from the tutorial's news",
            "Data: the 45k training tweets, text only (no labels, never test); ~530k words is enough",
            "Hyperparameters: Metaspace pre-tokenizer, min_frequency = 2 (tutorial); vocab size tested:",
            "  each size: BPE → bag-of-words → logistic regression → macro-F1 on validation",
            f"  diminishing returns: 16k = {pct(v16['macro_f1'])}, 32k = {pct(v32['macro_f1'])}; at 32k, {v32['features_seen_in_<5_tweets']:,} tokens occur in < 5 tweets",
            f"  500 = character level: tweets contain ~{v500['actual_vocab']} distinct characters, so no merges happen",
            "Decision: vocabulary = 16,000 (within 0.5 pt of 32k, half the dimensions)",
            "  '#blessed @ San Diego' → ▁#blessed ▁@ ▁San ▁Diego",
        ], images=[(FIG / "vocab_size.png", [0.50, 0.20, 0.47, 0.60])])

        # ---------------------------------------------------------------- slide 3: features
        example = (
            "tweet: \"so so happy with my best friend\"\n\n"
            "token indicator  (16,000 dims, 6 non-zeros)\n"
            "  ▁so=1 ▁happy=1 ▁with=1 ▁my=1 ▁best=1 ▁friend=1\n\n"
            "bag-of-words     (16,000 dims, 6 non-zeros)\n"
            "  ▁so=2 ▁happy=1 ▁with=1 ▁my=1 ▁best=1 ▁friend=1\n\n"
            "BoW + bigrams    (36,000 dims, 12 non-zeros)\n"
            "  ▁so=2 ▁happy=1 ▁with=1 ▁my=1 ▁best=1 ▁friend=1\n"
            "  ▁so+▁so=1 ▁so+▁happy=1 ▁happy+▁with=1\n"
            "  ▁with+▁my=1 ▁my+▁best=1 ▁best+▁friend=1"
        )
        slide(pdf, "3", "Features: three representation spaces", [
            "Every tweet becomes a vector with one position per feature; stored sparsely (≈ 17 non-zeros out of 16,000)",
            "Token indicator (tutorial): 1 if the token occurs, else 0",
            "Bag-of-words (tutorial): how many times each token occurs",
            "  repetition can carry emotion: 'so so happy', 'love love love'",
            "Ours, BoW + bigrams: also counts pairs of neighbouring tokens",
            "  keeps local word order: 'not good' ≠ 'good'; '▁San+▁Diego', '▁best+▁friend' become single features",
            "  only the 20,000 most frequent pairs (seen ≥ 3 times in training) get a dimension; all V² = 256M pairs would be far too many",
            "  still misses long-range order ('not … at all good') and gets bigger and sparser",
            "All three use the same 16k BPE tokenizer, so differences come from the representation only",
            "→ compared with 3 architectures in cross-validation (slides 4–6)",
        ], right_text=example)
    print("wrote", OUT / "slides_teammate1.pdf")


if __name__ == "__main__":
    main()
