"""
k-fold cross-validation for every (encoder, architecture) pair on TweetEval emoji.

Usage (from the repository root):
    .venv/bin/python -m project.run_experiments                   # main grid: 9 systems x 5 folds x 3 seeds, leak removed
    .venv/bin/python -m project.run_experiments --keep_leak --tag leaky ...   # same, on the raw (leaky) data

Outputs in results/project/<tag>/:
    cv_results.json        one record per (system, seed, fold): macro-F1, accuracy, fit time, best epoch
    test_preds/*.npy       predicted test labels of every fold model (int8), for the majority-vote ensemble
    y_test.npy             gold test labels
The tokenizer is shared: results/project/tokenizer_vs{V}[_clean].json
The script is resumable: finished (system, seed, fold) records are skipped.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from project.lib import (BagOfNgrams, BagOfWords, MODEL_FACTORIES, TokenIndicator, kfold_ids,
                         load_tweeteval_emoji, metrics, timed, train_bpe)

ROOT = Path("results/project")
DEFAULT_VOCAB = 16000   # chosen in project/tokenizer_vocab.py (see TEAM_GUIDE.md, section 5)


def get_tokenizer(train_texts, vocab_size, clean):
    path = ROOT / f"tokenizer_vs{vocab_size}{'_clean' if clean else ''}.json"
    if path.exists():
        return Tokenizer.from_file(str(path))
    ROOT.mkdir(parents=True, exist_ok=True)
    tok = train_bpe(train_texts, vocab_size=vocab_size)   # trained on training tweets only (no labels used)
    tok.save(str(path))
    return tok


def make_encoders(tok, train_texts):
    return {
        "indicator": lambda: TokenIndicator(tok),
        "bow": lambda: BagOfWords(tok),
        "bow+bigrams": lambda: BagOfNgrams(tok, train_texts),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--vocab_size", type=int, default=DEFAULT_VOCAB)
    ap.add_argument("--encoders", nargs="+", default=["indicator", "bow", "bow+bigrams"])
    ap.add_argument("--models", nargs="+", default=list(MODEL_FACTORIES))
    ap.add_argument("--keep_leak", action="store_true",
                    help="keep the U+FE0F label leak (default: removed from all splits; see TEAM_GUIDE.md, section 6)")
    ap.add_argument("--tag", default="main", help="sub-folder of results/project for this experiment")
    args = ap.parse_args()
    args.clean = not args.keep_leak

    out = ROOT / args.tag
    (out / "test_preds").mkdir(parents=True, exist_ok=True)

    train, _, test, names = load_tweeteval_emoji(clean=args.clean)
    texts = [t for t, _ in train]
    y = np.array([l for _, l in train])
    test_texts = [t for t, _ in test]
    np.save(out / "y_test.npy", np.array([l for _, l in test], dtype=np.int8))

    tok = get_tokenizer(texts, args.vocab_size, args.clean)
    encoders = make_encoders(tok, texts)
    print(f"[{args.tag}] clean={args.clean}  BPE vocab={tok.get_vocab_size()}  train={len(texts)}  test={len(test_texts)}")

    res_path = out / "cv_results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else []
    done = {(r["encoder"], r["model"], r["seed"], r["fold"]) for r in results}

    for enc_name in args.encoders:
        enc = encoders[enc_name]()
        X, t_enc = timed(enc.transform, texts)
        X_test = enc.transform(test_texts)
        print(f"[{enc_name}] dim={len(enc)}  nnz/tweet={X.nnz / X.shape[0]:.1f}  encode={t_enc:.1f}s", flush=True)

        for model_name in args.models:
            for seed in args.seeds:
                folds = kfold_ids(len(y), args.folds, seed)
                for f, val_ids in enumerate(folds):
                    if (enc_name, model_name, seed, f) in done:
                        continue
                    tr_ids = np.concatenate([p for j, p in enumerate(folds) if j != f])
                    model = MODEL_FACTORIES[model_name](seed)
                    _, t_fit = timed(model.fit, X[tr_ids], y[tr_ids])
                    m = metrics(y[val_ids], model.predict_proba(X[val_ids]))
                    np.save(out / "test_preds" / f"{enc_name}__{model_name}__s{seed}__f{f}.npy",
                            model.predict_proba(X_test).argmax(1).astype(np.int8))
                    rec = {"encoder": enc_name, "model": model_name, "system": f"{model_name} | {enc_name}",
                           "seed": seed, "fold": f, **m, "fit_seconds": round(t_fit, 1),
                           "best_epoch": getattr(model, "best_epoch", None)}
                    results.append(rec)
                    res_path.write_text(json.dumps(results, indent=1))
                    print(f"  {rec['system']:26s} s={seed} f={f}  macroF1={m['macro_f1']:.4f}  "
                          f"acc={m['accuracy']:.4f}  ({t_fit:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
