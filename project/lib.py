"""
Project 1 library: TweetEval emoji prediction (20 classes).

Everything here follows the tutorial's design (encoder -> probabilistic model -> most-probable-class),
but encoders work on *batches* of texts and return SciPy sparse matrices, so that we never
materialise a dense [num_texts x vocab] matrix (that would need ~1GB of RAM for SST-2).

Encoders (representation spaces)
    TokenIndicator : 1 if a BPE token occurs in the text, else 0          (tutorial)
    BagOfWords     : number of times each BPE token occurs                (tutorial exercise)
    BagOfNgrams    : BoW counts  +  counts of frequent adjacent token pairs (our choice)

Architectures
    logistic : log-linear softmax regressor (scikit-learn, saga)          (tutorial)
    mlp      : 1 hidden layer (128, ReLU, dropout 0.1)                    (tutorial)
    deep_mlp : 2 hidden layers (256 -> 128, ReLU, dropout 0.3)            (our choice)
"""
import time
from collections import Counter

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.trainers import BpeTrainer


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------

LEAK_CHAR = "️"   # VARIATION SELECTOR-16, left behind when ❤️ / ☀️ were removed from the tweets


def clean_leak(text):
    """Remove the leftover emoji variation selector (U+FE0F) and collapse the extra spaces."""
    return " ".join(text.replace(LEAK_CHAR, " ").split())


def load_tweeteval_emoji(clean=False):
    """
    Returns (train, validation, test, class_names); each split is a list of (text, label).
      train      45,000 tweets -> cross-validation (5 folds x several seeds)
      validation  5,000 tweets -> model-selection decisions (tokenizer vocab size, MLP early stopping)
      test       50,000 tweets -> touched once, for the final reported scores
    clean=True removes the U+FE0F label leak (see project/explore_data.py).
    """
    from datasets import load_dataset
    d = load_dataset("cardiffnlp/tweet_eval", "emoji")
    names = d["train"].features["label"].names
    f = clean_leak if clean else (lambda t: t)
    splits = [list(zip([f(t) for t in d[s]["text"]], d[s]["label"])) for s in ("train", "validation", "test")]
    return (*splits, names)


def kfold_ids(n, k, seed):
    """Shuffle 0..n-1 with `seed` and split into k folds (list of index arrays)."""
    ids = np.random.default_rng(seed).permutation(n)
    return np.array_split(ids, k)


# ----------------------------------------------------------------------------------------------
# Tokenizer
# ----------------------------------------------------------------------------------------------

def train_bpe(texts, vocab_size=2000, min_frequency=2, unk_token="[UNK]"):
    """Same recipe as part1.experiment.train_tokenizer (BPE on Metaspace pre-tokens)."""
    tok = Tokenizer(BPE(unk_token=unk_token))
    tok.pre_tokenizer = Metaspace()
    trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=[unk_token])
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


# ----------------------------------------------------------------------------------------------
# Encoders
# ----------------------------------------------------------------------------------------------

def _rows_to_csr(rows, num_cols):
    """rows: list of Counter(col -> value). Returns a float32 CSR matrix."""
    indptr, indices, data = [0], [], []
    for r in rows:
        indices.extend(r.keys())
        data.extend(r.values())
        indptr.append(len(indices))
    return sp.csr_matrix((np.array(data, dtype=np.float32), np.array(indices, dtype=np.int64), np.array(indptr)),
                         shape=(len(rows), num_cols))


class Encoder:
    name = "encoder"

    def __len__(self):
        raise NotImplementedError

    def transform(self, texts) -> sp.csr_matrix:
        raise NotImplementedError

    def __call__(self, text):
        """Tutorial-style API: one text -> one dense vector."""
        return self.transform([text]).toarray()[0]


class TokenIndicator(Encoder):
    name = "indicator"

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __len__(self):
        return self.tokenizer.get_vocab_size()

    def transform(self, texts):
        encs = self.tokenizer.encode_batch(list(texts))
        return _rows_to_csr([Counter({i: 1.0 for i in set(e.ids)}) for e in encs], len(self))


class BagOfWords(Encoder):
    name = "bow"

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __len__(self):
        return self.tokenizer.get_vocab_size()

    def transform(self, texts):
        encs = self.tokenizer.encode_batch(list(texts))
        return _rows_to_csr([Counter(e.ids) for e in encs], len(self))


class BagOfNgrams(Encoder):
    """
    Bag-of-words counts plus counts of adjacent token pairs (bigrams).
    Only the `max_bigrams` most frequent bigrams of the fitting corpus get a dimension
    (all V^2 pairs would be 4M dimensions, almost all of them never observed).
    Motivation: bigrams keep a little word order, e.g. "not good" vs "good".
    """
    name = "bow+bigrams"

    def __init__(self, tokenizer, corpus, max_bigrams=20000, min_count=3):
        self.tokenizer = tokenizer
        self.V = tokenizer.get_vocab_size()
        counts = Counter()
        for e in tokenizer.encode_batch(list(corpus)):
            counts.update(zip(e.ids, e.ids[1:]))
        kept = [bg for bg, c in counts.most_common(max_bigrams) if c >= min_count]
        self.bigram_index = {bg: self.V + j for j, bg in enumerate(kept)}

    def __len__(self):
        return self.V + len(self.bigram_index)

    def transform(self, texts):
        rows = []
        for e in self.tokenizer.encode_batch(list(texts)):
            c = Counter(e.ids)
            for bg in zip(e.ids, e.ids[1:]):
                j = self.bigram_index.get(bg)
                if j is not None:
                    c[j] += 1
            rows.append(c)
        return _rows_to_csr(rows, len(self))


# ----------------------------------------------------------------------------------------------
# Models: every model exposes fit(X, y) and predict_proba(X)
# ----------------------------------------------------------------------------------------------

class LogLinear:
    name = "logistic"

    def __init__(self, max_iter=200, seed=0):
        self.m = LogisticRegression(solver="saga", max_iter=max_iter, random_state=seed)

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)


class SparseMLPNet(nn.Module):
    """
    Generalisation of part1.smlp.SparseMLP to any number of hidden layers.
    hidden_dims=[128]      -> the tutorial's 1-hidden-layer MLP
    hidden_dims=[256, 128] -> our deeper variant
    """

    def __init__(self, input_dim, hidden_dims, num_classes, dropout):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_dim, hidden_dims[0]))
        self.bias = nn.Parameter(torch.zeros(hidden_dims[0]))
        nn.init.xavier_uniform_(self.weight)
        layers = [nn.ReLU(), nn.Dropout(dropout)]
        for a, b in zip(hidden_dims, hidden_dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
        layers += [nn.Linear(hidden_dims[-1], num_classes), nn.LogSoftmax(dim=-1)]
        self.rest = nn.Sequential(*layers)

    def forward(self, x):
        return self.rest(torch.sparse.mm(x, self.weight) + self.bias)


def _to_torch(X):
    X = X.tocoo()
    idx = torch.tensor(np.vstack([X.row, X.col]), dtype=torch.long)
    return torch.sparse_coo_tensor(idx, torch.tensor(X.data, dtype=torch.float32), size=X.shape).coalesce()


class MLP:
    """
    Trains with AdamW + NLL loss like part1.smlp, with one methodological change:
    the epoch to keep is chosen on an *inner* 10% dev split of the training fold, not on the
    evaluation fold itself (the tutorial does the latter, which makes CV scores slightly optimistic).
    """

    def __init__(self, hidden_dims=(128,), dropout=0.1, lr=1e-3, weight_decay=1e-4, batch_size=128,
                 max_epochs=10, patience=2, seed=0, name="mlp"):
        self.hidden_dims, self.dropout = list(hidden_dims), dropout
        self.lr, self.wd, self.bs = lr, weight_decay, batch_size
        self.max_epochs, self.patience, self.seed = max_epochs, patience, seed
        self.name = name

    def _loss(self, X, y):
        self.net.eval()
        tot = 0.0
        with torch.no_grad():
            for s in range(0, X.shape[0], 1024):
                lp = self.net(_to_torch(X[s:s + 1024]))
                tot += nn.functional.nll_loss(lp, torch.tensor(y[s:s + 1024]), reduction="sum").item()
        return tot / X.shape[0]

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        y = np.asarray(y)
        perm = rng.permutation(X.shape[0])
        n_dev = X.shape[0] // 10
        dev, tr = perm[:n_dev], perm[n_dev:]
        Xtr, ytr, Xdev, ydev = X[tr], y[tr], X[dev], y[dev]

        self.net = SparseMLPNet(X.shape[1], self.hidden_dims, int(y.max()) + 1, self.dropout)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        best, best_state, bad = np.inf, None, 0
        self.history = []
        for epoch in range(self.max_epochs):
            self.net.train()
            order = rng.permutation(Xtr.shape[0])
            for s in range(0, len(order), self.bs):
                b = order[s:s + self.bs]
                opt.zero_grad()
                loss = nn.functional.nll_loss(self.net(_to_torch(Xtr[b])), torch.tensor(ytr[b]))
                loss.backward()
                opt.step()
            dev_loss = self._loss(Xdev, ydev)
            self.history.append(dev_loss)
            if dev_loss < best:
                best, bad = dev_loss, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        self.best_epoch = int(np.argmin(self.history))
        return self

    def predict_proba(self, X):
        self.net.eval()
        out = []
        with torch.no_grad():
            for s in range(0, X.shape[0], 1024):
                out.append(self.net(_to_torch(X[s:s + 1024])).exp().numpy())
        return np.vstack(out)


MODEL_FACTORIES = {
    "logistic": lambda seed: LogLinear(seed=seed),
    "mlp": lambda seed: MLP(hidden_dims=(128,), dropout=0.1, seed=seed, name="mlp"),
    "deep_mlp": lambda seed: MLP(hidden_dims=(256, 128), dropout=0.3, seed=seed, name="deep_mlp"),
}


def metrics(y_true, probs):
    """Macro-F1 is our main metric (classes are imbalanced: ❤ = 21%, 💯 = 2%); accuracy is reported alongside."""
    pred = probs.argmax(axis=1)
    return {"macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y_true, pred))}


def timed(f, *a, **kw):
    t = time.time()
    r = f(*a, **kw)
    return r, time.time() - t
