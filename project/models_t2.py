import numpy as np
from sklearn.linear_model import LogisticRegression
import copy
import torch
from torch import nn


class LogReg:
      """Multinomial logistic regression (log-linear model), as in the tutorial."""

      def __init__(self, seed, max_iter=200):
          self.seed = seed
          self.max_iter = max_iter
          self.best_epoch = None   # the runner reads this; only the MLPs use it
          self.model = None

      def fit(self, X, y):
          # saga: designed for large sparse data (our X is 45k tweets x 16k-36k mostly-zero columns),
          # and with 20 classes it fits one multinomial (softmax) model, like the tutorial.
          self.model = LogisticRegression(solver="saga", max_iter=self.max_iter, random_state=self.seed)
          self.model.fit(X, y)
          return self

      def predict_proba(self, X):
          # shape (n_tweets, 20): one probability per emoji, each row sums to 1
          return self.model.predict_proba(X)

class Net(nn.Module):
      """Feed-forward network: input -> [Linear -> ReLU -> Dropout] x len(hidden_dims) -> Linear -> 20 logits.

      hidden_dims=[128]      -> the tutorial's MLP (1 hidden layer)
      hidden_dims=[256, 128] -> our deep MLP (2 hidden layers)
      """

      def __init__(self, input_dim, hidden_dims, num_classes, dropout):
          super().__init__()
          layers = []
          size = input_dim
          for h in hidden_dims:                  # one block per hidden layer
              layers += [nn.Linear(size, h), nn.ReLU(), nn.Dropout(dropout)]
              size = h                           # next layer's input = this layer's output
          layers.append(nn.Linear(size, num_classes))   # output layer: one score per emoji
          self.layers = nn.Sequential(*layers)

      def forward(self, x):
          return self.layers(x)    
      

class MLP:
      """Trains a Net with AdamW + early stopping. hidden_dims=[128] -> MLP, [256, 128] -> deep MLP."""

      def __init__(self, seed, hidden_dims=(128,), dropout=0.1, lr=1e-3, weight_decay=1e-4,
                   batch_size=128, max_epochs=10, patience=2, es_fraction=0.1, num_classes=20):
          self.seed = seed
          self.hidden_dims = list(hidden_dims)
          self.dropout = dropout
          self.lr = lr
          self.weight_decay = weight_decay
          self.batch_size = batch_size
          self.max_epochs = max_epochs
          self.patience = patience
          self.es_fraction = es_fraction      # share of the training fold kept aside for early stopping
          self.num_classes = num_classes
          self.best_epoch = None
          self.net = None
          self.loss_fn = nn.CrossEntropyLoss()   # = softmax + negative log-likelihood, in one stable step

      def _batches(self, X, y, shuffle, rng=None):
          """Yield (inputs, labels) mini-batches. Sparse rows are made dense one batch at a time (saves memory)."""
          order = rng.permutation(X.shape[0]) if shuffle else np.arange(X.shape[0])
          for start in range(0, len(order), self.batch_size):
              b = order[start:start + self.batch_size]
              yield (torch.tensor(X[b].toarray(), dtype=torch.float32),
                     torch.tensor(y[b], dtype=torch.long))

      def _avg_loss(self, X, y):
          self.net.eval()                     # dropout OFF while measuring
          total = 0.0
          with torch.no_grad():               # no learning here, so no gradients needed
              for xb, yb in self._batches(X, y, shuffle=False):
                  total += self.loss_fn(self.net(xb), yb).item() * len(yb)
          return total / len(y)

      def fit(self, X, y):
          torch.manual_seed(self.seed)        # same seed -> same initial weights and dropout
          rng = np.random.default_rng(self.seed)

          # 1. split the training fold: 90% to learn from, 10% to decide when to stop
          perm = rng.permutation(X.shape[0])
          n_es = int(len(perm) * self.es_fraction)
          es_ids, tr_ids = perm[:n_es], perm[n_es:]
          X_tr, y_tr, X_es, y_es = X[tr_ids], y[tr_ids], X[es_ids], y[es_ids]

          # 2. build the network and the optimizer
          self.net = Net(X.shape[1], self.hidden_dims, self.num_classes, self.dropout)
          optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

          # 3. train, epoch by epoch, keeping the best weights
          best_loss, best_state, bad_epochs = float("inf"), None, 0
          for epoch in range(self.max_epochs):
              self.net.train()                # dropout ON while learning
              for xb, yb in self._batches(X_tr, y_tr, shuffle=True, rng=rng):
                  optimizer.zero_grad()                     # forget last batch's gradients
                  loss = self.loss_fn(self.net(xb), yb)     # forward + loss
                  loss.backward()                           # backward: compute gradients
                  optimizer.step()                          # update the weights

              es_loss = self._avg_loss(X_es, y_es)
              print(f"    epoch {epoch}: early-stopping loss {es_loss:.4f}")
              if es_loss < best_loss:
                  best_loss, bad_epochs, self.best_epoch = es_loss, 0, epoch
                  best_state = copy.deepcopy(self.net.state_dict())   # a real copy, not a reference
              else:
                  bad_epochs += 1
                  if bad_epochs >= self.patience:
                      break

          # 4. go back to the best epoch's weights
          self.net.load_state_dict(best_state)
          return self

      def predict_proba(self, X):
          self.net.eval()
          dummy = np.zeros(X.shape[0], dtype=int)   # _batches wants labels; they're not used here
          out = []
          with torch.no_grad():
              for xb, _ in self._batches(X, dummy, shuffle=False):
                  out.append(torch.softmax(self.net(xb), dim=1).numpy())
          return np.concatenate(out)              # logits (raw scores); softmax happens in the loss / predict_proba


MODEL_FACTORIES = {
      "logistic": lambda seed: LogReg(seed=seed),
      "mlp":      lambda seed: MLP(seed=seed, hidden_dims=[128], dropout=0.1),
      "deep_mlp": lambda seed: MLP(seed=seed, hidden_dims=[256, 128], dropout=0.3),
  }

if __name__ == "__main__":
      # Quick sanity check on a small sample (only runs when you run this file directly)
      from sklearn.metrics import f1_score
      from tokenizers import Tokenizer
      from project.lib import load_tweeteval_emoji, BagOfWords

      train, _, _, names = load_tweeteval_emoji(clean=True)
      tok = Tokenizer.from_file("results/project/tokenizer_vs16000_clean.json")

      texts = [t for t, _ in train[:3000]]
      y = np.array([l for _, l in train[:3000]])
      X = BagOfWords(tok).transform(texts)
      print("X:", X.shape, " y:", y.shape)

      model = LogReg(seed=0).fit(X[:2500], y[:2500])   # train on 2500 tweets
      probs = model.predict_proba(X[2500:])            # predict the other 500
      pred = probs.argmax(axis=1)                      # highest probability = predicted emoji

      print("probs shape:", probs.shape)
      print("row sums:", probs.sum(axis=1)[:3])
      print("macro-F1:", round(f1_score(y[2500:], pred, average="macro"), 3))
      print("example:", texts[2500], "→ predicted", names[pred[0]], "| true", names[y[2500]])
      mlp = MLP(seed=0, hidden_dims=[128]).fit(X[:2500], y[:2500])
      pred = mlp.predict_proba(X[2500:]).argmax(axis=1)
      print("MLP best epoch:", mlp.best_epoch, " macro-F1:", round(f1_score(y[2500:], pred, average="macro"), 3))