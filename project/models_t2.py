import numpy as np
from sklearn.linear_model import LogisticRegression


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