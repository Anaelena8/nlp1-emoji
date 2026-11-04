# Team guide: NLP1 Project 1 (TweetEval emoji)

This file explains **everything done so far**, written for someone who is new to this.
Read it top to bottom once, then use the table of contents to come back to parts.

**Deadline:** Friday 6 Nov, 23:59 on Canvas: **slides PDF** (max 6 slides) + **link to this public repository**.
**Oral:** either of us can be asked about *anything*, so both of us need to understand both halves.

---

## Table of contents
1. [The big picture](#1-the-big-picture)
2. [Who does what](#2-who-does-what)
3. [How to open and run the project (VS Code)](#3-how-to-open-and-run-the-project-vs-code)
4. [What we did, step by step (Teammate 1's half)](#4-what-we-did-step-by-step-teammate-1s-half)
5. [Deep dive: the tokenizer](#5-deep-dive-the-tokenizer)
6. [Deep dive: the leak and "inflation"](#6-deep-dive-the-leak-and-inflation)
7. [Which file does what](#7-which-file-does-what)
8. [Teammate 2: your part, step by step](#8-teammate-2-your-part-step-by-step)
9. [The 6 slides](#9-the-6-slides)
10. [Oral exam preparation](#10-oral-exam-preparation)
11. [AI-use log (required for the blog post)](#11-ai-use-log-required-for-the-blog-post)

---

## 1. The big picture

We are teaching a computer to **read a tweet and guess which emoji the person used**.

Researchers collected **100,000 real tweets**. Each originally contained exactly one of **20 popular emojis**
(❤ 😍 😂 💕 🔥 😊 😎 ✨ 💙 😘 📷 🇺🇸 ☀ 💜 😉 💯 😁 🎄 📸 😜). They **deleted the emoji** from the text and wrote it down separately as the answer:

> **Tweet:** "Man these are the funniest kids ever!! That face!"  →  **Answer:** 😂

The pipeline (the same as in the course tutorial):

```
tweet ──► chop into pieces ──► turn into numbers ──► model learns ──► 20 probabilities ──► pick highest ──► emoji
          (tokenizer)          (features)            (architecture)
          └──────── Teammate 1 ────────┘             └─────────────── Teammate 2 ───────────────┘
```

The tweets come in **three piles**:

| Pile | How many | What it's for | Analogy |
|---|---|---|---|
| **Train** | 45,000 | the model **learns** from these (cross-validation happens here) | studying with an answer key |
| **Validation** | 5,000 | used only to **make choices** (e.g. tokenizer size) | practice test |
| **Test** | 50,000 | used **once at the end** for the final numbers | final exam: no peeking before |

---

## 2. Who does what

The project's "Get to work!" list has **6 items**, and we get **one slide per item**. We split them **3 + 3**:

| # | "Get to work!" item | Owner | Status |
|---|---|---|---|
| 1 | Choose a dataset and understand it | **Teammate 1** | ✅ done |
| 2 | Tokenizer: train one? which settings? | **Teammate 1** | ✅ done |
| 3 | Features: how to turn tweets into numbers | **Teammate 1** | ✅ done |
| 4 | Architectures: logistic regression, MLP, + one of our own | **Teammate 2** | ⏳ to do |
| 5 | Evaluation: cross-validation, metric, Bayesian ROPE | **Teammate 2** | ⏳ to do |
| 6 | Analysis: which system is best? | **Teammate 2** | ⏳ to do |

Both together, at the end: merge the slides into one PDF, push to GitHub, submit on Canvas, and **teach each other your half**.

---

## 3. How to open and run the project (VS Code)

Do one step, check you see what it says, then go to the next.

### A. Install VS Code (one time)
1. Go to **https://code.visualstudio.com** → **Download for macOS** (or your system).
2. Open the downloaded file and drag **Visual Studio Code** into **Applications**. Open it.

### B. Get the project and its Python environment (one time)
1. Download / clone this repository and open a terminal **inside the project folder**.
2. Install `uv` (a fast Python installer): see https://docs.astral.sh/uv/getting-started/installation/
3. Create the environment and install the libraries (about 1–2 GB, a few minutes):
   ```bash
   uv venv --python 3.14 .venv
   ```
   ```bash
   uv pip install --python .venv/bin/python -r requirements.txt
   ```
   This creates a hidden folder `.venv` with Python and all libraries inside.

### C. Open the folder in VS Code
1. **File → Open Folder…** → choose the project folder (**the main folder**, not `part1` or `project`).
2. If asked *"Do you trust the authors?"* → **Yes**.
3. On the left you now see the file list (`part1`, `project`, `results`, …).

### D. Install two extensions (one time)
1. Click the **four-squares icon** on the far left (Extensions), or press `Cmd + Shift + X`.
2. Search **Python** (by Microsoft) → **Install**.
3. Search **Jupyter** (by Microsoft) → **Install**.

### E. Open the walkthrough notebook and connect it to our Python
1. In the file list open **`project/walkthrough_teammate1.ipynb`**.
2. **Top-right corner** → **Select Kernel** → **Python Environments…** → choose **`.venv`** (`./.venv/bin/python`).
   - If VS Code says *"requires the ipykernel package"* → click **Install**.

### F. Run it
1. Optional: click **Clear All Outputs** at the top so you can watch the results appear.
2. Click in the first grey code cell → press **`Shift + Enter`**. This runs it and moves to the next one.
3. **Read the text above each cell, then run it.** Go down one cell at a time.
   - The first data cell downloads the tweets (needs internet, takes 10–30 s the first time).

**Common error:** `ModuleNotFoundError: No module named 'project'` → you opened the wrong folder (redo C) or the kernel isn't `.venv` (redo E).

💡 To read a `.md` file nicely formatted in VS Code: open it and press `Cmd + Shift + V`.

### G. Re-running the scripts (optional)
VS Code menu **Terminal → New Terminal**, then (from the main folder):

| Command | What it does | Time |
|---|---|---|
| `.venv/bin/python -m project.explore_data` | data statistics + charts | ~30 s |
| `.venv/bin/python -m project.tokenizer_vocab` | tries 7 tokenizer sizes | ~3 min |
| `.venv/bin/python -m project.leakage` | the leak experiment | ~2 min |
| `.venv/bin/python -m project.make_slides_t1` | rebuilds slides 1–3 | ~5 s |
| `.venv/bin/python -m project.run_experiments` | **Teammate 2:** all 9 systems with cross-validation | ~20 min |

---

## 4. What we did, step by step (Teammate 1's half)

### Step 1: Looked at the data
We counted how many tweets each emoji has. It's **very uneven**:
- ❤ is **21%** of all tweets (about 1 in 5)
- 😍 and 😂 are about 10–11% each
- 💯 and 🇺🇸 are only about **2%** each

Other facts: tweets are short (**median 12 words**), **50%** contain a location tag like `@ San Diego` (Instagram-style posts), and **47%** contain a hashtag.

📊 Chart: `results/project/figures/class_distribution.png`

### Step 2: Chose how to grade the models (the metric)
**Accuracy** = "what % did it get right?" That sounds fair, but imagine a lazy model that **always answers ❤**. Because ❤ is the answer 21% of the time, it gets **21.6% accuracy** while learning nothing.

**Macro-F1** = compute a score **for each emoji separately**, then average the 20 scores. Every emoji counts equally, rare or common. The lazy "always ❤" model gets only **1.8%** macro-F1, because it completely fails the other 19 emojis.

👉 **Decision: macro-F1 is our main metric** (accuracy is reported alongside). It is also the official metric for this dataset.

| Baseline (on test) | Accuracy | Macro-F1 |
|---|---|---|
| always ❤ | 21.6% | 1.8% |
| random guess, using the class proportions | 8.5% | 4.9% |
| random guess, uniform | 5.1% | 4.6% |

### Step 3: Found a hidden "cheat" in the data 🔍
On a computer, some emojis are stored as **two pieces**: the emoji itself + an **invisible helper character** called `U+FE0F` ("draw me in colour"). ❤ and ☀ use this helper.

When the researchers deleted the emojis, they deleted the visible part but **forgot the invisible helper**. So:
- **19.8%** of training tweets still contain the invisible character
- **86%** of those tweets are ❤ (most of the rest are ☀)
- The train and test piles contain it, but the **validation pile was cleaned** (only 1.6%). The piles are **inconsistent**.

A model can learn a lazy trick: *"invisible character → answer ❤"*. That has nothing to do with understanding language. This is called a **label leak**. Section 6 shows how big its effect is.

👉 **Decision: remove the invisible character from every tweet in all three piles before any experiment.**

📊 Chart: `results/project/figures/leak_by_class.png`

### Step 4: Trained a tokenizer (chops tweets into pieces)
We trained a **new BPE tokenizer** on our training tweets and tested **7 sizes** to choose one. Result: **16,000 pieces**. Details in Section 5.

### Step 5: Built three ways to turn tweets into numbers (features)
Computers only understand numbers, so each tweet becomes a **long list of numbers**, with **one position per piece** in the tokenizer's vocabulary.

Example tweet: **"so so happy with my best friend"**

| Method | What it writes down | For this tweet | Size |
|---|---|---|---|
| **Token indicator** (tutorial) | 1 if the piece appears, 0 if not | `▁so=1 ▁happy=1 ▁with=1 ▁my=1 ▁best=1 ▁friend=1` | 16,000 |
| **Bag-of-words** (tutorial) | **how many times** each piece appears | `▁so=2` …(rest the same) | 16,000 |
| **BoW + bigrams** (**our own**) | the counts **plus pairs of neighbouring pieces** | … + `▁best+▁friend=1`, `▁so+▁happy=1`, … | 36,000 |

**Why add pairs (bigrams)?** The first two methods **forget word order**: "not good" and "good not" look identical. Pairs keep a little order: "not good" becomes its own feature, and `San+Diego` becomes one feature.
We only keep the **20,000 most common pairs** (seen at least 3 times in training). All possible pairs would be 16,000 × 16,000 = 256 million, far too many.

💡 **"Sparse"**: a tweet only uses about 17 of the 16,000 pieces, so almost every number is 0. The code only **stores the non-zero numbers**. That's all "sparse" means.

---

## 5. Deep dive: the tokenizer

### What is a tokenizer?
It **chops text into pieces** (tokens). We use **BPE** (Byte-Pair Encoding), the same as the tutorial. It starts from single letters and keeps **gluing together the pair of neighbours that appears most often**:

```
start:         l o w   l o w e r   l o w e s t
glue "l"+"o":  lo w    lo w e r    lo w e s t
glue "lo"+"w": low     low e r     low e s t
... repeat until the vocabulary is full
```

Common words become one piece; rare words get split into known parts. So **no word is ever completely unknown**.

### Did we train a new one? **Yes.**
The tutorial's tokenizer was trained on **news articles**. Tweets look very different (hashtags, `@user`, location tags, slang, "soooo"), so a news tokenizer would chop them into strange pieces. Ours learns pieces that actually appear in tweets: `#blessed` becomes one piece.

### The settings ("hyperparameters")
A **hyperparameter** is a setting **we choose before training**. The computer doesn't learn it.

| Setting | Meaning | Our choice | How we decided |
|---|---|---|---|
| Algorithm | the chopping method | **BPE** | same as the tutorial |
| Training data | what it learns from | **the 45,000 training tweets** (cleaned) | never test tweets; uses only the text, not the answers |
| **Vocabulary size** | max number of different pieces | **16,000** | ✅ **tested** (see below) |
| Minimum frequency | a pair is only glued if seen at least this often | **2** | from the tutorial, not tested |
| Pre-tokenizer | the first rough split before BPE | **Metaspace** | from the tutorial, not tested |
| Unknown symbol | written for characters never seen | **`[UNK]`** | from the tutorial |
| Lowercasing | turn "LOVE" into "love"? | **no** | our choice: capitals can show emotion ("SO HAPPY"); not tested |

### What is a pre-tokenizer?
A **rough first cut** of the text **before** BPE starts (*pre* = before).
Without it, BPE could glue letters **across two words**, e.g. learning a nonsense piece `sa` from "i**s a**". The pre-tokenizer first splits at spaces, so BPE only glues **inside** words.

Ours, **Metaspace**, does two things:
1. splits the text at every space
2. marks each word start with `▁`, so we remember where words began

```
"Best day ever"  →  ▁Best | ▁day | ▁ever     (BPE then works inside each chunk)
```

The `▁` mark lets the model tell `▁cat` (the word *cat*) apart from `cat` inside *lo**cat**ion*, and it lets us turn the pieces back into text (`▁` → space).

### What is minimum frequency?
*"Only glue a pair if it appears **at least N times** in the training tweets."* We used **N = 2**.

| Pair | Times seen | Glued? (N = 2) |
|---|---|---|
| `▁lo` + `ve` | thousands | ✅ → `▁love` |
| `▁xq` + `zt` | 1 | ❌ probably a typo |

It stops one-time typos from wasting vocabulary slots. With 45,000 tweets almost every useful pair appears far more than twice, so in our case it only filters out noise.

### How we chose the vocabulary size: `project/tokenizer_vocab.py`
For each of **7 sizes** (500 → 32,000) the script does the same **5-step recipe**:
1. train a tokenizer of that size on the training tweets
2. turn tweets into numbers with bag-of-words
3. train a simple model (logistic regression)
4. let it guess the emojis of the **validation** tweets (practice test, not the final exam)
5. grade it with macro-F1

Only the size changes, so any score difference is **caused by the size**. That's a fair comparison.

| Size | Pieces per tweet | Macro-F1 |
|---|---|---|
| 500 | 72 | 8.9 ❌ |
| 1,000 | 33 | 17.5 |
| 2,000 | 25 | 19.7 |
| 4,000 | 21 | 20.3 |
| 8,000 | 19 | 21.4 |
| **16,000** | 17 | **22.2** ✅ |
| 32,000 | 15 | 22.6 |

What it shows:
1. **500 is terrible.** Tweets already contain **about 557 different characters** (letters, accents, symbols), so the whole budget goes on single characters and no real words are ever learned.
2. **Bigger is better, but the gains shrink.** 500 → 1,000 adds about 9 points; 16,000 → 32,000 adds only 0.4.
3. **Too big has a cost.** At 32,000, **13,234 pieces appear in fewer than 5 tweets**, so there's almost no data to learn what they mean.

👉 **Decision: 16,000**: almost as good as 32,000, with half the size and far fewer useless rare pieces.

📊 Chart: `results/project/figures/vocab_size.png`

### Answers to the project sheet's tokenizer questions
| Question | Answer |
|---|---|
| Use the tutorial's tokenizer? | No, it was trained on news; tweets are too different |
| Train a new one? | Yes, BPE on our tweets |
| On what data? | the 45,000 training tweets (cleaned) |
| Is the dataset too small? | No: 45,000 tweets ≈ 530,000 words; scores still improve up to 32,000 pieces |
| Which hyperparameters? | vocab **16,000** (tested); min frequency **2** and Metaspace (from the tutorial) |

---

## 6. Deep dive: the leak and "inflation"

### What does "inflation" mean?
**A score that looks higher than it deserves.** Like a student who scores 90% because the answers were faintly printed on the back of the exam paper, when their real knowledge deserves 60%. The extra 30% is inflated.

### What `project/leakage.py` does
It measures **how much the invisible character fools us** and **whether the model depends on it**.
It trains the **same model** (bag-of-words + bigrams, logistic regression) **4 times**, flipping two switches:
1. **While studying (training):** does the model see the invisible character?
2. **During the exam (test):** does the exam contain it?

Results (test set, average of 3 runs):

| # | Studied **with** it? | Exam **has** it? | Macro-F1 | F1 for ❤ | What it means |
|---|---|---|---|---|---|
| A | ✅ | ✅ | **25.6** | **82** | the trick works, so the score looks great (**inflated**) |
| B | ✅ | ❌ | 18.7 | **13** | the trick is taken away and ❤ collapses |
| C | ❌ | ✅ | 19.4 | 40 | never learned the trick, so the character doesn't matter |
| D | ❌ | ❌ | **19.5** | 39 | the **honest** score |

**How to read it:**
- **A vs D:** the same model scores 25.6 with the cheat and 19.5 honestly. **About 6 points are inflation**: points earned by spotting a character, not by understanding language.
- **B:** the model trained with the cheat **relies on it**. Without the character, its ❤ score drops from 82 to **13**, worse than a model that never saw the trick (39). It had learned *"no invisible character → not ❤"*. This is called **shortcut learning**.
- **C vs D:** a model trained without the character ignores it (19.4 vs 19.5), so it's safe.

👉 That's why **every experiment we report uses the cleaned tweets**.

📊 Chart: `results/project/figures/leakage__bow+bigrams__logistic.png`

---

## 7. Which file does what

```
.
├── README.md                     course setup instructions + short project intro
├── TEAM_GUIDE.md                 ← this file
├── requirements.txt              list of Python libraries
├── part1/                        COURSE files (tutorial + project instructions + helper code)
├── project/                      OUR code
│   ├── walkthrough_teammate1.ipynb   ⭐ start here: Teammate 1's work, step by step, runnable
│   ├── lib.py                        the toolbox (used by all scripts below)
│   ├── explore_data.py               steps 1–3: data statistics, baselines, leak check
│   ├── tokenizer_vocab.py            step 4: tries 7 tokenizer sizes
│   ├── leakage.py                    section 6: the leak experiment
│   ├── run_experiments.py            Teammate 2: all 9 systems with cross-validation
│   └── make_slides_t1.py             builds slides 1–3
└── results/project/              everything the code produced (don't edit by hand)
    ├── figures/                      all charts (PNG)
    ├── slides_teammate1.pdf          slides 1–3
    ├── data_stats.json               numbers from explore_data.py
    ├── tokenizer_vocab.json          numbers from tokenizer_vocab.py
    ├── leakage/                      numbers from leakage.py
    └── tokenizer_vs16000*.json       the trained tokenizers (the chopping rules)
```

Course files: `part1/tutorial.ipynb` (tutorial), `part1/project.ipynb` (project instructions), `part1/*.py` (tutorial helper code). We changed **one line** in `part1/clslib.py` to fix a bug (`TokenIndicator.__len__` used a global variable instead of `self.tokenizer`).

### `lib.py`, the toolbox
It doesn't run anything by itself. It holds the tools the other scripts use, like a kitchen drawer of knives and pans, while the scripts are the recipes.

| Group | Tool | What it does | Owner |
|---|---|---|---|
| 1. Get data | `load_tweeteval_emoji(clean=…)` | downloads the tweets, returns the 3 piles as `(text, label)` pairs + the 20 emojis | T1 |
| | `clean_leak(text)` | deletes the invisible cheat character from one tweet | T1 |
| | `kfold_ids(...)` | shuffles the training tweets and cuts them into 5 groups for cross-validation | T1 |
| 2. Chop | `train_bpe(texts, vocab_size)` | trains the BPE tokenizer | T1 |
| 3. Numbers | `TokenIndicator`, `BagOfWords`, `BagOfNgrams` | the 3 feature methods; each turns a list of tweets into a sparse table of numbers | T1 |
| 4. Models | `LogLinear` | logistic regression (tutorial) | **T2** |
| | `MLP(hidden_dims=(128,))` | neural network with 1 hidden layer (tutorial) | **T2** |
| | `MLP(hidden_dims=(256,128), dropout=0.3)` | deeper network, 2 hidden layers (our own) | **T2** |
| | `MODEL_FACTORIES` | the "menu" of models by name: `logistic`, `mlp`, `deep_mlp` | **T2** |
| 5. Grade | `metrics(answers, probabilities)` | returns macro-F1 and accuracy | T1 |

Every model has `fit(X, y)` (learn) and `predict_proba(X)` (give 20 probabilities per tweet; the highest is the guess).

How the scripts use it:
```
explore_data.py    → load_tweeteval_emoji
tokenizer_vocab.py → train_bpe + BagOfWords + LogLinear + metrics   (7 times)
leakage.py         → clean_leak + train_bpe + BagOfNgrams + LogLinear + metrics   (with / without leak)
run_experiments.py → everything: 3 feature methods × 3 models × 5 folds × 3 seeds
```

---

## 8. Teammate 2: your part, step by step

Everything you need from Teammate 1 is ready: clean data, the 16k tokenizer and the 3 feature methods.

### Step 1: Understand the 3 models (`project/lib.py`, group 4)
| Model | Idea | Settings |
|---|---|---|
| **Logistic regression** (tutorial) | one weight per piece per emoji; linear | sklearn, solver `saga`, max 200 iterations |
| **MLP** (tutorial) | 1 hidden layer of 128 units + ReLU; learns combinations of pieces | dropout 0.1, AdamW, learning rate 0.001, weight decay 0.0001, batch 128 |
| **Deep MLP** (our own) | 2 hidden layers (256 → 128) + more dropout | dropout 0.3, otherwise the same |

Both MLPs stop training early: they keep **10% of the training fold aside** and stop when the loss on it stops improving (at most 10 rounds, patience 2). The tutorial instead picked the best round using the same fold it then reported, which makes its scores slightly too optimistic.

### Step 2: Run all 9 systems (about 20 minutes)
```bash
.venv/bin/python -m project.run_experiments
```
It runs **3 feature methods × 3 models = 9 systems**, each with **5-fold cross-validation × 3 seeds = 15 scores**. Cleaned tweets and the 16k tokenizer are the defaults. Outputs, in `results/project/main/`:
- `cv_results.json`: one line per system/seed/fold with `macro_f1` and `accuracy`
- `test_preds/`: each model's guesses on the test tweets (for the ensemble in step 5)
- `y_test.npy`: the true test answers

If it stops halfway, just run it again: finished parts are skipped.

### Step 3: Box plot of the 15 scores per system
`part1.analysis.plot_box_and_points` from the tutorial works, or plot with pandas/seaborn from `cv_results.json`.

### Step 4: Bayesian comparison with a ROPE
- Use `part1.analysis.BenavoliCorrelatedT(scores_A, scores_B, runs=3, names=[...])`. **Pass `runs=3`** because we use 3 seeds.
- **ROPE proposal: ±0.01 macro-F1 (1 point)**. Justify it yourself, e.g. "differences under 1 point are within seed-to-seed noise and wouldn't change which system we'd use".
- Compare at least: logistic vs MLP, MLP vs deep MLP, bag-of-words vs bigrams, and best vs second-best.
- Report P(A better), P(practically equal), P(B better) + the posterior plot.

### Step 5: Choose the best system, error analysis, final test
- Confusion matrix + F1 per emoji: which emojis get mixed up? (we expect ❤ / 💕 / 😍 and 📷 / 📸)
- **Ensemble**: majority vote of the best system's 15 models, scored **once** on the test set.
- Tell Teammate 1 the best system, who reruns the leak experiment on it:
  `.venv/bin/python -m project.leakage --encoder <encoder> --model <model>` then `.venv/bin/python -m project.make_slides_t1 --leak_system <encoder>__<model>`.

### Something we already noticed
In an early quick test, the MLPs had **higher accuracy but lower macro-F1** than logistic regression: they lean towards ❤. Worth checking in your results and discussing on slide 6.

---

## 9. The 6 slides

One slide per "Get to work!" item. These are **backup slides** for answering the TA's questions, not a presentation.

| # | Title | Owner | Content | Status |
|---|---|---|---|---|
| 1 | **Dataset** | T1 | task, 3 piles, imbalance, "always ❤" baseline, the leak + its effect | ✅ `results/project/slides_teammate1.pdf` |
| 2 | **Tokenizer** | T1 | new BPE on tweets, settings, vocabulary-size curve, 16k | ✅ same PDF |
| 3 | **Features** | T1 | indicator, bag-of-words, BoW + bigrams, worked example | ✅ same PDF |
| 4 | **Architectures** | T2 | logistic, MLP, deep MLP + settings and why | ⏳ |
| 5 | **Evaluation** | T2 | 5-fold CV × 3 seeds, macro-F1, box plot, Bayesian ROPE results | ⏳ |
| 6 | **Analysis / best model** | T2 | best system, confusion matrix, which emojis are hard, final test score | ⏳ |

---

## 10. Oral exam preparation

### Questions about Teammate 1's half
| Likely question | Short answer |
|---|---|
| Why macro-F1, not accuracy? | Classes are imbalanced; "always ❤" gets 21.6% accuracy but 1.8% macro-F1. Macro-F1 counts every emoji equally. |
| Why train your own tokenizer? | Tweets differ from the tutorial's news (hashtags, slang). Trained on the training tweets only, no labels, never test. |
| Why 16,000? | Tested 7 sizes on validation: gains shrink after 16k, and at 32k 13k pieces appear in < 5 tweets. |
| What is a pre-tokenizer / min frequency? | Section 5: split at spaces before BPE so it never glues across words; only glue pairs seen ≥ 2 times. |
| Which settings did you *not* test? | min frequency, pre-tokenizer, lowercasing: natural next steps. |
| What do bigrams add, and what do they miss? | Local word order ("not good"); still miss long-range order and make the space bigger and sparser. |
| How did you find the leak? Why does it matter? | Looked at which characters go with which labels. With the leak the score is inflated by about 6 points and the model relies on a shortcut instead of language. |

### Questions about Teammate 2's half
- Why does an MLP sometimes beat logistic regression? (it can combine features: non-linear)
- What do dropout / weight decay / early stopping do? (all fight overfitting)
- Why cross-validation, and why several seeds? (more than one score → you see the spread, not one lucky number)
- Why Bayesian estimation instead of a p-value? How did you pick the ROPE? How do you read the posterior plot?
- Why ensemble? (combining many models is usually more stable and a bit better)

### Questions about the tutorial
These can also be asked. The **Questions** blocks inside `part1/tutorial.ipynb` are the examples the TA uses. Go through them together.

---

## 11. AI-use log (required for the blog post)

The project asks us to record what we delegated to AI assistants. Fill in the last column honestly.

| What | Tool | What we checked / changed ourselves |
|---|---|---|
| Explaining the tutorial and the project, suggesting datasets, splitting the work | Claude Code (Anthropic) | *(fill in)* |
| Code in `project/` (data loading, encoders, experiment scripts, slides script) | Claude Code | *(fill in)* |
| Finding the `U+FE0F` leak and the inconsistent validation split | Claude Code flagged it; confirmed with the per-label scan in `explore_data.py` | *(fill in)* |
| This guide and the walkthrough notebook | Claude Code | *(fill in)* |
| *(Teammate 2: add your entries)* | | |
