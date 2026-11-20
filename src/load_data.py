"""
load_data.py -- Load and describe the SMS Spam Collection (Iteration 2, Section 7).

Expected file (not committed to Git, see README):
    data/SMSSpamCollection   UCI version: tab-separated "label<TAB>text", no header
or  data/spam.csv            Kaggle version: columns v1 (label), v2 (text), latin-1 encoding

Run from the repository root:   python src/load_data.py
Outputs: printed description, figures/data_*.png, figures/data_description.txt
"""
import os
import sys
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from methods import my_tokenize


def tokenize(text):
    """Iteration 2 tokenizer (keeps apostrophes) so this script's numbers stay unchanged."""
    return my_tokenize(text, keep_apostrophes=True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
lines = []


def report(msg=""):
    print(msg)
    lines.append(str(msg))


def load_sms(data_dir=DATA):
    """Return a DataFrame with columns label ('ham'/'spam'), text, y (0 = ham, 1 = spam)."""
    uci = os.path.join(data_dir, "SMSSpamCollection")
    kaggle = os.path.join(data_dir, "spam.csv")
    if os.path.exists(uci):
        df = pd.read_csv(uci, sep="\t", header=None, names=["label", "text"],
                         quoting=3, encoding="utf-8")          # quoting=3: keep quote marks in texts
        source = "UCI SMSSpamCollection"
    elif os.path.exists(kaggle):
        df = pd.read_csv(kaggle, encoding="latin-1").iloc[:, :2]
        df.columns = ["label", "text"]
        source = "Kaggle spam.csv"
    else:
        sys.exit("Dataset not found. See README.md -> 'Getting the data'.")
    df["label"] = df["label"].str.strip().str.lower()
    df["y"] = (df["label"] == "spam").astype(int)
    df.attrs["source"] = source
    return df


def stratified_split(y, test_frac=0.2, seed=0):
    """80/20 split that keeps the spam proportion equal in train and test."""
    rng = np.random.default_rng(seed)
    test = []
    for c in (0, 1):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        test.extend(idx[: int(round(test_frac * len(idx)))])
    test = np.sort(np.array(test))
    train = np.setdiff1d(np.arange(len(y)), test)
    return train, test


if __name__ == "__main__":
    df = load_sms()
    report(f"=== SMS Spam Collection ({df.attrs['source']}) ===")
    report(f"rows (messages): {len(df)}, raw columns: 2 (label: categorical ham/spam, text: string)")
    report(f"missing values per column:\n{df[['label', 'text']].isna().sum().to_string()}")
    counts = df["label"].value_counts()
    report(f"class counts: ham = {counts.get('ham', 0)}, spam = {counts.get('spam', 0)}, "
           f"spam share = {df['y'].mean():.4f}")

    dup = df.duplicated(subset="text")
    report(f"exact duplicate texts: {dup.sum()} (unique messages: {(~dup).sum()}); "
           f"spam share among unique = {df.loc[~dup, 'y'].mean():.4f}")

    toks = df["text"].map(tokenize)
    df["n_tokens"] = toks.map(len)
    df["n_chars"] = df["text"].str.len()
    report("tokens per message by class:\n"
           + df.groupby("label")["n_tokens"].describe().round(2).to_string())
    vocab_all = {t for ts in toks for t in ts}
    report(f"vocabulary size (all messages, our tokenizer): {len(vocab_all)}")
    report(f"empty messages after tokenizing: {(df['n_tokens'] == 0).sum()}")

    # Shape of the document-term matrix and its sparsity (computed without building it)
    nnz = sum(len(set(ts)) for ts in toks)
    report(f"document-term matrix X: {len(df)} x {len(vocab_all)}, "
           f"non-zero entries: {nnz} ({nnz / (len(df) * len(vocab_all)):.4%} dense)")

    train, test = stratified_split(df["y"].to_numpy())
    report(f"stratified split (seed 0): train {len(train)} (spam {df['y'].iloc[train].mean():.4f}), "
           f"test {len(test)} (spam {df['y'].iloc[test].mean():.4f})")
    np.savez(os.path.join(DATA, "split_indices.npz"), train=train, test=test)

    # Plot 1: message length by class
    plt.figure(figsize=(5.5, 3.5))
    bins = np.arange(0, df["n_tokens"].quantile(0.99) + 2)
    for lab, col in [("ham", "tab:blue"), ("spam", "tab:red")]:
        plt.hist(df.loc[df["label"] == lab, "n_tokens"], bins=bins, alpha=0.6,
                 density=True, label=lab, color=col)
    plt.xlabel("tokens per message"); plt.ylabel("fraction of messages"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "data_length_by_class.png"), dpi=200); plt.close()

    # Plot 2: most frequent words in each class
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, lab, col in [(axes[0], "ham", "tab:blue"), (axes[1], "spam", "tab:red")]:
        cnt = Counter(t for ts in toks[df["label"] == lab] for t in ts)
        total = sum(cnt.values())
        words, freqs = zip(*cnt.most_common(15))
        ax.barh(words[::-1], np.array(freqs[::-1]) / total, color=col)
        ax.set_title(f"{lab}: top 15 words"); ax.set_xlabel(r"$n_{w,c}/N_c$")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "data_top_words.png"), dpi=200); plt.close()

    with open(os.path.join(FIG, "data_description.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\nSaved figures/data_length_by_class.png, figures/data_top_words.png, "
          "figures/data_description.txt, data/split_indices.npz")
