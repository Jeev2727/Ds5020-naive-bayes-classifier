"""
nb_core.py -- From-scratch Multinomial Naive Bayes (NumPy only).

Every function maps to one step of the algorithm in the report
(Section "Algorithm as Math"). Symbols follow the notation table:

    X      (n x |V|)  document-term count matrix
    Y      (n x 2)    one-hot label matrix, column 0 = ham, column 1 = spam
    C      (2 x |V|)  class-word counts, C = Y^T X
    Theta  (2 x |V|)  smoothed word probabilities theta_{w|c}
    pi     (2,)       class priors
"""
import re
import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text):
    """Lower-case a message and split it into word tokens."""
    return TOKEN_RE.findall(text.lower())


def build_vocab(token_lists):
    """Vocabulary V built from the TRAINING messages only (sorted for reproducibility)."""
    vocab = sorted({tok for toks in token_lists for tok in toks})
    return {w: j for j, w in enumerate(vocab)}


def count_matrix(token_lists, vocab):
    """Document-term matrix X: X[d, w] = number of times word w occurs in message d.
    Words that are not in the training vocabulary are ignored."""
    X = np.zeros((len(token_lists), len(vocab)), dtype=np.int64)
    for d, toks in enumerate(token_lists):
        for tok in toks:
            j = vocab.get(tok)
            if j is not None:
                X[d, j] += 1
    return X


def one_hot(y, k=2):
    """Y[d, c] = 1 if message d has class c."""
    Y = np.zeros((len(y), k), dtype=np.int64)
    Y[np.arange(len(y)), y] = 1
    return Y


def fit(X, y, alpha=1.0):
    """Steps 1-4: counts, totals, smoothed log-probabilities, log-priors.
    Returns (log_theta (2 x |V|), log_prior (2,), C (2 x |V|))."""
    Y = one_hot(y)
    C = Y.T @ X                                    # step 1: (2 x n)(n x |V|)
    N = C.sum(axis=1, keepdims=True)               # step 2: N_c
    V = X.shape[1]
    with np.errstate(divide="ignore"):             # alpha = 0 may give log 0 = -inf
        log_theta = np.log((C + alpha) / (N + alpha * V))   # step 3
        log_prior = np.log(Y.sum(axis=0) / len(y))          # step 4
    return log_theta, log_prior, C


def scores(X, log_theta, log_prior):
    """Step 5: S = X (log Theta)^T + 1 (log pi)^T, shape (m x 2). Linear in the counts."""
    finite = np.isfinite(log_theta)
    S = X @ np.where(finite, log_theta, 0.0).T + log_prior
    # With alpha = 0 some theta are 0 (log = -inf). A word with count 0 contributes
    # theta^0 = 1 (nothing), but a word that occurs gives log 0 = -inf for that class.
    hits_zero = ((X > 0).astype(int) @ (~finite).T.astype(int)) > 0
    S[hits_zero] = -np.inf
    return S


def posterior(S):
    """Step 6: P(c | x) from log-scores with the log-sum-exp trick (no underflow)."""
    m = S.max(axis=1, keepdims=True)
    log_Z = m + np.log(np.exp(S - m).sum(axis=1, keepdims=True))
    return np.exp(S - log_Z)


def confusion(y_true, y_pred):
    """Step 7: (TP, FP, FN, TN) with spam = 1 as the positive class."""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    return tp, fp, fn, tn


def wilson(k, n, z=1.959963984540054):
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return (centre - half, centre + half)
