"""
methods.py -- From-scratch Multinomial Naive Bayes, one function per algorithm step.

Each function implements one numbered step of the algorithm in the report
(Iteration 2, Section 8; Iteration 3, Table "Code-to-math map") and uses the same
symbols. Only NumPy is used here. Libraries (scikit-learn, scipy) appear only in the
check scripts, never in this file.

    X      (n x |V|)  document-term count matrix, X[d, w] = x_dw
    Y      (n x 2)    one-hot labels, Y[d, c] = 1{y_d = c}; column 0 = ham, 1 = spam
    C      (2 x |V|)  class-word counts, C[c, w] = n_{w,c}
    N      (2,)       class word totals N_c
    log_theta (2 x |V|)  log of smoothed theta_{w|c}
    log_prior (2,)       log pi_c
    S      (m x 2)    class log-scores of m messages
"""
import re
import numpy as np

# ----------------------------------------------------------------------------- Step 0
# Cleaning: lower-case, delete apostrophes ("don't" -> "dont"), replace every other
# character outside [a-z0-9] by a space. Tokens are the remaining runs of [a-z0-9].
_APOS = re.compile(r"['\u2019]")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_ITER2_TOKEN = re.compile(r"[a-z0-9']+")


def my_tokenize(text, keep_apostrophes=False):
    """Step 0a: message text -> list of tokens w_1, ..., w_L.
    keep_apostrophes=True reproduces the Iteration 2 tokenizer exactly."""
    text = text.lower()
    if keep_apostrophes:
        return _ITER2_TOKEN.findall(text)
    return _NON_ALNUM.sub(" ", _APOS.sub("", text)).split()


def my_build_vocab(token_lists):
    """Step 0b: V = set of all tokens in the TRAINING messages (sorted -> column order)."""
    vocab = sorted({tok for toks in token_lists for tok in toks})
    return {w: j for j, w in enumerate(vocab)}


def my_count_matrix(token_lists, vocab):
    """Step 0c: X[d, w] = x_dw = number of times word w occurs in message d.
    Tokens not in V are dropped (closed vocabulary, assumption A6)."""
    X = np.zeros((len(token_lists), len(vocab)), dtype=np.int64)
    for d, toks in enumerate(token_lists):
        for tok in toks:
            j = vocab.get(tok)
            if j is not None:
                X[d, j] += 1
    return X


# ----------------------------------------------------------------------------- Step 1
# Y[d, c] = 1{y_d = c}
def my_one_hot(y, k=2):
    """Y in {0,1}^{n x 2}."""
    y = np.asarray(y, dtype=int)
    Y = np.zeros((len(y), k), dtype=np.int64)
    Y[np.arange(len(y)), y] = 1
    return Y


# C = Y^T X,   C[c, w] = n_{w,c} = sum_{d: y_d = c} x_dw
def my_class_counts(X, y):
    """(2 x n)(n x |V|) = 2 x |V| class-word count matrix."""
    return my_one_hot(y).T @ np.asarray(X)


# ----------------------------------------------------------------------------- Step 2
# N = C 1,   N_c = sum_w n_{w,c}
def my_class_totals(C):
    """(2 x |V|)(|V| x 1) = 2-vector of word totals per class."""
    return C @ np.ones(C.shape[1])


# ----------------------------------------------------------------------------- Step 3
# log theta_{w|c} = log(n_{w,c} + alpha) - log(N_c + alpha |V|)     (Theorem 4)
def my_log_likelihoods(C, N, alpha=1.0):
    """2 x |V| matrix log Theta. With alpha = 0 a zero count gives log 0 = -inf."""
    V = C.shape[1]
    with np.errstate(divide="ignore"):
        return np.log(C + alpha) - np.log(N + alpha * V)[:, None]


# ----------------------------------------------------------------------------- Step 4
# log pi_c = log(n_c / n),   n_c = (Y^T 1)_c                         (Theorem 2)
def my_log_priors(y):
    """2-vector of log class priors (maximum-likelihood estimate)."""
    Y = my_one_hot(y)
    with np.errstate(divide="ignore"):
        return np.log((Y.T @ np.ones(len(Y))) / len(Y))


# ----------------------------------------------------------------------------- Step 5
# S = X (log Theta)^T + 1 (log pi)^T                                 (Theorem 6)
def my_scores(X, log_theta, log_prior):
    """(m x |V|)(|V| x 2) + broadcast prior = m x 2 log-scores, linear in the counts.
    A word with count 0 contributes nothing (theta^0 = 1), even if log theta = -inf."""
    X = np.asarray(X)
    finite = np.isfinite(log_theta)
    S = X @ np.where(finite, log_theta, 0.0).T + log_prior
    hits_zero = ((X > 0).astype(int) @ (~finite).T.astype(int)) > 0
    S[hits_zero] = -np.inf                     # x_w >= 1 with theta_w = 0  ->  log 0
    return S


# w = log theta_1 - log theta_0,   b = log pi_1 - log pi_0;  spam iff w^T x + b > 0
def my_linear_rule(log_theta, log_prior):
    """Weight vector w (|V|,) and bias b of the equivalent linear classifier."""
    return log_theta[1] - log_theta[0], log_prior[1] - log_prior[0]


# ----------------------------------------------------------------------------- Step 6
# y_hat_d = argmax_c S_dc                                            (Proposition 5)
def my_predict(S):
    """Predicted class of each message."""
    return np.argmax(S, axis=1)


# P(c | x) = exp(s_c - M - log sum_k exp(s_k - M)),  M = max_k s_k   (Proposition 9)
def my_posterior(S):
    """m x 2 posterior probabilities via log-sum-exp (no underflow, rows sum to 1)."""
    M = S.max(axis=1, keepdims=True)
    log_Z = M + np.log(np.exp(S - M).sum(axis=1, keepdims=True))
    return np.exp(S - log_Z)


# ----------------------------------------------------------------------------- Step 7
# CM[i, j] = #{y = i, y_hat = j}:   [[TN, FP], [FN, TP]]               (Definition 10)
def my_confusion_matrix(y_true, y_pred):
    """2 x 2 confusion matrix, rows = true class, columns = predicted class.
    Computed as a matrix product of one-hot matrices: CM = Y_true^T Y_pred."""
    return my_one_hot(y_true).T @ my_one_hot(y_pred)


# precision = TP / (TP + FP) = P(Y=1 | Y_hat=1);  recall = TP / (TP + FN) = P(Y_hat=1 | Y=1)
def my_precision_recall(cm):
    """(precision, recall) from the confusion matrix (Proposition 7). nan if undefined."""
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    prec = tp / (tp + fp) if tp + fp > 0 else np.nan
    rec = tp / (tp + fn) if tp + fn > 0 else np.nan
    return prec, rec


# (p_hat + z^2/2N  +/-  z sqrt(p_hat(1-p_hat)/N + z^2/4N^2)) / (1 + z^2/N)   (Theorem 10)
def my_wilson(k, N, z=1.959963984540054):
    """95% Wilson score interval for k successes in N trials."""
    if N == 0:
        return (np.nan, np.nan)
    p = k / N
    centre = (p + z**2 / (2 * N)) / (1 + z**2 / N)
    half = z * np.sqrt(p * (1 - p) / N + z**2 / (4 * N**2)) / (1 + z**2 / N)
    return (centre - half, centre + half)


# ----------------------------------------------------------------------------- wrappers
def my_nb_fit(X, y, alpha=1.0):
    """Steps 1-4. Returns the fitted model as a dict."""
    C = my_class_counts(X, y)
    N = my_class_totals(C)
    return {"C": C, "N": N, "alpha": alpha,
            "log_lik": my_log_likelihoods(C, N, alpha),
            "log_prior": my_log_priors(y)}


def my_nb_predict(model, X):
    """Steps 5-6. Returns (predictions, joint log-scores S = log P(c) + sum_w x_w log P(w|c))."""
    S = my_scores(X, model["log_lik"], model["log_prior"])
    return my_predict(S), S


# ----------------------------------------------------------------------------- Q5 check
# Pairs of distinct token positions (j != k) inside the class-c messages:
#   P_hat(w1, w2 | c) = sum_d x_dw1 x_dw2 / sum_d L_d (L_d - 1)        (w1 != w2)
# Under conditional independence E[x_dw1 x_dw2 | L_d] = L_d (L_d - 1) theta_w1 theta_w2,
# so the ratio P_hat(w1, w2 | c) / (theta_hat_w1 theta_hat_w2) should be close to 1.
def my_pair_ratio_matrix(X, y, c):
    """|V| x |V| matrix of P_hat(w1, w2 | c) / (P_hat(w1|c) P_hat(w2|c)), from the
    co-occurrence matrix X_c^T X_c. Diagonal and pairs with a zero count are nan."""
    Xc = np.asarray(X)[np.asarray(y) == c].astype(float)
    L = Xc.sum(axis=1)
    pairs = Xc.T @ Xc / np.sum(L * (L - 1))           # joint, off-diagonal entries
    theta = Xc.sum(axis=0) / Xc.sum()                  # unsmoothed MLE n_{w,c} / N_c
    with np.errstate(divide="ignore", invalid="ignore"):
        R = pairs / np.outer(theta, theta)
    np.fill_diagonal(R, np.nan)
    return R, pairs, theta
