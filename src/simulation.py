"""
simulation.py -- Iteration 2 simulation check with a KNOWN true model.

Messages are generated exactly as the Naive Bayes model assumes:
    class  c ~ Categorical(prior)
    words  x | c ~ Multinomial(L, theta_c)      (conditional independence holds exactly)
so the true P(w | c) is known and every derivation can be checked numerically.

Run from the repository root:   python src/simulation.py
Outputs: printed checks, figures/sim_*.png, figures/simulation_results.txt
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from nb_core import fit, scores, posterior, confusion, wilson

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
lines = []


def report(msg=""):
    print(msg)
    lines.append(str(msg))


# ---------------------------------------------------------------- 1. the true model
rng = np.random.default_rng(0)
V = 20                                   # vocabulary size
theta = {0: rng.dirichlet(np.ones(V)),   # ham word probabilities
         1: rng.dirichlet(np.ones(V))}   # spam word probabilities
prior = np.array([0.8, 0.2])
n_docs, L = 2000, 3                      # messages, words per message


def generate(n, th0, th1, pr, rng):
    y = rng.choice(2, size=n, p=pr)
    X = np.array([rng.multinomial(L, th1 if c == 1 else th0) for c in y])
    return X, y


X, y = generate(n_docs, theta[0], theta[1], prior, rng)
tr, te = slice(0, 1600), slice(1600, None)
Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

log_theta, log_prior, C = fit(Xtr, ytr, alpha=1.0)
theta_hat = np.exp(log_theta)
theta_true = np.vstack([theta[0], theta[1]])

report("=== Simulation check: Multinomial Naive Bayes ===")
report(f"V = {V}, L = {L} words/message, n = {n_docs} (train 1600 / test 400), prior = {prior}")
report(f"training words per class N_c = {C.sum(axis=1)}")

# ---------------------------------------------------------------- Q2: MLE recovers theta
report("\n[Q2] estimated vs true parameters")
report(f"  prior: true {prior}, estimated {np.exp(log_prior).round(4)}")
for c, name in [(0, "ham"), (1, "spam")]:
    err = np.abs(theta_hat[c] - theta_true[c])
    report(f"  {name}: max |theta_hat - theta| = {err.max():.4f}, mean = {err.mean():.4f}")
report("  first five words (true / estimated):")
for j in range(5):
    report(f"    w{j}: ham {theta_true[0, j]:.4f} / {theta_hat[0, j]:.4f}   "
           f"spam {theta_true[1, j]:.4f} / {theta_hat[1, j]:.4f}")

# MLE (alpha = 0) equals the count ratio exactly
lt0, _, _ = fit(Xtr, ytr, alpha=0.0)
ratio = C / C.sum(axis=1, keepdims=True)
report(f"  max |exp(log theta, alpha=0) - n_wc/N_c| = {np.abs(np.exp(lt0) - ratio).max():.2e}")

# Consistency: error shrinks as the training set grows (roughly like 1/sqrt(N_c))
sizes = [100, 300, 1000, 3000, 10000, 30000]
errs = []
rng_n = np.random.default_rng(1)
for n in sizes:                          # average RMS error over 20 repetitions
    e = []
    for _ in range(20):
        Xn, yn = generate(n, theta[0], theta[1], prior, rng_n)
        lt, _, _ = fit(Xn, yn, alpha=1.0)
        e.append(np.sqrt(np.mean((np.exp(lt) - theta_true) ** 2)))
    errs.append(np.mean(e))
report("  RMS error by training size (mean of 20 runs): "
       + ", ".join(f"n={n}: {e:.4f}" for n, e in zip(sizes, errs)))
plt.figure(figsize=(5, 3.5))
plt.loglog(sizes, errs, "o-", label=r"RMS error of $\hat{\theta}$")
plt.loglog(sizes, errs[0] * np.sqrt(sizes[0] / np.array(sizes)), "--", label=r"slope $-1/2$ reference")
plt.xlabel("training messages n"); plt.ylabel("RMS error"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(FIG, "sim_theta_error_vs_n.png"), dpi=200); plt.close()

# ---------------------------------------------------------------- Q3: smoothing and zero counts
report("\n[Q3] Laplace smoothing")
report(f"  row sums of smoothed theta: {theta_hat.sum(axis=1)}")

# Force an unseen word: spam can never emit word 0
th1_zero = theta[1].copy(); th1_zero[0] = 0.0; th1_zero /= th1_zero.sum()
rng_z = np.random.default_rng(2)
Xz, yz = generate(1600, theta[0], th1_zero, prior, rng_z)
lt_a0, lp_a0, Cz = fit(Xz, yz, alpha=0.0)
lt_a1, lp_a1, _ = fit(Xz, yz, alpha=1.0)
msg = np.zeros((1, V), dtype=int); msg[0, 0] = 1; msg[0, 1] = 2   # contains word 0 once, word 1 twice
report(f"  spam training count of word 0: {Cz[1, 0]}")
report(f"  alpha = 0: log-scores (ham, spam) = {scores(msg, lt_a0, lp_a0)[0]}   <- spam is -inf")
report(f"  alpha = 1: log-scores (ham, spam) = {scores(msg, lt_a1, lp_a1)[0].round(4)}   <- finite")

# ---------------------------------------------------------------- Q4: linear decision rule
report("\n[Q4] decision rule is linear in the counts")
S = scores(Xte, log_theta, log_prior)
w = log_theta[1] - log_theta[0]
b = log_prior[1] - log_prior[0]
report(f"  max |(S_spam - S_ham) - (X w + b)| = {np.abs((S[:, 1] - S[:, 0]) - (Xte @ w + b)).max():.2e}")
pred = S.argmax(axis=1)
post = posterior(S)
report(f"  argmax of log-score == argmax of posterior for every test message: "
       f"{np.array_equal(pred, post.argmax(axis=1))}")
report(f"  posterior rows sum to 1: max |sum - 1| = {np.abs(post.sum(axis=1) - 1).max():.2e}")

# Compare with the decision rule that uses the TRUE parameters (best possible here)
S_true = scores(Xte, np.log(theta_true), np.log(prior))
acc = np.mean(pred == yte); acc_true = np.mean(S_true.argmax(axis=1) == yte)
report(f"  accuracy: estimated parameters {acc:.4f}, true parameters {acc_true:.4f}")

# ---------------------------------------------------------------- Q5: precision / recall
report("\n[Q5] precision and recall as conditional probabilities")
tp, fp, fn, tn = confusion(yte, pred)
report(f"  confusion matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
prec, rec = tp / (tp + fp), tp / (tp + fn)
lo_p, hi_p = wilson(tp, tp + fp); lo_r, hi_r = wilson(tp, tp + fn)
report(f"  precision = P(spam | pred spam)  = {tp}/{tp+fp} = {prec:.4f}, 95% Wilson [{lo_p:.4f}, {hi_p:.4f}]")
report(f"  recall    = P(pred spam | spam)  = {tp}/{tp+fn} = {rec:.4f}, 95% Wilson [{lo_r:.4f}, {hi_r:.4f}]")
ci_p = stats.binomtest(tp, tp + fp).proportion_ci(method="wilson")
ci_r = stats.binomtest(tp, tp + fn).proportion_ci(method="wilson")
report(f"  Wilson check vs scipy.stats: max diff = "
       f"{max(abs(lo_p-ci_p.low), abs(hi_p-ci_p.high), abs(lo_r-ci_r.low), abs(hi_r-ci_r.high)):.2e}")

# True precision/recall of this classifier, by simulating a very large test set
Xbig, ybig = generate(200_000, theta[0], theta[1], prior, np.random.default_rng(3))
pbig = scores(Xbig, log_theta, log_prior).argmax(axis=1)
tpb, fpb, fnb, _ = confusion(ybig, pbig)
tp_true, tr_true = tpb / (tpb + fpb), tpb / (tpb + fnb)
report(f"  'true' precision/recall of this classifier (200,000 new messages): "
       f"{tp_true:.4f} / {tr_true:.4f}; inside the Wilson intervals: "
       f"{lo_p <= tp_true <= hi_p} / {lo_r <= tr_true <= hi_r}")

# Coverage of the Wilson interval for recall: repeat the 400-message test 2000 times
true_rec = tpb / (tpb + fnb)
rng_c = np.random.default_rng(4); cover = 0; R = 2000
for _ in range(R):
    Xr, yr = generate(400, theta[0], theta[1], prior, rng_c)
    t, _, f, _ = confusion(yr, scores(Xr, log_theta, log_prior).argmax(axis=1))
    lo, hi = wilson(t, t + f)
    cover += (lo <= true_rec <= hi)
report(f"  Wilson 95% coverage for recall over {R} repeated test sets: {cover/R:.3f}")

# ---------------------------------------------------------------- similarity sweep
report("\n[Extra] accuracy falls as the two word distributions become more similar")
ts = np.linspace(0, 1, 11); accs = []
rng_s = np.random.default_rng(5)
for t in ts:
    th1_t = (1 - t) * theta[1] + t * theta[0]
    Xs, ys = generate(1600 + 20000, theta[0], th1_t, prior, rng_s)   # large test set: smooth curve
    lt, lp, _ = fit(Xs[:1600], ys[:1600])
    accs.append(np.mean(scores(Xs[1600:], lt, lp).argmax(axis=1) == ys[1600:]))
report("  " + ", ".join(f"t={t:.1f}: {a:.3f}" for t, a in zip(ts, accs)))
plt.figure(figsize=(5, 3.5))
plt.plot(ts, accs, "o-", label="Naive Bayes test accuracy")
plt.axhline(prior[0], ls="--", c="gray", label="always predict ham (0.8)")
plt.xlabel(r"mixing $t$:  $\theta_{spam}=(1-t)\theta_{spam}+t\,\theta_{ham}$")
plt.ylabel("accuracy"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(FIG, "sim_accuracy_vs_similarity.png"), dpi=200); plt.close()

with open(os.path.join(FIG, "simulation_results.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("\nSaved figures and figures/simulation_results.txt")
