"""
check_simulation.py -- Iteration 3 simulation test on the Iteration 2 model.

  1. Library check: from-scratch functions vs scikit-learn MultinomialNB (alpha = 1).
  2. Truth check:   estimated theta and pi vs the true values, in standard errors.
  3. Error vs n:    n = 50, 150, 500, 1500, 5000, 15000 on log-log axes.
  4. Q4 and Q5:     confusion matrix, precision/recall; conditional-independence ratio.
  5. Assumptions:   the Iteration 2 assumption checks, which must pass here by construction.
  6. Stress test:   make the two word distributions more similar; precision and recall fall.

Run from the repository root:   python src/check_simulation.py
Outputs: figures/check_simulation.txt, figures/sim3_*.png, docs/sim_results.tex
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB               # check only
from sklearn.metrics import confusion_matrix                # check only
from simulate import simulate_nb
from methods import (my_nb_fit, my_nb_predict, my_posterior, my_confusion_matrix,
                     my_precision_recall, my_wilson, my_pair_ratio_matrix)
from verify import report, tex_number, TexResults, reliability

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
log, res = [], TexResults()


def say(s=""):
    print(s)
    log.append(str(s))


# ============================================================ 1. main run, library check
Xc, y, theta = simulate_nb(n_docs=2000, V=20, L=3, seed=0)
theta_true = np.vstack([theta[0], theta[1]])
prior_true = np.array([0.8, 0.2])
tr, te = slice(0, 1600), slice(1600, None)

model = my_nb_fit(Xc[tr], y[tr], alpha=1.0)
pred, joint = my_nb_predict(model, Xc[te])
post = my_posterior(joint)
lib = MultinomialNB(alpha=1.0).fit(Xc[tr], y[tr])

say("=== 1. Library check: from-scratch vs sklearn MultinomialNB(alpha=1), n = 2000 ===")
match = np.mean(pred == lib.predict(Xc[te]))
say(f"predictions matching sklearn: {match:.1%} of {len(pred)}")
e_prior, _ = report("log priors vs sklearn", model["log_prior"], lib.class_log_prior_, log)
e_lik, _ = report("log P(w|c) vs sklearn", model["log_lik"], lib.feature_log_prob_, log)
e_joint, _ = report("joint log-prob vs sklearn", joint, lib.predict_joint_log_proba(Xc[te]), log)
e_post, _ = report("posterior P(c|x) vs sklearn", post, lib.predict_proba(Xc[te]), log)
res["simMatch"] = f"{100 * match:.1f}"
res["simErrPrior"] = tex_number(e_prior); res["simErrLik"] = tex_number(e_lik)
res["simErrJoint"] = tex_number(e_joint); res["simErrPost"] = tex_number(e_post)

# ============================================================ 2. truth check
say("\n=== 2. Truth check: estimates vs the parameters we chose ===")
theta_hat = np.exp(model["log_lik"])                          # smoothed (alpha = 1)
theta_mle = np.exp(my_nb_fit(Xc[tr], y[tr], alpha=0.0)["log_lik"])   # unsmoothed MLE
N = model["N"]
se = np.sqrt(theta_true * (1 - theta_true) / N[:, None])      # SE of the MLE n_wc / N_c
z = np.abs(theta_mle - theta_true) / se
z_s = np.abs(theta_hat - theta_true) / se
e_true, _ = report("P(w|c) (alpha=1) vs truth", theta_hat, theta_true, log)
report("P(w|c) (MLE, alpha=0) vs truth", theta_mle, theta_true, log)
for c, name in [(0, "ham"), (1, "spam")]:
    say(f"  {name:4s}: N_c = {int(N[c])}, ||theta_hat - theta||_2 = "
        f"{np.linalg.norm(theta_hat[c] - theta_true[c]):.4f}, max z (MLE) = {z[c].max():.2f}")
within = np.mean(z <= 3)
say(f"  MLE: fraction of the 40 estimates within 3 standard errors: {within:.1%}")
iw = np.unravel_index(z_s.argmax(), z_s.shape)
bias = theta_hat[iw] - theta_mle[iw]
say(f"  smoothed: max z = {z_s.max():.2f} at class {iw[0]}, word {iw[1]} (true theta = {theta_true[iw]:.2e},"
    f" count {int(model['C'][iw])}); smoothing shifts it by {bias:.2e} = "
    f"(n+1)/(N+|V|) - n/N, larger than its SE {se[iw]:.2e}: bias, not sampling error")
res["simSmoothMaxZ"] = f"{z_s.max():.2f}"; res["simRareTheta"] = tex_number(theta_true[iw])
res["simRareCount"] = str(int(model["C"][iw])); res["simRareBias"] = tex_number(bias)
res["simRareSE"] = tex_number(se[iw])
res["simTrueErrHam"] = f"{np.linalg.norm(theta_hat[0] - theta_true[0]):.4f}"
res["simTrueErrSpam"] = f"{np.linalg.norm(theta_hat[1] - theta_true[1]):.4f}"
res["simMaxZHam"] = f"{z[0].max():.2f}"; res["simMaxZSpam"] = f"{z[1].max():.2f}"
res["simWithin"] = f"{100 * within:.0f}"
res["simNHam"] = str(int(N[0])); res["simNSpam"] = str(int(N[1]))
pi_hat = np.exp(model["log_prior"])
se_pi = np.sqrt(0.2 * 0.8 / 1600)
say(f"  prior: pi_hat_spam = {pi_hat[1]:.4f}, true 0.2, z = {abs(pi_hat[1] - 0.2) / se_pi:.2f}")
res["simTrueErrHam"] = f"{np.linalg.norm(theta_hat[0] - theta_true[0]):.4f}"
res["simTrueErrSpam"] = f"{np.linalg.norm(theta_hat[1] - theta_true[1]):.4f}"
res["simMaxZHam"] = f"{z[0].max():.2f}"; res["simMaxZSpam"] = f"{z[1].max():.2f}"
res["simWithin"] = f"{100 * within:.0f}"
res["simNHam"] = str(int(N[0])); res["simNSpam"] = str(int(N[1]))
res["simPiSpam"] = f"{pi_hat[1]:.4f}"; res["simPiZ"] = f"{abs(pi_hat[1] - 0.2) / se_pi:.2f}"

# ============================================================ 3. error vs n
say("\n=== 3. Error against the truth vs sample size (fit on all n messages, seed 0) ===")
sizes = [50, 150, 500, 1500, 5000, 15000]
rows = []
for n in sizes:
    Xn, yn, _ = simulate_nb(n_docs=n, V=20, L=3, seed=0)     # same theta for every n
    mn = my_nb_fit(Xn, yn, alpha=1.0)
    th = np.exp(mn["log_lik"])
    err = [np.linalg.norm(th[c] - theta_true[c]) for c in (0, 1)]
    libn = MultinomialNB(alpha=1.0).fit(Xn, yn)
    lib_err = np.abs(mn["log_lik"] - libn.feature_log_prob_).max()
    rows.append((n, err[0], err[1], lib_err))
    say(f"  n = {n:6d}: ||err||_2 ham = {err[0]:.4f}, spam = {err[1]:.4f};  vs sklearn {lib_err:.1e}")
rows = np.array(rows)
slope_h = np.polyfit(np.log(rows[:, 0]), np.log(rows[:, 1]), 1)[0]
slope_s = np.polyfit(np.log(rows[:, 0]), np.log(rows[:, 2]), 1)[0]
say(f"  fitted log-log slopes: ham {slope_h:.2f}, spam {slope_s:.2f}  (theory: -0.5)")
for n, eh, es, _ in rows:
    if n in (50, 500, 5000):
        res[f"simErrN{['Fifty', 'FiveHundred', 'FiveThousand'][[50, 500, 5000].index(int(n))]}"] = \
            f"{np.hypot(eh, es):.4f}"
res["simSlopeHam"] = f"{slope_h:.2f}"; res["simSlopeSpam"] = f"{slope_s:.2f}"
res["simLibMaxAllN"] = tex_number(rows[:, 3].max())
plt.figure(figsize=(5, 3.6))
plt.loglog(rows[:, 0], rows[:, 1], "o-", label=r"ham $\|\hat\theta_0-\theta_0\|_2$")
plt.loglog(rows[:, 0], rows[:, 2], "s-", label=r"spam $\|\hat\theta_1-\theta_1\|_2$")
ref = rows[0, 2] * np.sqrt(rows[0, 0] / rows[:, 0])
plt.loglog(rows[:, 0], ref, "--", c="gray", label=r"slope $-1/2$")
plt.xlabel("simulated messages n"); plt.ylabel("error vs truth"); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "sim3_error_vs_n.png"), dpi=200); plt.close()

# ============================================================ 4. Q4 and Q5
say("\n=== 4. Q4: confusion matrix on the 400 test messages ===")
cm = my_confusion_matrix(y[te], pred)
e_cm = np.abs(cm - confusion_matrix(y[te], pred)).max()
prec, rec = my_precision_recall(cm)
tn, fp, fn, tp = cm.ravel()
lp, hp = my_wilson(tp, tp + fp); lr, hr = my_wilson(tp, tp + fn)
say(f"  CM = [[TN, FP], [FN, TP]] = {cm.tolist()};  max diff vs sklearn = {e_cm}")
say(f"  precision = {tp}/{tp + fp} = {prec:.4f}  95% Wilson [{lp:.4f}, {hp:.4f}]")
say(f"  recall    = {tp}/{tp + fn} = {rec:.4f}  95% Wilson [{lr:.4f}, {hr:.4f}]")
res["simTN"], res["simFP"], res["simFN"], res["simTP"] = map(str, (tn, fp, fn, tp))
res["simPrec"] = f"{prec:.3f}"; res["simRec"] = f"{rec:.3f}"
res["simPrecCI"] = f"[{lp:.3f}, {hp:.3f}]"; res["simRecCI"] = f"[{lr:.3f}, {hr:.3f}]"

say("\n=== 4. Q5: conditional independence, P(w1,w2|c) / (P(w1|c) P(w2|c)) ===")
Xbig, ybig, _ = simulate_nb(n_docs=200_000, V=20, L=3, seed=0)   # large sample: small noise
for c, name in [(0, "ham"), (1, "spam")]:
    R, pairs, th = my_pair_ratio_matrix(Xbig, ybig, c)
    top = np.argsort(th)[::-1][:2]
    iu = np.triu_indices(20, 1)
    big = (np.outer(th, th)[iu] * np.sum(ybig == c) * 6) > 200     # expected pair count > 200
    med = np.nanmedian(R[iu][big])
    say(f"  {name:4s}: two most frequent words w{top[0]}, w{top[1]}: ratio = {R[top[0], top[1]]:.3f}; "
        f"median over {big.sum()} frequent pairs = {med:.3f}")
    res[f"simPairRatio{name.capitalize()}"] = f"{R[top[0], top[1]]:.3f}"
    res[f"simPairMedian{name.capitalize()}"] = f"{med:.3f}"

# ============================================================ 5. assumption checks
say("\n=== 5. Assumption checks on simulated data (all hold by construction) ===")
pr, ob, cnt = reliability(post[:, 1], y[te])
gap = np.max(np.abs(pr - ob)[cnt >= 20]) if np.any(cnt >= 20) else np.nan
say(f"  A1 calibration: max |predicted - observed| over bins with >= 20 messages = {gap:.3f}")
extreme = np.mean((post[:, 1] > 0.999) | (post[:, 1] < 0.001))
say(f"  A1 fraction of posteriors beyond 0.999 or 0.001 = {extreme:.3f}")
share_tr, share_te = y[tr].mean(), y[te].mean()
say(f"  A4 spam share: train {share_tr:.4f}, test {share_te:.4f}")
seen = model["C"].sum(axis=0) > 0
oov = Xc[te][:, ~seen].sum() / Xc[te].sum()
say(f"  A6 fraction of test tokens unseen in training = {oov:.4f}")
say(f"  A2, A3, A5, A7 hold by construction (i.i.d. positions and messages, fixed L = 3, true labels)")
res["simCalGap"] = f"{gap:.3f}"; res["simExtreme"] = f"{extreme:.3f}"
res["simShareTr"] = f"{share_tr:.4f}"; res["simShareTe"] = f"{share_te:.4f}"
res["simOOV"] = f"{oov:.4f}"
plt.figure(figsize=(4, 3.6))
plt.plot([0, 1], [0, 1], "--", c="gray")
plt.scatter(pr, ob, s=np.maximum(cnt, 5))
plt.xlabel("mean predicted P(spam)"); plt.ylabel("observed spam rate")
plt.title("simulation"); plt.tight_layout()
plt.savefig(os.path.join(FIG, "sim3_reliability.png"), dpi=200); plt.close()

# ============================================================ 6. stress test
say("\n=== 6. Stress test: theta_spam <- (1-t) theta_spam + t theta_ham ===")
ts = np.linspace(0, 1, 11)
P_R = []
for t in ts:
    Xs, ys, _ = simulate_nb(n_docs=1600 + 20000, V=20, L=3, seed=0, mix=t)
    ms = my_nb_fit(Xs[:1600], ys[:1600])
    ps, _ = my_nb_predict(ms, Xs[1600:])
    p_, r_ = my_precision_recall(my_confusion_matrix(ys[1600:], ps))
    P_R.append((p_, r_, np.mean(ps == 1)))
    say(f"  t = {t:.1f}: precision = {p_:.3f}, recall = {r_:.3f}, share predicted spam = {np.mean(ps == 1):.3f}")
P_R = np.array(P_R)
res["stressPrecZero"] = f"{P_R[0, 0]:.3f}"; res["stressRecZero"] = f"{P_R[0, 1]:.3f}"
res["stressRecHalf"] = f"{P_R[5, 1]:.3f}"; res["stressRecNine"] = f"{P_R[9, 1]:.3f}"
plt.figure(figsize=(5, 3.6))
plt.plot(ts, P_R[:, 0], "o-", label="precision")
plt.plot(ts, P_R[:, 1], "s-", label="recall")
plt.axhline(0.2, ls=":", c="gray", label="spam base rate 0.2")
plt.xlabel(r"mixing $t$ (classes identical at $t=1$)"); plt.ylabel("value"); plt.ylim(-0.02, 1.02)
plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(os.path.join(FIG, "sim3_stress_precision_recall.png"), dpi=200); plt.close()

with open(os.path.join(FIG, "check_simulation.txt"), "w") as f:
    f.write("\n".join(log) + "\n")
res.write(os.path.join(ROOT, "docs", "sim_results.tex"), "src/check_simulation.py")
print("\nSaved figures/check_simulation.txt, figures/sim3_*.png, docs/sim_results.tex")
