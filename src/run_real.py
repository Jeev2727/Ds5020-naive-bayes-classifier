"""
run_real.py -- Iteration 3: the from-scratch classifier on the SMS Spam Collection,
verified against scikit-learn, plus the assumption checks on real data.

Cleaning (recorded in the report): lower-case; delete apostrophes; replace all other
punctuation and symbols by spaces; tokens = runs of [a-z0-9]. No rows are removed.
Split: stratified 80/20 with seed 0 (same split as Iteration 2). Vocabulary from the
training messages only. A second run removes duplicate texts BEFORE splitting.

Run from the repository root:   python src/run_real.py
Outputs: figures/real_results.txt, figures/real_*.png, figures/real_errors.txt,
         docs/real_results.tex
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB               # check only
from sklearn.metrics import confusion_matrix                # check only
from load_data import load_sms, stratified_split
from methods import (my_tokenize, my_build_vocab, my_count_matrix, my_nb_fit,
                     my_nb_predict, my_posterior, my_linear_rule, my_confusion_matrix,
                     my_precision_recall, my_wilson, my_pair_ratio_matrix)
from verify import report, tex_number, TexResults, reliability

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
log, res = [], TexResults()


def say(s=""):
    print(s)
    log.append(str(s))


def run(texts, y, seed=0):
    """Split, build V from training only, fit, predict. Returns a dict of everything."""
    train, test = stratified_split(y, test_frac=0.2, seed=seed)
    toks = [my_tokenize(t) for t in texts]
    vocab = my_build_vocab([toks[i] for i in train])
    X_tr = my_count_matrix([toks[i] for i in train], vocab)
    X_te = my_count_matrix([toks[i] for i in test], vocab)
    model = my_nb_fit(X_tr, y[train], alpha=1.0)
    pred, S = my_nb_predict(model, X_te)
    cm = my_confusion_matrix(y[test], pred)
    n_te_tokens = sum(len(toks[i]) for i in test)
    return dict(train=train, test=test, toks=toks, vocab=vocab, X_tr=X_tr, X_te=X_te,
                model=model, pred=pred, S=S, post=my_posterior(S), cm=cm,
                oov=1 - X_te.sum() / max(n_te_tokens, 1))


def metrics(cm):
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    prec, rec = my_precision_recall(cm)
    return dict(tn=tn, fp=fp, fn=fn, tp=tp, prec=prec, rec=rec,
                prec_ci=my_wilson(tp, tp + fp), rec_ci=my_wilson(tp, tp + fn),
                acc=(tp + tn) / (tn + fp + fn + tp))


# ============================================================ data and cleaning
df = load_sms()
texts = df["text"].tolist()
y = df["y"].to_numpy()
say(f"=== Real data: {df.attrs['source']}, {len(df)} messages ({y.sum()} spam) ===")
empty = sum(len(my_tokenize(t)) == 0 for t in texts)
say(f"cleaning: lower-case, delete apostrophes, other punctuation -> space; rows removed: 0; "
    f"messages empty after cleaning: {empty}")
res["realN"] = f"{len(df):,}".replace(",", "{,}"); res["realSpam"] = str(int(y.sum()))
res["realEmpty"] = str(empty)

R = run(texts, y)
m = R["model"]
say(f"split (seed 0): train {len(R['train'])} (spam {y[R['train']].mean():.4f}), "
    f"test {len(R['test'])} (spam {y[R['test']].mean():.4f})")
say(f"training vocabulary |V| = {len(R['vocab'])};  N_ham = {int(m['N'][0])}, N_spam = {int(m['N'][1])}")
res["realTrain"] = f"{len(R['train']):,}".replace(",", "{,}")
res["realTest"] = f"{len(R['test']):,}".replace(",", "{,}")
res["realShareTr"] = f"{100 * y[R['train']].mean():.2f}"; res["realShareTe"] = f"{100 * y[R['test']].mean():.2f}"
res["realV"] = f"{len(R['vocab']):,}".replace(",", "{,}")
res["realNHam"] = f"{int(m['N'][0]):,}".replace(",", "{,}")
res["realNSpam"] = f"{int(m['N'][1]):,}".replace(",", "{,}")

# ============================================================ Q3: library comparison
say("\n=== Q3: from-scratch vs sklearn MultinomialNB(alpha=1), same count matrix ===")
lib = MultinomialNB(alpha=1.0).fit(R["X_tr"], y[R["train"]])
match = np.mean(R["pred"] == lib.predict(R["X_te"]))
say(f"predictions matching sklearn: {match:.2%} of {len(R['pred'])}")
e_prior, _ = report("log priors vs sklearn", m["log_prior"], lib.class_log_prior_, log)
e_lik, r_lik = report("log P(w|c) vs sklearn", m["log_lik"], lib.feature_log_prob_, log)
e_joint, r_joint = report("joint log-prob vs sklearn", R["S"], lib.predict_joint_log_proba(R["X_te"]), log)
e_post, _ = report("posterior P(c|x) vs sklearn", R["post"], lib.predict_proba(R["X_te"]), log)
say(f"joint log-probabilities range from {R['S'].min():.1f} to {R['S'].max():.1f}; "
    f"a difference of {e_joint:.1e} is {e_joint / np.abs(R['S']).max():.1e} of the largest value")
res["realMatch"] = f"{100 * match:.2f}"
res["realErrPrior"] = tex_number(e_prior); res["realErrLik"] = tex_number(e_lik)
res["realErrJoint"] = tex_number(e_joint); res["realErrPost"] = tex_number(e_post)
res["realRelJoint"] = tex_number(r_joint); res["realSMin"] = f"{R['S'].min():.1f}"

# ============================================================ Q4: confusion matrix
say("\n=== Q4: confusion matrix on the test set ===")
M = metrics(R["cm"])
e_cm = int(np.abs(R["cm"] - confusion_matrix(y[R["test"]], R["pred"])).max())
say(f"CM = [[TN, FP], [FN, TP]] = {R['cm'].tolist()};  max diff vs sklearn = {e_cm}")
say(f"precision = {M['tp']}/{M['tp'] + M['fp']} = {M['prec']:.4f}, 95% Wilson "
    f"[{M['prec_ci'][0]:.4f}, {M['prec_ci'][1]:.4f}]")
say(f"recall    = {M['tp']}/{M['tp'] + M['fn']} = {M['rec']:.4f}, 95% Wilson "
    f"[{M['rec_ci'][0]:.4f}, {M['rec_ci'][1]:.4f}]")
base = 1 - y[R["test"]].mean()
say(f"accuracy = {M['acc']:.4f}  (always-ham baseline {base:.4f})")
for k in ("tn", "fp", "fn", "tp"):
    res[f"real{k.upper()}"] = str(M[k])
res["realPrec"] = f"{M['prec']:.3f}"; res["realRec"] = f"{M['rec']:.3f}"
res["realPrecCI"] = f"[{M['prec_ci'][0]:.3f}, {M['prec_ci'][1]:.3f}]"
res["realRecCI"] = f"[{M['rec_ci'][0]:.3f}, {M['rec_ci'][1]:.3f}]"
res["realAcc"] = f"{M['acc']:.3f}"; res["realBase"] = f"{base:.3f}"
res["realCMDiff"] = str(e_cm)

# ============================================================ duplicates removed (A3)
say("\n=== A3: duplicates ===")
test_set = {texts[i] for i in R["test"]}
train_set = {texts[i] for i in R["train"]}
leak = sum(texts[i] in train_set for i in R["test"])
say(f"test messages whose exact text also appears in training: {leak} of {len(R['test'])}")
keep = ~df.duplicated(subset="text").to_numpy()
texts_u = [t for t, k in zip(texts, keep) if k]
y_u = y[keep]
Ru = run(texts_u, y_u)
Mu = metrics(Ru["cm"])
say(f"de-duplicated: {keep.sum()} messages ({y_u.sum()} spam), |V| = {len(Ru['vocab'])}, "
    f"test {len(Ru['test'])}")
say(f"  precision = {Mu['prec']:.4f} [{Mu['prec_ci'][0]:.4f}, {Mu['prec_ci'][1]:.4f}], "
    f"recall = {Mu['rec']:.4f} [{Mu['rec_ci'][0]:.4f}, {Mu['rec_ci'][1]:.4f}], accuracy = {Mu['acc']:.4f}")
lib_u = MultinomialNB(alpha=1.0).fit(Ru["X_tr"], y_u[Ru["train"]])
match_u = np.mean(Ru["pred"] == lib_u.predict(Ru["X_te"]))
e_joint_u, _ = report("dedup: joint log-prob vs sklearn", Ru["S"], lib_u.predict_joint_log_proba(Ru["X_te"]), log)
say(f"  dedup predictions matching sklearn: {match_u:.2%}")
res["realLeak"] = str(leak); res["realUnique"] = f"{int(keep.sum()):,}".replace(",", "{,}")
res["dedupTest"] = f"{len(Ru['test']):,}".replace(",", "{,}")
for k in ("tn", "fp", "fn", "tp"):
    res[f"dedup{k.upper()}"] = str(Mu[k])
res["dedupPrec"] = f"{Mu['prec']:.3f}"; res["dedupRec"] = f"{Mu['rec']:.3f}"
res["dedupPrecCI"] = f"[{Mu['prec_ci'][0]:.3f}, {Mu['prec_ci'][1]:.3f}]"
res["dedupRecCI"] = f"[{Mu['rec_ci'][0]:.3f}, {Mu['rec_ci'][1]:.3f}]"
res["dedupAcc"] = f"{Mu['acc']:.3f}"; res["dedupMatch"] = f"{100 * match_u:.2f}"
res["dedupErrJoint"] = tex_number(e_joint_u)

# ============================================================ linear rule: spammiest words
w, b = my_linear_rule(m["log_lik"], m["log_prior"])
inv = {j: t for t, j in R["vocab"].items()}
tot = m["C"].sum(axis=0)
ok = tot >= 10                                   # words seen at least 10 times in training
order = np.argsort(w)
top_spam = [(inv[j], w[j]) for j in order[::-1] if ok[j]][:10]
top_ham = [(inv[j], w[j]) for j in order if ok[j]][:10]
say(f"\nlinear rule: b = log(pi_1/pi_0) = {b:.4f}")
say("largest w (spam evidence): " + ", ".join(f"{t} {v:.2f}" for t, v in top_spam))
say("smallest w (ham evidence): " + ", ".join(f"{t} {v:.2f}" for t, v in top_ham))
res["realB"] = f"{b:.3f}"
res["realTopSpam"] = ", ".join(f"\\texttt{{{t}}}" for t, _ in top_spam[:6])
res["realTopHam"] = ", ".join(f"\\texttt{{{t}}}" for t, _ in top_ham[:6])
res["realTopSpamW"] = f"{top_spam[0][1]:.2f}"; res["realTopSpamWord"] = f"\\texttt{{{top_spam[0][0]}}}"
fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
for ax, lst, col, ttl in [(axes[0], top_ham[::-1], "tab:blue", "ham evidence (w < 0)"),
                          (axes[1], top_spam[::-1], "tab:red", "spam evidence (w > 0)")]:
    ax.barh([t for t, _ in lst], [v for _, v in lst], color=col); ax.set_title(ttl)
    ax.set_xlabel(r"$w = \log\theta_{w|spam} - \log\theta_{w|ham}$")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "real_top_weights.png"), dpi=200); plt.close()

# ============================================================ Q5 / A1: conditional independence
say("\n=== Q5 / A1: P(w1,w2|c) / (P(w1|c) P(w2|c)) on the training messages ===")
pairs = {1: [("free", "call"), ("claim", "prize"), ("txt", "stop")],
         0: [("i", "you"), ("going", "home"), ("i", "love")]}
pair_rows = []
for c, name in [(1, "spam"), (0, "ham")]:
    Rm, P2, th = my_pair_ratio_matrix(R["X_tr"], y[R["train"]], c)
    for a, bb in pairs[c]:
        if a in R["vocab"] and bb in R["vocab"]:
            i, j = R["vocab"][a], R["vocab"][bb]
            say(f"  {name:4s}: ({a}, {bb}): P(w1,w2|c) = {P2[i, j]:.2e}, product = {th[i] * th[j]:.2e}, "
                f"ratio = {Rm[i, j]:.2f}")
            pair_rows.append((name, a, bb, P2[i, j], th[i] * th[j], Rm[i, j]))
    iu = np.triu_indices(len(th), 1)
    Lc = R["X_tr"][y[R["train"]] == c].sum(axis=1)
    expct = np.outer(th, th)[iu] * np.sum(Lc * (Lc - 1))
    big = expct > 20                  # real data are sparse: 20 expected pairs (simulation: 200)
    med = np.nanmedian(Rm[iu][big])
    frac2 = np.mean(Rm[iu][big] > 2)
    say(f"  {name:4s}: median ratio over {big.sum()} frequent pairs = {med:.2f}; share with ratio > 2 = {frac2:.2f}")
    res[f"realPairMedian{name.capitalize()}"] = f"{med:.2f}"
    res[f"realPairShareTwo{name.capitalize()}"] = f"{frac2:.2f}"
    res[f"realPairCount{name.capitalize()}"] = str(int(big.sum()))
rows_tex = " \\\\\n".join(f"{n} & \\texttt{{{a}}}, \\texttt{{{bb}}} & {tex_number(p, 3)} & "
                          f"{tex_number(q, 3)} & {r:.2f}" for n, a, bb, p, q, r in pair_rows)
res["realPairRows"] = rows_tex

# ============================================================ A1: calibration
post_spam = R["post"][:, 1]
yt = y[R["test"]]
pr, ob, cnt = reliability(post_spam, yt)
extreme = np.mean((post_spam > 0.999) | (post_spam < 0.001))
wrong_conf = np.mean(((post_spam > 0.999) & (yt == 0)) | ((post_spam < 0.001) & (yt == 1)))
say(f"\nA1 calibration: share of test posteriors beyond 0.999 or 0.001 = {extreme:.3f} "
    f"(simulation 0.117); share confidently wrong = {wrong_conf:.4f}")
for p_, o_, n_ in zip(pr, ob, cnt):
    say(f"   bin mean predicted {p_:.3f}  observed {o_:.3f}  (n = {n_})")
res["realExtreme"] = f"{extreme:.3f}"; res["realWrongConf"] = f"{wrong_conf:.4f}"
fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
axes[0].hist(post_spam[yt == 0], bins=40, range=(0, 1), alpha=0.6, label="ham", color="tab:blue")
axes[0].hist(post_spam[yt == 1], bins=40, range=(0, 1), alpha=0.6, label="spam", color="tab:red")
axes[0].set_yscale("log"); axes[0].set_xlabel(r"$P(\mathrm{spam}\mid\mathbf{x})$")
axes[0].set_ylabel("test messages (log scale)"); axes[0].legend()
axes[1].plot([0, 1], [0, 1], "--", c="gray")
axes[1].scatter(pr, ob, s=np.maximum(cnt, 5)); axes[1].set_xlabel("mean predicted P(spam)")
axes[1].set_ylabel("observed spam rate"); axes[1].set_title("real data")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "real_posterior_calibration.png"), dpi=200); plt.close()

# ============================================================ A4, A5, A6
say(f"\nA4 spam share: train {y[R['train']].mean():.4f}, test {yt.mean():.4f}")
say(f"A6 fraction of test tokens not in the training vocabulary = {R['oov']:.4f}")
L_te = np.array([len(R["toks"][i]) for i in R["test"]])
logodds = R["X_te"] @ w + b
corr = np.corrcoef(L_te[yt == 1], logodds[yt == 1])[0, 1]
say(f"A5 spam test messages: correlation between length L and log-odds w^T x + b = {corr:.2f}")
short_spam = (yt == 1) & (L_te < 20)
rec_short = np.mean(R["pred"][short_spam] == 1) if short_spam.any() else np.nan
rec_long = np.mean(R["pred"][(yt == 1) & (L_te >= 20)] == 1)
say(f"A5 recall on spam with < 20 tokens = {rec_short:.3f} (n = {short_spam.sum()}), "
    f">= 20 tokens = {rec_long:.3f}")
res["realOOV"] = f"{R['oov']:.4f}"; res["realLenCorr"] = f"{corr:.2f}"
res["realRecShort"] = f"{rec_short:.3f}"; res["realRecLong"] = f"{rec_long:.3f}"
res["realNShort"] = str(int(short_spam.sum()))

# ============================================================ A2, A7: read the errors
err_idx = np.flatnonzero(R["pred"] != yt)
conf = np.abs(post_spam[err_idx] - 0.5)
with open(os.path.join(FIG, "real_errors.txt"), "w") as f:
    f.write("Misclassified test messages, most confident first\n"
            "true | P(spam|x) | text\n")
    for k in err_idx[np.argsort(-conf)]:
        f.write(f"{'spam' if yt[k] else 'ham '} | {post_spam[k]:.4f} | {texts[R['test'][k]]}\n")
say(f"\nA2/A7: {len(err_idx)} misclassified test messages written to figures/real_errors.txt")
res["realNErr"] = str(len(err_idx))

with open(os.path.join(FIG, "real_results.txt"), "w") as f:
    f.write("\n".join(log) + "\n")
res.write(os.path.join(ROOT, "docs", "real_results.tex"), "src/run_real.py")
print("\nSaved figures/real_results.txt, figures/real_*.png, figures/real_errors.txt, docs/real_results.tex")
