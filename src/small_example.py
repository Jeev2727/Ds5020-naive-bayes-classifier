"""
small_example.py -- Iteration 3, Q1 and Q2, step by step.

Q1: 5 training messages, 5-word vocabulary. Every step is computed twice:
    (a) by hand-style exact arithmetic with fractions.Fraction, and
    (b) with the from-scratch functions in methods.py (floating point),
and the two are compared. Then the result is checked a second way.
Q2: the underflow problem with 500 probabilities of 0.01.

Run from the repository root:   python src/small_example.py
"""
import math
import os
from fractions import Fraction as F
import numpy as np
from verify import TexResults, tex_number
from methods import (my_tokenize, my_count_matrix, my_one_hot, my_class_counts,
                     my_class_totals, my_log_likelihoods, my_log_priors, my_scores,
                     my_linear_rule, my_predict, my_posterior)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out.append(s)


def frac_vec(v):
    return "(" + ", ".join(str(x) for x in v) + ")"


def section(title):
    say("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ============================================================================ Q1 data
vocab_words = ["free", "win", "call", "meet", "home"]
vocab = {w: j for j, w in enumerate(vocab_words)}
train = ["free win call", "win free free", "call win",      # spam
         "meet home", "call meet home meet"]                 # ham
y = np.array([1, 1, 1, 0, 0])
new_msg = "free call home"
alpha = 1
V = len(vocab)

section("Q1  Small worked example: 5 messages, |V| = 5, alpha = 1")
say("Vocabulary (column order):", vocab_words)
for d, (t, c) in enumerate(zip(train, y), 1):
    say(f"  d{d}: '{t}'  ->  {'spam' if c else 'ham'}")
say(f"New message: '{new_msg}'")

# Step 0: count matrix
X = my_count_matrix([my_tokenize(t) for t in train], vocab)
x_new = my_count_matrix([my_tokenize(new_msg)], vocab)
say("\nStep 0  document-term matrix X (rows d1..d5, columns", vocab_words, ")")
say(X)
say("x_new =", x_new[0])

# Step 1: C = Y^T X
Y = my_one_hot(y)
C = my_class_counts(X, y)
say("\nStep 1  one-hot labels Y (columns ham, spam):")
say(Y)
say("C = Y^T X (row 0 = ham, row 1 = spam):")
say(C)

# Step 2: N = C 1
N = my_class_totals(C)
say(f"\nStep 2  N = C 1 = {N.astype(int)}   (N_ham = {int(N[0])}, N_spam = {int(N[1])})")

# Step 3: smoothed likelihoods, exact fractions
theta_frac = [[F(int(C[c, w]) + alpha, int(N[c]) + alpha * V) for w in range(V)] for c in (0, 1)]
say(f"\nStep 3  theta_(w|c) = (n_wc + 1) / (N_c + |V|)   (Theorem 4)")
say(f"  ham : denominator {int(N[0])} + {V} = {int(N[0]) + V},  theta = {frac_vec(theta_frac[0])}")
say(f"  spam: denominator {int(N[1])} + {V} = {int(N[1]) + V},  theta = {frac_vec(theta_frac[1])}")
say(f"  row sums: ham {sum(theta_frac[0])}, spam {sum(theta_frac[1])}")

# Step 4: priors
n_c = [int(v) for v in Y.sum(axis=0)]
prior_frac = [F(n_c[0], len(y)), F(n_c[1], len(y))]
say(f"\nStep 4  priors pi_c = n_c / n = {prior_frac[0]} (ham), {prior_frac[1]} (spam)   (Theorem 2)")

# Step 5a: classify with probabilities (exact)
say("\nStep 5a  classify the new message WITH PROBABILITIES (Theorem 1)")
words_in = [w for w in my_tokenize(new_msg)]
unnorm = []
for c, name in [(0, "ham"), (1, "spam")]:
    factors = [prior_frac[c]] + [theta_frac[c][vocab[w]] for w in words_in]
    prod = math.prod(factors)
    unnorm.append(prod)
    say(f"  {name:4s}: " + " * ".join(str(f) for f in factors) + f" = {prod}")
Z = sum(unnorm)
post_frac = [u / Z for u in unnorm]
say(f"  normalizer P(x) = {Z}")
say(f"  P(ham | x) = {post_frac[0]} = {float(post_frac[0]):.6f}")
say(f"  P(spam | x) = {post_frac[1]} = {float(post_frac[1]):.6f}")
say(f"  prediction: {'spam' if post_frac[1] > post_frac[0] else 'ham'}")

# Step 5b: classify with logs (code)
log_theta = my_log_likelihoods(C, N, alpha)
log_prior = my_log_priors(y)
S = my_scores(x_new, log_theta, log_prior)
say("\nStep 5b  classify WITH LOGS: s_c = log pi_c + sum_w x_w log theta_(w|c)   (Theorem 6)")
say(f"  s_ham  = {S[0, 0]:.6f}   (exact: log of {unnorm[0]} = {math.log(unnorm[0]):.6f})")
say(f"  s_spam = {S[0, 1]:.6f}   (exact: log of {unnorm[1]} = {math.log(unnorm[1]):.6f})")
pred = my_predict(S)[0]
say(f"  prediction (argmax of log-scores): {'spam' if pred else 'ham'}")

# Step 6: posterior by log-sum-exp
P = my_posterior(S)[0]
say(f"\nStep 6  log-sum-exp posterior (Proposition 9): P(ham|x) = {P[0]:.6f}, P(spam|x) = {P[1]:.6f}")

# Linear rule
w_vec, b = my_linear_rule(log_theta, log_prior)
odds_frac = unnorm[1] / unnorm[0]
say(f"\nLinear rule: w = log theta_spam - log theta_ham = {np.round(w_vec, 4)},  b = log(3/2) = {b:.4f}")
say(f"  w^T x + b = {float(w_vec @ x_new[0] + b):.6f} = log(spam/ham odds) = log({odds_frac}) = {math.log(odds_frac):.6f}")

# ============================================================================ checks
section("Q1  Check the result a second way")
theta_float = np.array([[float(v) for v in row] for row in theta_frac])
c1 = np.abs(np.exp(log_theta) - theta_float).max()
c2 = abs(P[1] - float(post_frac[1]))
c3 = abs(float(w_vec @ x_new[0] + b) - math.log(odds_frac))
say(f"  (1) every row of Theta sums to 1 (exact): {sum(theta_frac[0]) == 1 and sum(theta_frac[1]) == 1}")
say(f"  (2) posteriors sum to 1 (exact): {sum(post_frac) == 1};  code: {P.sum():.16f}")
say(f"  (3) probability route and log route predict the same class: "
    f"{(post_frac[1] > post_frac[0]) == bool(pred)}")
say(f"  (4) code vs exact fractions: max |theta| diff = {c1:.1e}, |P(spam|x)| diff = {c2:.1e}, "
    f"|log-odds| diff = {c3:.1e}")
say(f"  (5) Bayes by hand: P(spam|x) = pi_1 prod theta / P(x) = {unnorm[1]} / {Z} = {post_frac[1]}")

# ============================================================================ Q2
section("Q2  Underflow: multiply 500 probabilities of about 0.01")
p = np.full(500, 0.01)
running = np.cumprod(p)
first_zero = int(np.argmax(running == 0)) + 1
say(f"  np.prod(500 x 0.01)       = {float(np.prod(p))!r}     (true value 1e-1000)")
say(f"  running product after 161 factors = {float(running[160])!r}  (subnormal, tiny but not 0)")
say(f"  running product first becomes exactly 0.0 after {first_zero} factors")
say(f"  smallest positive double  = {float(np.nextafter(0, 1))!r}")
log_sum = np.sum(np.log(p))
say(f"  with logs: sum(log 0.01)  = {log_sum:.4f}  = 500 * log(0.01) = {500 * math.log(0.01):.4f}")
# Two classes whose products both underflow
p_spam = np.full(500, 0.012)
say(f"  two classes: prod(0.010) = {float(np.prod(p))!r}, prod(0.012) = {float(np.prod(p_spam))!r}  -> cannot compare")
say(f"  in logs:     {np.sum(np.log(p)):.2f} vs {np.sum(np.log(p_spam)):.2f}  -> second class wins by "
    f"{np.sum(np.log(p_spam)) - np.sum(np.log(p)):.2f} = 500 log(1.2)")

# ============================================================================ save
res = TexResults()
res["smThetaDiff"] = tex_number(c1)
res["smPostDiff"] = tex_number(c2)
res["smOddsDiff"] = tex_number(c3)
res["smPostSpam"] = f"{float(post_frac[1]):.6f}"
res["smUnderflowAt"] = str(first_zero)
res["smLogSum"] = f"{log_sum:.4f}"
os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
res.write(os.path.join(ROOT, "docs", "small_results.tex"), "src/small_example.py")
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
with open(os.path.join(ROOT, "figures", "small_example_output.txt"), "w") as f:
    f.write("\n".join(out) + "\n")
print("\nSaved figures/small_example_output.txt")
