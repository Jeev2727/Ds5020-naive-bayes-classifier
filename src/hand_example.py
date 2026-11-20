"""
hand_example.py -- Reproduce the hand-worked example in the report (Appendix B).

Training messages (vocabulary: free, win, call, meet):
    d1 spam "free win free"     d2 spam "win call"
    d3 ham  "call meet"         d4 ham  "meet call meet"
Test messages: t1 "free meet", t2 "free call".

By hand: theta_spam = (3,3,2,1)/9, theta_ham = (1,1,3,4)/9,
         P(spam | t1) = 3/7, P(spam | t2) = 2/3.
"""
import numpy as np
from methods import my_tokenize as tokenize, my_count_matrix as count_matrix, my_nb_fit, my_scores as scores, my_posterior as posterior


def fit(X, y, alpha=1.0):
    m = my_nb_fit(X, y, alpha)
    return m["log_lik"], m["log_prior"], m["C"]

vocab = {"free": 0, "win": 1, "call": 2, "meet": 3}
train = ["free win free", "win call", "call meet", "meet call meet"]
y = np.array([1, 1, 0, 0])
test = ["free meet", "free call"]

X = count_matrix([tokenize(t) for t in train], vocab)
Xt = count_matrix([tokenize(t) for t in test], vocab)
log_theta, log_prior, C = fit(X, y, alpha=1.0)

hand_theta = np.array([[1, 1, 3, 4], [3, 3, 2, 1]]) / 9      # rows: ham, spam
hand_post_spam = np.array([3 / 7, 2 / 3])

post = posterior(scores(Xt, log_theta, log_prior))
print("Count matrix C = Y^T X (rows ham, spam):\n", C)
print("theta (code):\n", np.exp(log_theta).round(6))
print("max |theta_code - theta_hand| =", np.abs(np.exp(log_theta) - hand_theta).max())
print("P(spam | t1), P(spam | t2) code:", post[:, 1].round(6), " hand: 3/7, 2/3")
print("max |post_code - post_hand| =", np.abs(post[:, 1] - hand_post_spam).max())

w = log_theta[1] - log_theta[0]
b = log_prior[1] - log_prior[0]
print("weight vector w = log theta_spam - log theta_ham:", w.round(4), " b =", b)
print("w^T x + b for t1, t2:", (Xt @ w + b).round(4), "(negative -> ham)")

# Without smoothing both classes get probability 0 for t1 -> posterior undefined.
lt0, lp0, _ = fit(X, y, alpha=0.0)
print("alpha = 0, log-scores of t1 (ham, spam):", scores(Xt[:1], lt0, lp0))
