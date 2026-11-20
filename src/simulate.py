"""
simulate.py -- The Iteration 2 simulation model as a function.

    class   y_d ~ Categorical(prior),          prior = (0.8, 0.2)
    words   x_d | y_d = c ~ Multinomial(L, theta_c),   theta_0, theta_1 ~ Dir(1, ..., 1)

The random draws happen in exactly the same order as in Iteration 2 (theta_ham,
theta_spam, labels, counts), so seed = 0 and n_docs = 2000 reproduce the Iteration 2
data and numbers.
"""
import numpy as np


def simulate_nb(n_docs=2000, V=20, L=3, seed=0, prior=(0.8, 0.2), mix=0.0):
    """Return (Xc, y, theta): counts (n_docs x V), labels, and the TRUE word
    distributions theta = {0: ham, 1: spam}.

    mix (stress test): theta_spam <- (1 - mix) theta_spam + mix theta_ham, so mix = 1
    makes the two classes identical. mix = 0 is the Iteration 2 model."""
    rng = np.random.default_rng(seed)
    theta = {0: rng.dirichlet(np.ones(V)), 1: rng.dirichlet(np.ones(V))}
    theta[1] = (1 - mix) * theta[1] + mix * theta[0]
    y = rng.choice(2, size=n_docs, p=np.asarray(prior))
    Xc = np.array([rng.multinomial(L, theta[c]) for c in y])
    return Xc, y, theta
