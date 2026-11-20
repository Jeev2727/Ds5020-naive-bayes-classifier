"""verify.py -- comparison helper used by every check script (from the Iteration 3 brief)."""
import numpy as np


def report(name, mine, ref, log=None):
    """Print max |mine - ref| and the relative error ||mine - ref|| / ||ref||; return both."""
    mine, ref = np.asarray(mine, float), np.asarray(ref, float)
    abs_err = np.abs(mine - ref).max()
    rel_err = np.linalg.norm(mine - ref) / np.linalg.norm(ref)
    line = f"{name:34s} max|diff| = {abs_err:.2e}   rel = {rel_err:.2e}"
    print(line)
    if log is not None:
        log.append(line)
    return abs_err, rel_err


def tex_number(x, digits=2):
    """Format a float for LaTeX: 3.6e-15 -> 3.6\\times10^{-15}; 0 -> 0."""
    if x == 0:
        return "0"
    if not np.isfinite(x):
        return r"\infty" if x > 0 else r"-\infty"
    if 1e-3 <= abs(x) < 1e4:
        return f"{x:.{digits + 2}g}"
    m, e = f"{x:.{digits - 1}e}".split("e")
    return rf"{m}\times10^{{{int(e)}}}"


class TexResults:
    """Collect named results and write them as LaTeX macros: \\resdef{name}{value}.
    The report prints them with \\res{name}; a missing value shows as a red placeholder."""

    def __init__(self):
        self.items = {}

    def __setitem__(self, key, value):
        assert key.isalpha(), "macro names must be letters only"
        self.items[key] = value

    def write(self, path, header):
        with open(path, "w") as f:
            f.write(f"% {header}\n% Generated automatically -- do not edit by hand.\n")
            for k, v in self.items.items():
                f.write(f"\\resdef{{{k}}}{{{v}}}\n")


def reliability(post_spam, y, bins=10):
    """Mean predicted P(spam) and observed spam rate in equal-width bins."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(post_spam, edges) - 1, 0, bins - 1)
    pred, obs, cnt = [], [], []
    for k in range(bins):
        m = idx == k
        if m.sum() > 0:
            pred.append(post_spam[m].mean()); obs.append(y[m].mean()); cnt.append(m.sum())
    return np.array(pred), np.array(obs), np.array(cnt)
