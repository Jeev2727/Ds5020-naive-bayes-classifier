# DS5020 Final Project: Naive Bayes Spam Classifier from Scratch

**Team:** Jeevan Mandala, Anand Joel Battula  
**Project Leader (Iteration 2):** Jeevan Mandala  
**Project Leader (Iteration 3):** Anand Joel Battula  
**Repository:** https://github.com/Jeev2727/Ds5020-naive-bayes-classifier  
**Course:** DS5020 Intro Linear Algebra & Probability, Fall 2026  
**Current iteration:** 3 (small worked example, from-scratch implementation, verification)

We classify SMS messages as spam or ham with a Multinomial Naive Bayes classifier that we
derive by hand and implement with NumPy. scikit-learn is used **only to check** our results.

## Repository layout

```
src/
  methods.py           from-scratch Naive Bayes: one my_ function per algorithm step,
                       with the equation it computes in a comment above it (NumPy only)
  simulate.py          simulate_nb(n_docs, V, L, seed): the Iteration 2 simulation model
  small_example.py     Iteration 3 Q1 (tiny example, step by step, exact fractions) and Q2 (underflow)
  check_simulation.py  simulation test: truth and library comparisons, error vs n, stress test
  run_real.py          real data: cleaning, fit, comparison with scikit-learn, assumption checks
  verify.py            report() comparison helper and result-file writer (checks only)
  load_data.py         Iteration 2: loads and describes the data, makes data plots, stratified split
  hand_example.py      Iteration 2: hand-worked example (report Appendix A)
data/                  raw data goes here (NOT committed, see below)
figures/               plots and printed results written by the scripts
docs/                  Overleaf reports (.tex, PDF), result files (*_results.tex), Excel plan
```

## Getting the data

The raw dataset is not stored in Git. Download one of these and place it in `data/`:

1. **UCI (preferred):** SMS Spam Collection, https://archive.ics.uci.edu/dataset/228/sms+spam+collection  
   Unzip and copy the file `SMSSpamCollection` (no extension) to `data/SMSSpamCollection`.
2. **Kaggle mirror:** https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset  
   Copy `spam.csv` to `data/spam.csv` (the loader handles its latin-1 encoding).

## How to run

Python 3.12+ with the packages in `requirements.txt` (numpy, pandas, matplotlib, scipy, and
scikit-learn for the comparisons). With pip: `pip install -r requirements.txt`; with uv: `uv sync`
and prefix each command with `uv run`.

From the repository root:

```bash
python src/small_example.py      # Q1, Q2 step by step -> figures/small_example_output.txt, docs/small_results.tex
python src/check_simulation.py   # simulation test      -> figures/sim3_*.png, figures/check_simulation.txt, docs/sim_results.tex
python src/run_real.py           # real data + sklearn   -> figures/real_*.png, figures/real_results.txt, docs/real_results.tex
```

Iteration 2 scripts (still runnable): `python src/hand_example.py`, `python src/load_data.py`.

Every number in the Iteration 3 report comes from the three `docs/*_results.tex` files written by
these scripts. Upload them with `figures/` to Overleaf and recompile; nothing is retyped by hand.
All randomness uses fixed seeds (`np.random.default_rng(0)`), so every member gets the same numbers.

## Iteration status

| Iteration | Contents | Status |
|---|---|---|
| 1 | Proposal and math plan | Submitted |
| 2 | Notation, definitions, assumptions, derivations, math questions, algorithm, data description, simulation check | Submitted |
| 3 | Small worked example, from-scratch implementation, simulation test, verification vs scikit-learn, assumption checks | In progress |
| 4 | Analysis, final report, presentation | Not started |
