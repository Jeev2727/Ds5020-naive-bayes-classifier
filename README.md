# DS5020 Final Project: Naive Bayes Spam Classifier from Scratch

**Team:** Jeevan Mandala, Anand Joel Battula  
**Project Leader (Iteration 2):** Jeevan Mandala  
**Course:** DS5020 Intro Linear Algebra & Probability, Fall 2026  
**Current iteration:** 2 (math foundation, data loading, simulation check)

We classify SMS messages as spam or ham with a Multinomial Naive Bayes classifier that we
derive by hand and implement with NumPy. Library routines (scikit-learn) are used only as a
final comparison in Iteration 3.

## Repository layout

```
src/
  nb_core.py        from-scratch Naive Bayes: tokenizer, count matrix, fit, scores,
                    log-sum-exp posterior, confusion matrix, Wilson interval
  hand_example.py   reproduces the hand-worked example (report, Appendix A)
  load_data.py      loads and describes the SMS Spam Collection, makes data plots,
                    saves the stratified 80/20 split
  simulation.py     simulation with known truth (report, Section "Simulation Check")
data/               raw data goes here (NOT committed, see below)
figures/            plots and printed results written by the scripts
docs/               Overleaf report (.tex and PDF) and Excel project plan
```

## Getting the data

The raw dataset is not stored in Git. Download one of these and place it in `data/`:

1. **UCI (preferred):** SMS Spam Collection, https://archive.ics.uci.edu/dataset/228/sms+spam+collection  
   Unzip and copy the file `SMSSpamCollection` (no extension) to `data/SMSSpamCollection`.
2. **Kaggle mirror:** https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset  
   Copy `spam.csv` to `data/spam.csv` (the loader handles its latin-1 encoding).

## How to run

Python 3.12+ with the packages in `requirements.txt` (numpy, pandas, matplotlib, scipy).

```bash
pip install -r requirements.txt
python src/hand_example.py     # code reproduces the hand-worked numbers
python src/simulation.py       # simulation check -> figures/sim_*.png, figures/simulation_results.txt
python src/load_data.py        # data description -> figures/data_*.png, figures/data_description.txt
```

All randomness uses `np.random.default_rng(seed)` with fixed seeds, so every member gets the
same numbers.

## Iteration status

| Iteration | Contents | Status |
|---|---|---|
| 1 | Proposal and math plan | Submitted |
| 2 | Notation, definitions, assumptions, derivations, math questions, algorithm, data description, simulation check | In progress |
| 3 | Full implementation on real data, verification vs scikit-learn `MultinomialNB` | Not started |
| 4 | Analysis, final report, presentation | Not started |
