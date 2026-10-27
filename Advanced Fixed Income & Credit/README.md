# Advanced Fixed Income and Credit — Group Assignment

EDHEC Business School, MSc · Prof. Thibault Lair

The report is `main.tex` / `main.pdf`. It has two parts:

| Part | Topic | Code |
|---|---|---|
| **I** | Deutsche Bank — Finding Relative-Value Trades (HBS 205-059): bootstrap, DB model comparison, cubic spline, Nelson–Siegel(–Svensson), key-rate durations | `ga_project/01_…` to `05_…` |
| **II** | Python Labs — US Treasury curve reaction to Trump tariff threats (event study) | `ga_project/06_event_study.py` |

## Repository layout

```
Advanced Fixed Income & Credit/
├── main.tex                     # report source (Part I + Part II)
├── main.pdf                     # compiled report
└── ga_project/
    ├── 01_bootstrap.py          # Part I, Q1.1  discount curve bootstrap
    ├── 02_compare_db_model.py   # Part I, Q1.2  market vs. DB model signals
    ├── 03_cubic_spline.py       # Part I, Q2    natural cubic spline
    ├── 04_nelson_siegel_svensson.py  # Part I, Q3
    ├── 05_krd_hedging.py        # Part I, Q4    key-rate durations
    ├── 06_event_study.py        # Part II, Q1.1–Q2.1  event study
    ├── python_labs/
    │   └── pylab_trump_tariff_threats_assignment.ipynb  # course notebook (+ call to 06)
    ├── functions_library/module_1/fred.py   # course helper: FRED download
    ├── data/
    │   └── fred/
    │       ├── api_key.py                   # YOUR key — git-ignored, never commit
    │       └── treasury_yields_cache.csv    # FRED yields cache (public data)
    └── outputs/                 # figures and tables used in main.tex
                                 #   q1_… q4_…  Part I    ·   pl_…  Part II
```

## Setup

```bash
pip install numpy pandas scipy matplotlib requests "polars[rtcompat]"
```

> **Apple-silicon Mac with x86 Anaconda (Rosetta):** the standard `polars` wheel
> crashes the Python kernel (missing AVX instructions). Install
> `polars[rtcompat]` as above.

**FRED API key (Part II only).** Get a free key at
<https://fred.stlouisfed.org/docs/api/api_key.html> and create
`ga_project/data/fred/api_key.py` containing:

```python
FRED_API_KEY = "your_key_here"
```

The file is git-ignored (`**/api_key.py`). Without a key, `06_event_study.py`
falls back to `treasury_yields_cache.csv`, so the results are still reproducible.

## Reproduce the results

```bash
cd "Advanced Fixed Income & Credit/ga_project"
python 01_bootstrap.py            # … and so on up to 05 for Part I
python 06_event_study.py          # Part II: ~5 s, writes outputs/pl_*
```

Then compile the report from `Advanced Fixed Income & Credit/`:

```bash
pdflatex main.tex && pdflatex main.tex     # twice, for the table of contents
```

Figures are found automatically through `\graphicspath{{ga_project/outputs/}}`.

## Part II at a glance

- **Q1.1–Q1.4:** the assignment's event study (±30 trading days, abnormal
  change vs. non-event baseline), by event type, curve component
  (level/slope/curvature) and Trump mandate. Sample: 2015-01-02 to 2025-08-01.
- **Q2.1:** improved design.
  - Cleaned and signed event list.
  - Sample extended to 2025-10-31.
  - Short windows `[-1,+1]` and `[0,+5]`.
  - Mandate-specific baseline.
  - Placebo p-values (5,000 draws, fixed seed).
  - De-clustered 30-day windows.
  - FOMC / same-day-overlap robustness.
- **Main finding:** the smooth 60-day yield declines in Q1 are a rate-cycle
  artefact. What survives:
  - in 2025, escalations and especially foreign retaliation pushed long yields
    up and steepened the curve;
  - de-escalations raised front-end yields.

## Working together (git)

- `git pull` **before** you start, and `git push` when you finish.
- Commit only the files of your own part. Do not commit outputs of the other
  part regenerated on your machine (discard them in Source Control).
- LaTeX auxiliary files (`*.aux`, `*.log`, `*.toc`, …), `.DS_Store`,
  `__pycache__/` and `api_key.py` are git-ignored.
