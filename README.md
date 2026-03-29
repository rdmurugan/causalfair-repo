# CausalFair

**Causal Mediation Analysis for Algorithmic Fairness in Credit Decisions**

CausalFair is a Python package that decomposes racial disparities in credit decisions into direct discrimination (disparate treatment) and structural inequality (disparate impact) using causal mediation analysis.

## Overview

Traditional fairness metrics (statistical parity, equalized odds, SHAP) detect that a disparity exists but cannot distinguish *why*. CausalFair separates the denial gap into:

- **IDE (Interventional Direct Effect)**: The portion of the gap caused by direct discrimination &mdash; race affecting the decision even after equalizing financial features. Maps to ECOA *disparate treatment*.
- **IIE (Interventional Indirect Effect)**: The portion flowing through financial mediators (DTI, credit score, income, LTV) shaped by historical structural inequality. Maps to ECOA *disparate impact*.

This decomposition uses interventional effects (IDE/IIE) rather than natural effects (NDE/NIE), which are not identified under treatment-induced confounding &mdash; the standard setting in credit decisions where unmeasured factors affect both financial mediators and outcomes.

## Key Features

- **`causal_fair_dag`** &mdash; DAG specification with domain constraints from ECOA regulatory framework
- **`causal_fair_estimate`** &mdash; Doubly-robust AIPW estimator with K-fold cross-fitting for IDE/IIE decomposition and path-specific indirect effects
- **`causal_fair_sensitivity`** &mdash; E-value sensitivity analysis for robustness to unmeasured confounding

## Installation

```bash
pip install causalfair
```

Or install from source:

```bash
git clone https://github.com/rdmurugan/causalfair.git
cd causalfair
pip install -e .
```

## Quick Start

```python
import pandas as pd
from causalfair import AIPWEstimator

# Load your data (e.g., HMDA mortgage applications)
df = pd.read_csv("hmda_data.csv")

# Define variables
treatment = "race"        # Protected attribute (0=White, 1=Black)
outcome = "denied"        # Binary outcome
mediators = ["dti", "ltv", "income_quintile", "credit_score_quintile"]
covariates = ["tract_minority_pct", "tract_income_pct"]

# Fit the AIPW estimator
estimator = AIPWEstimator(n_folds=5, n_estimators=100, max_depth=3)
estimator.fit(df, treatment=treatment, outcome=outcome,
              mediators=mediators, covariates=covariates)

# Results
print(f"Total Effect:     {estimator.te_ * 100:.1f} pp")
print(f"Direct (IDE):     {estimator.ide_ * 100:.1f} pp ({estimator.ide_/estimator.te_*100:.1f}%)")
print(f"Indirect (IIE):   {estimator.iie_ * 100:.1f} pp ({estimator.iie_/estimator.te_*100:.1f}%)")
```

## Empirical Results (HMDA NY 2022)

Applied to 89,465 real HMDA conventional purchase mortgage applications from New York State (2022):

| Estimand | Estimate | 95% CI | % of TE |
|----------|----------|--------|---------|
| Total Effect (TE) | 7.9 pp | &mdash; | 100% |
| Direct Effect (IDE) | 1.9 pp | [0.1, 3.6] | 23.4% |
| Indirect Effect (IIE) | 6.1 pp | [4.1, 8.0] | 76.6% |

Approximately **77% of the racial denial gap** flows through financial mediators shaped by structural inequality, while 23% represents a conservative lower bound on direct discrimination. E-value = 1.68.

## Citation

```bibtex
@article{rajamanickam2026decomposing,
  title={Decomposing Discrimination: Causal Mediation Analysis for AI-Driven Credit Decisions},
  author={Rajamanickam, Duraimurugan},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Author

**Duraimurugan Rajamanickam**
- VP, Artificial Intelligence, Hudson Valley Credit Union
- PhD Candidate, University of Arkansas at Little Rock
- Author, *Causal Inference for Machine Learning Engineers* (Springer, 2024)
