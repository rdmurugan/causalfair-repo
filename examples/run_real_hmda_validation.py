"""
Run CausalFair AIPW analysis on REAL HMDA data (NY 2022)
Validates paper claims: TE=8.6pp, IDE~40%, IIE~60%
"""
import sys
sys.path.insert(0, '/sessions/elegant-busy-planck/mnt/causal -fairness/code')

import pandas as pd
import numpy as np
from causalfair.causal_fair_estimate import AIPWEstimator, CausalFairPipeline
from causalfair.causal_fair_sensitivity import EValueAnalysis
import json
import time

# Load real HMDA data
print("=" * 70)
print("CAUSAL MEDIATION ANALYSIS ON REAL HMDA DATA (NY 2022)")
print("=" * 70)

df = pd.read_csv('/sessions/elegant-busy-planck/mnt/causal -fairness/data/hmda_real_analysis.csv')
print(f"\nDataset: {len(df)} observations")
print(f"  White: {(df['A']==0).sum()}, Black: {(df['A']==1).sum()}")
print(f"  White denial rate: {df[df['A']==0]['Y'].mean()*100:.2f}%")
print(f"  Black denial rate: {df[df['A']==1]['Y'].mean()*100:.2f}%")
te_raw = df[df['A']==1]['Y'].mean()*100 - df[df['A']==0]['Y'].mean()*100
print(f"  Raw TE: {te_raw:.2f} pp")

# Subsample to 30K for computational feasibility with cross-fitting
np.random.seed(2026)
n_sample = 30000
if len(df) > n_sample:
    # Stratified sample preserving race ratio
    black = df[df['A'] == 1]
    white = df[df['A'] == 0]
    n_black = min(len(black), int(n_sample * len(black) / len(df)))
    n_white = n_sample - n_black
    df_sample = pd.concat([
        black.sample(n=n_black, random_state=2026),
        white.sample(n=n_white, random_state=2026)
    ]).reset_index(drop=True)
    print(f"\nSubsampled to {len(df_sample)} (White={n_white}, Black={n_black})")
else:
    df_sample = df.copy()

# Verify denial rates preserved
print(f"  Sample White denial: {df_sample[df_sample['A']==0]['Y'].mean()*100:.2f}%")
print(f"  Sample Black denial: {df_sample[df_sample['A']==1]['Y'].mean()*100:.2f}%")

# Prepare arrays
A = df_sample['A'].values
Y = df_sample['Y'].values
mediators = ['dti_numeric', 'ltv', 'income_quintile', 'credit_score_quintile']
covariates = ['tract_minority_pct', 'tract_income_pct']
M = df_sample[mediators].values
W = df_sample[covariates].values

print(f"\nMediator columns: {mediators}")
print(f"Covariate columns: {covariates}")

# Run AIPW estimator
print("\n" + "-" * 70)
print("Running AIPW Estimator (5-fold cross-fitting, LogisticRegression)...")
print("-" * 70)

start = time.time()
estimator = AIPWEstimator(n_folds=5, n_estimators=100, max_depth=3)
estimator.fit(df_sample, treatment='A', outcome='Y',
              mediators=mediators, covariates=covariates)
elapsed = time.time() - start
print(f"Completed in {elapsed:.1f} seconds")

# Extract results
ide_pp = estimator.ide_ * 100
iie_pp = estimator.iie_ * 100
te_pp = estimator.te_ * 100
ide_ci = (estimator.ide_ci_[0]*100, estimator.ide_ci_[1]*100) if estimator.ide_ci_ is not None else (np.nan, np.nan)
iie_ci = (estimator.iie_ci_[0]*100, estimator.iie_ci_[1]*100) if estimator.iie_ci_ is not None else (np.nan, np.nan)
ide_pct = ide_pp / te_pp * 100 if te_pp != 0 else 0
iie_pct = iie_pp / te_pp * 100 if te_pp != 0 else 0

print(f"\n{'='*50}")
print(f"REAL HMDA DATA RESULTS")
print(f"{'='*50}")
print(f"Total Effect (TE):  {te_pp:.2f} pp")
print(f"IDE (Direct):       {ide_pp:.2f} pp  [{ide_ci[0]:.2f}, {ide_ci[1]:.2f}]")
print(f"IIE (Indirect):     {iie_pp:.2f} pp  [{iie_ci[0]:.2f}, {iie_ci[1]:.2f}]")
print(f"IDE % of TE:        {ide_pct:.1f}%")
print(f"IIE % of TE:        {iie_pct:.1f}%")

# Path-specific effects
print(f"\n{'='*50}")
print(f"PATH-SPECIFIC INDIRECT EFFECTS")
print(f"{'='*50}")
path_results = estimator.fit_path_specific(df_sample, treatment='A', outcome='Y',
                                            mediators=mediators, covariates=covariates)
for name, res in path_results.items():
    effect_pp = res['estimate'] * 100
    pval = res['p_value']
    print(f"  {name:30s}: {effect_pp:.2f} pp (p={pval:.2e})")

# E-value sensitivity analysis
print(f"\n{'='*50}")
print(f"E-VALUE SENSITIVITY ANALYSIS")
print(f"{'='*50}")
evalue = EValueAnalysis()
white_denial = df_sample[df_sample['A']==0]['Y'].mean()
black_denial = df_sample[df_sample['A']==1]['Y'].mean()
ide_rr = (white_denial + ide_pp/100) / white_denial
evalue.compute(ide_estimate=ide_pp/100, ide_ci_lower=ide_ci[0]/100,
               baseline_risk=white_denial)
e_results = {'e_value_point': evalue.e_value_, 'e_value_ci': evalue.e_value_ci_}
print(f"IDE Risk Ratio:     {ide_rr:.3f}")
print(f"E-value (point):    {e_results['e_value_point']:.3f}")
print(f"E-value (CI):       {e_results['e_value_ci']:.3f}")

# Compare with paper claims
print(f"\n{'='*70}")
print(f"COMPARISON: REAL DATA vs PAPER CLAIMS")
print(f"{'='*70}")
print(f"{'Metric':<25} {'Real HMDA (NY 2022)':>20} {'Paper (Synthetic)':>20}")
print(f"{'-'*65}")
print(f"{'White denial rate':<25} {df_sample[df_sample['A']==0]['Y'].mean()*100:>19.2f}% {'7.83%':>20}")
print(f"{'Black denial rate':<25} {df_sample[df_sample['A']==1]['Y'].mean()*100:>19.2f}% {'16.45%':>20}")
print(f"{'Total Effect (pp)':<25} {te_pp:>20.2f} {'8.61':>20}")
print(f"{'IDE (pp)':<25} {ide_pp:>20.2f} {'3.41':>20}")
print(f"{'IIE (pp)':<25} {iie_pp:>20.2f} {'5.21':>20}")
print(f"{'IDE % of TE':<25} {ide_pct:>19.1f}% {'39.6%':>20}")
print(f"{'IIE % of TE':<25} {iie_pct:>19.1f}% {'60.4%':>20}")
print(f"{'E-value':<25} {e_results['e_value_point']:>20.3f} {'2.225':>20}")

# Save results
real_results = {
    'source': 'Real HMDA data, NY 2022, CFPB Data Browser',
    'n_total': len(df_sample),
    'n_white': int((df_sample['A']==0).sum()),
    'n_black': int((df_sample['A']==1).sum()),
    'white_denial_rate': float(df_sample[df_sample['A']==0]['Y'].mean()),
    'black_denial_rate': float(df_sample[df_sample['A']==1]['Y'].mean()),
    'total_effect_pp': float(te_pp),
    'ide_pp': float(ide_pp),
    'iie_pp': float(iie_pp),
    'ide_pct_of_te': float(ide_pct),
    'iie_pct_of_te': float(iie_pct),
    'ide_ci_lower': float(ide_ci[0]),
    'ide_ci_upper': float(ide_ci[1]),
    'iie_ci_lower': float(iie_ci[0]),
    'iie_ci_upper': float(iie_ci[1]),
    'e_value_point': float(e_results['e_value_point']),
    'e_value_ci': float(e_results['e_value_ci']),
    'path_effects': {name: {'effect_pp': float(res['estimate']*100), 'pvalue': float(res['p_value'])}
                     for name, res in path_results.items()}
}

outpath = '/sessions/elegant-busy-planck/mnt/causal -fairness/output/real_hmda_results.json'
with open(outpath, 'w') as f:
    json.dump(real_results, f, indent=2)
print(f"\nResults saved to {outpath}")
