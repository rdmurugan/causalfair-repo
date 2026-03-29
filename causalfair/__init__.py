"""
CausalFair: Causal Mediation Analysis for Fair Lending
=======================================================

A Python package implementing the causal mediation framework from:
"Decomposing Discrimination: Causal Mediation Analysis for AI-Driven
Credit Decisions" (Rajamanickam, 2026).

Modules:
    causal_fair_dag       - DAG specification and structural learning
    causal_fair_estimate  - AIPW estimator for IDE/IIE with cross-fitting
    causal_fair_sensitivity - E-value computation and sensitivity analysis
    causal_fair_report    - CFPB-compatible reporting utilities

Usage:
    from causalfair import CausalFairPipeline
    pipeline = CausalFairPipeline(data, treatment='race', outcome='denial')
    results = pipeline.fit()
    results.summary()
"""

__version__ = "0.1.0"
__author__ = "Duraimurugan Rajamanickam"

from .causal_fair_dag import CreditDAG
from .causal_fair_estimate import AIPWEstimator, CausalFairPipeline
from .causal_fair_sensitivity import EValueAnalysis
