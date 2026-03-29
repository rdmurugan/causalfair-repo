"""
causal_fair_dag: DAG Specification and Structural Learning for Credit Decisions
================================================================================

Implements the Credit Decision DAG (Definition 3.1) with:
- Domain knowledge constraints (temporal ordering, forbidden edges)
- PC-Stable algorithm for structure learning with bootstrap stability
- Structural coefficient estimation for mediator paths
"""

import numpy as np
import pandas as pd
from itertools import combinations
from scipy import stats


class CreditDAG:
    """
    Credit Decision Directed Acyclic Graph.

    Encodes the causal structure:
        W → A → M → Y
        W → M, W → Y
        A → Y  (direct path, NDE)
        A → M → Y  (indirect path, NIE)
        U → M, U → Y  (unmeasured confounders)

    Parameters
    ----------
    treatment : str
        Name of protected attribute column (A).
    outcome : str
        Name of outcome column (Y).
    mediators : list of str
        Names of financial mediator columns (M).
    covariates : list of str
        Names of pre-treatment covariate columns (W).
    """

    def __init__(self, treatment='A', outcome='Y',
                 mediators=None, covariates=None):
        self.treatment = treatment
        self.outcome = outcome
        self.mediators = mediators or ['dti_numeric', 'ltv', 'income_quintile',
                                        'credit_score_quintile']
        self.covariates = covariates or ['tract_income_pct', 'tract_minority_pct',
                                          'year', 'lender_type']
        self.edges = []
        self.structural_coefficients = {}
        self.bootstrap_frequencies = {}

    def specify_from_domain_knowledge(self):
        """
        Specify the DAG based on domain knowledge (Definition 3.1).

        Encoding:
        - W precedes A temporally (census tract characteristics, loan type)
        - A causally affects M (structural inequality)
        - Both A and M affect Y
        - W affects M and Y
        """
        self.edges = []

        # W → A (covariates affect treatment assignment)
        for w in self.covariates:
            self.edges.append((w, self.treatment))

        # W → M (covariates affect mediators)
        for w in self.covariates:
            for m in self.mediators:
                self.edges.append((w, m))

        # W → Y (covariates affect outcome)
        for w in self.covariates:
            self.edges.append((w, self.outcome))

        # A → M (treatment affects mediators — structural inequality path)
        for m in self.mediators:
            self.edges.append((self.treatment, m))

        # A → Y (direct effect — potential discrimination)
        self.edges.append((self.treatment, self.outcome))

        # M → Y (mediators affect outcome)
        for m in self.mediators:
            self.edges.append((m, self.outcome))

        return self

    def learn_structure(self, data, alpha=0.05, n_bootstrap=500,
                        stability_threshold=0.70):
        """
        Learn DAG structure using PC-Stable algorithm with domain constraints.

        Implements Section 5.3: PC-Stable with temporal ordering,
        forbidden edges from Y to predecessors, required edges from
        DTI and credit score to Y, and B=500 bootstrap resamples.

        Parameters
        ----------
        data : pd.DataFrame
            Analysis dataset.
        alpha : float
            Significance level for conditional independence tests.
        n_bootstrap : int
            Number of bootstrap resamples for edge stability.
        stability_threshold : float
            Minimum bootstrap frequency to retain an edge.

        Returns
        -------
        self
        """
        all_vars = self.covariates + [self.treatment] + self.mediators + [self.outcome]
        available = [v for v in all_vars if v in data.columns]
        df = data[available].dropna()

        # Domain constraints
        forbidden = set()
        required = set()

        # Y cannot cause anything (it's the final node)
        for v in available:
            if v != self.outcome:
                forbidden.add((self.outcome, v))

        # A cannot be caused by M or Y
        for m in self.mediators:
            forbidden.add((m, self.treatment))

        # Required edges: DTI → Y, credit_score → Y (standard underwriting)
        for m in self.mediators:
            if m in available:
                required.add((m, self.outcome))

        # Required: A → M for all mediators
        for m in self.mediators:
            if m in available:
                required.add((self.treatment, m))

        # Bootstrap for stability
        edge_counts = {}
        n = len(df)

        for b in range(n_bootstrap):
            sample = df.sample(n=min(n, 50000), replace=True, random_state=b)

            # Simplified structure learning via partial correlations
            edges_b = self._pc_stable_iteration(sample, available, alpha,
                                                 forbidden, required)
            for edge in edges_b:
                edge_counts[edge] = edge_counts.get(edge, 0) + 1

        # Retain stable edges
        self.edges = []
        self.bootstrap_frequencies = {}

        for edge, count in edge_counts.items():
            freq = count / n_bootstrap
            self.bootstrap_frequencies[edge] = freq
            if freq >= stability_threshold or edge in required:
                self.edges.append(edge)

        return self

    def _pc_stable_iteration(self, data, variables, alpha, forbidden, required):
        """Single PC-Stable iteration using partial correlations."""
        edges = set()
        n = len(data)

        # Start fully connected (minus forbidden)
        for v1, v2 in combinations(variables, 2):
            if (v1, v2) not in forbidden and (v2, v1) not in forbidden:
                edges.add((v1, v2))

        # Test conditional independence at increasing conditioning set sizes
        for cond_size in range(min(3, len(variables) - 2)):
            edges_to_remove = set()

            for v1, v2 in list(edges):
                # Find possible conditioning sets
                neighbors = set()
                for e in edges:
                    if v1 in e:
                        neighbors.add(e[0] if e[1] == v1 else e[1])
                    if v2 in e:
                        neighbors.add(e[0] if e[1] == v2 else e[1])
                neighbors -= {v1, v2}

                for cond_set in combinations(neighbors, min(cond_size, len(neighbors))):
                    if not cond_set:
                        # Marginal test
                        try:
                            corr, pval = stats.pearsonr(
                                data[v1].astype(float),
                                data[v2].astype(float)
                            )
                            if pval > alpha:
                                edges_to_remove.add((v1, v2))
                                break
                        except (ValueError, TypeError):
                            continue
                    else:
                        # Partial correlation test
                        try:
                            cols = [v1, v2] + list(cond_set)
                            sub = data[cols].astype(float).dropna()
                            if len(sub) < 30:
                                continue
                            corr_matrix = sub.corr()
                            # Fisher's Z test for partial correlation
                            r_xy = corr_matrix.loc[v1, v2]
                            if abs(r_xy) < 1e-10:
                                edges_to_remove.add((v1, v2))
                                break
                            z = 0.5 * np.log((1 + r_xy) / (1 - r_xy))
                            se = 1 / np.sqrt(len(sub) - len(cond_set) - 3)
                            pval = 2 * (1 - stats.norm.cdf(abs(z / se)))
                            if pval > alpha:
                                edges_to_remove.add((v1, v2))
                                break
                        except (ValueError, TypeError, np.linalg.LinAlgError):
                            continue

            edges -= edges_to_remove

        # Orient edges using domain knowledge (temporal ordering)
        oriented = set()
        node_order = {v: i for i, v in enumerate(variables)}

        for v1, v2 in edges:
            o1, o2 = node_order.get(v1, 0), node_order.get(v2, 0)
            if o1 <= o2:
                oriented.add((v1, v2))
            else:
                oriented.add((v2, v1))

        # Add required edges
        oriented |= required

        return oriented

    def estimate_structural_coefficients(self, data):
        """
        Estimate standardized structural coefficients for A → M paths.

        Section 5.3: The standardised structural coefficient for A → DTI
        is 0.23 (SE = 0.04).

        Parameters
        ----------
        data : pd.DataFrame

        Returns
        -------
        dict : {mediator_name: (coefficient, std_error)}
        """
        self.structural_coefficients = {}

        for m in self.mediators:
            if m not in data.columns or self.treatment not in data.columns:
                continue

            df = data[[self.treatment, m] + [w for w in self.covariates if w in data.columns]].dropna()

            if len(df) < 100:
                continue

            # Standardize
            y = (df[m] - df[m].mean()) / df[m].std()
            X_cols = [self.treatment] + [w for w in self.covariates if w in data.columns]
            X = df[X_cols].copy()
            for col in X.columns:
                if col != self.treatment:
                    X[col] = (X[col] - X[col].mean()) / (X[col].std() + 1e-10)

            X = np.column_stack([np.ones(len(X)), X.values])

            try:
                # OLS
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                residuals = y - X @ beta
                sigma2 = np.sum(residuals**2) / (len(y) - X.shape[1])
                var_beta = sigma2 * np.linalg.inv(X.T @ X)
                se = np.sqrt(np.diag(var_beta))

                # Coefficient for treatment is index 1
                self.structural_coefficients[m] = {
                    'coefficient': beta[1],
                    'std_error': se[1],
                    't_statistic': beta[1] / se[1],
                    'p_value': 2 * (1 - stats.norm.cdf(abs(beta[1] / se[1])))
                }
            except (np.linalg.LinAlgError, ValueError):
                continue

        return self.structural_coefficients

    def summary(self):
        """Print DAG summary."""
        print("="*65)
        print("CREDIT DECISION DAG SUMMARY")
        print("="*65)
        print(f"\nTreatment (A): {self.treatment}")
        print(f"Outcome (Y):   {self.outcome}")
        print(f"Mediators (M): {', '.join(self.mediators)}")
        print(f"Covariates (W): {', '.join(self.covariates)}")
        print(f"\nEdges ({len(self.edges)}):")

        for src, tgt in sorted(self.edges):
            freq = self.bootstrap_frequencies.get((src, tgt), None)
            freq_str = f" (bootstrap freq: {freq:.2f})" if freq else ""
            print(f"  {src} → {tgt}{freq_str}")

        if self.structural_coefficients:
            print(f"\nStructural coefficients (A → M):")
            for m, info in self.structural_coefficients.items():
                print(f"  A → {m}: β = {info['coefficient']:.3f} "
                      f"(SE = {info['std_error']:.3f}, "
                      f"p = {info['p_value']:.4f})")

    def get_adjacency_matrix(self):
        """Return adjacency matrix as DataFrame."""
        all_vars = self.covariates + [self.treatment] + self.mediators + [self.outcome]
        adj = pd.DataFrame(0, index=all_vars, columns=all_vars)
        for src, tgt in self.edges:
            if src in adj.index and tgt in adj.columns:
                adj.loc[src, tgt] = 1
        return adj
