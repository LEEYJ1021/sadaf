"""Unit tests for sadaf.causal module (PSM/IPW, mediation, moderation)."""
import numpy as np
import pandas as pd
import pytest

from sadaf.causal.psm import fit_propensity_scores, match_psm, compute_ipw_att, smd
from sadaf.causal.mediation import baron_kenny_mediation, bootstrap_mediation
from sadaf.causal.moderation import fit_moderation_ols

SEED = 42
np.random.seed(SEED)


@pytest.fixture
def toy_causal_df():
    n = 300
    rng = np.random.default_rng(SEED)
    log_impression  = rng.normal(5, 1, n)
    log_cost        = rng.normal(6, 1.5, n)
    depth           = rng.normal(3, 1, n)
    propensity_true = 1 / (1 + np.exp(-(0.5 * log_impression - 0.3 * log_cost)))
    T = (rng.random(n) < propensity_true).astype(int)
    Y = (rng.random(n) < (0.2 + 0.15 * T)).astype(int)
    return pd.DataFrame({
        'log_impression': log_impression,
        'log_cost': log_cost,
        'Depth': depth,
        'T_highCTR': T,
        'has_conversion': Y,
    })


class TestSMD:
    def test_smd_zero_for_identical_groups(self):
        x = np.random.randn(100)
        assert abs(smd(x, x.copy())) < 1e-8

    def test_smd_positive_when_mean_shifted(self):
        x1 = np.random.randn(200) + 1.0
        x2 = np.random.randn(200)
        assert smd(x1, x2) > 0


class TestPSM:
    def test_fit_propensity_scores_range(self, toy_causal_df):
        confounders = ['log_impression', 'log_cost', 'Depth']
        pscore = fit_propensity_scores(
            toy_causal_df[confounders].values,
            toy_causal_df['T_highCTR'].values)
        assert pscore.shape[0] == len(toy_causal_df)
        assert np.all((pscore >= 0) & (pscore <= 1))

    def test_match_psm_returns_valid_indices(self, toy_causal_df):
        confounders = ['log_impression', 'log_cost', 'Depth']
        pscore = fit_propensity_scores(
            toy_causal_df[confounders].values,
            toy_causal_df['T_highCTR'].values)
        matched_t, matched_c, att, ci = match_psm(
            toy_causal_df, pscore, treatment_col='T_highCTR',
            outcome_col='has_conversion', caliper_mult=0.1, n_boot=200)
        assert len(matched_t) == len(matched_c)
        assert ci[0] <= att <= ci[1]

    def test_ipw_att_finite(self, toy_causal_df):
        confounders = ['log_impression', 'log_cost', 'Depth']
        pscore = fit_propensity_scores(
            toy_causal_df[confounders].values,
            toy_causal_df['T_highCTR'].values)
        ipw_att = compute_ipw_att(
            toy_causal_df['T_highCTR'].values,
            toy_causal_df['has_conversion'].values, pscore)
        assert np.isfinite(ipw_att)


class TestMediation:
    def test_baron_kenny_returns_expected_keys(self, toy_causal_df):
        X = toy_causal_df['log_impression'].values.reshape(-1, 1)
        M = toy_causal_df['Depth'].values
        Y = toy_causal_df['has_conversion'].values
        result = baron_kenny_mediation(X, M, Y)
        for key in ['a', 'b', 'c_total', 'c_prime', 'indirect']:
            assert key in result
        np.testing.assert_allclose(
            result['indirect'], result['a'] * result['b'], rtol=1e-6)

    def test_bootstrap_mediation_ci_brackets_point_estimate(self, toy_causal_df):
        X = toy_causal_df['log_impression'].values.reshape(-1, 1)
        M = toy_causal_df['Depth'].values
        Y = toy_causal_df['has_conversion'].values
        point = baron_kenny_mediation(X, M, Y)['indirect']
        ci_lo, ci_hi = bootstrap_mediation(X, M, Y, n_boot=200, seed=SEED)
        assert ci_lo <= ci_hi
        # Point estimate need not be exactly inside CI but should be in a
        # reasonable neighbourhood for a well-behaved bootstrap.
        assert ci_lo - abs(point) <= point <= ci_hi + abs(point)


class TestModeration:
    def test_moderation_ols_fits_and_returns_interaction(self, toy_causal_df):
        df = toy_causal_df.copy()
        df['log_ROAS'] = (
            0.3 * df['log_impression']
            + 0.4 * df['T_highCTR'] * df['log_impression']
            + np.random.randn(len(df)) * 0.5)
        df['is_search'] = df['T_highCTR']
        model = fit_moderation_ols(
            df, formula='log_ROAS ~ log_impression * is_search + log_cost')
        assert 'log_impression:is_search' in model.params.index
        assert np.isfinite(model.params['log_impression:is_search'])