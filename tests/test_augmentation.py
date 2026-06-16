"""Unit tests for sadaf.augmentation module (β-VAE, Copula, MBB, FSD)."""
import numpy as np
import pytest
import torch

from sadaf.augmentation.vae import AdSequenceVAE, train_vae, vae_augment
from sadaf.augmentation.copula import copula_augment
from sadaf.augmentation.mbb import mbb_augment
from sadaf.augmentation.pipeline import augment_pipeline, compute_fsd

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


@pytest.fixture
def toy_sequences():
    N, T, D = 40, 4, 7
    X = np.random.rand(N, T, D).astype(np.float32)
    Y = np.random.randn(N).astype(np.float32)
    return X, Y


class TestVAE:
    def test_vae_forward_shapes(self, toy_sequences):
        X, _ = toy_sequences
        N, T, D = X.shape
        vae = AdSequenceVAE(T, D, latent_dim=8)
        x_recon, mu, logvar, loss = vae(torch.from_numpy(X))
        assert x_recon.shape == (N, T, D)
        assert mu.shape == (N, 8)
        assert logvar.shape == (N, 8)
        assert loss.item() >= 0

    def test_vae_augment_output_shape(self, toy_sequences):
        X, _ = toy_sequences
        _, T, D = X.shape
        vae = AdSequenceVAE(T, D, latent_dim=8)
        X_syn = vae_augment(vae, n_synthetic=15)
        assert X_syn.shape == (15, T, D)
        assert np.all(X_syn >= 0) and np.all(X_syn <= 1)

    def test_train_vae_runs_without_error(self, toy_sequences):
        X, _ = toy_sequences
        vae = train_vae(X, epochs=5, batch_size=8)
        assert isinstance(vae, AdSequenceVAE)


class TestCopula:
    def test_copula_augment_shape(self, toy_sequences):
        X, _ = toy_sequences
        N, T, D = X.shape
        X_syn = copula_augment(X, n_synthetic=20)
        assert X_syn.shape == (20, T, D)

    def test_copula_preserves_dimensionality(self, toy_sequences):
        X, _ = toy_sequences
        X_syn = copula_augment(X, n_synthetic=10, random_state=1)
        assert X_syn.shape[1:] == X.shape[1:]
        assert not np.isnan(X_syn).any()


class TestMBB:
    def test_mbb_augment_shapes(self, toy_sequences):
        X, Y = toy_sequences
        X_syn, Y_syn = mbb_augment(X, Y, block_size=3, n_synthetic=12)
        assert X_syn.shape[0] == Y_syn.shape[0]
        assert X_syn.shape[1:] == X.shape[1:]

    def test_mbb_reproducible_with_seed(self, toy_sequences):
        X, Y = toy_sequences
        X1, Y1 = mbb_augment(X, Y, n_synthetic=8, seed=99)
        X2, Y2 = mbb_augment(X, Y, n_synthetic=8, seed=99)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(Y1, Y2)


class TestFSDAndPipeline:
    def test_fsd_zero_for_identical_distributions(self, toy_sequences):
        X, _ = toy_sequences
        from sadaf.models.gru import GRUForecaster
        ref_model = GRUForecaster(input_dim=X.shape[-1])
        fsd_self = compute_fsd(X, X, ref_model)
        assert fsd_self < 1e-3

    def test_augment_pipeline_runs_end_to_end(self, toy_sequences):
        X, Y = toy_sequences
        from sadaf.models.gru import GRUForecaster
        ref_model = GRUForecaster(input_dim=X.shape[-1])
        X_aug, Y_aug = augment_pipeline(X, Y, target_n=150, ref_lstm=ref_model)
        assert len(X_aug) >= len(X)
        assert len(X_aug) == len(Y_aug)

    def test_augment_pipeline_handles_no_ref_model(self, toy_sequences):
        X, Y = toy_sequences
        X_aug, Y_aug = augment_pipeline(X, Y, target_n=120, ref_lstm=None)
        assert len(X_aug) == len(Y_aug)
        assert len(X_aug) >= len(X)