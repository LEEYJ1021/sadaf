"""Unit tests for sadaf.models module (LSTM/GRU/BayesianLSTM/Mamba/ProtoNet)."""
import numpy as np
import pytest
import torch

from sadaf.models.lstm import LSTMForecaster, LSTMClassifier, BayesianLSTM
from sadaf.models.gru import GRUForecaster
from sadaf.models.mamba import MambaForecaster
from sadaf.models.protonet import ProtoNetEncoder
from sadaf.models.attention import LSTMWithAttention

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

BATCH, SEQ_LEN, D_IN = 6, 4, 7


@pytest.fixture
def toy_batch():
    return torch.randn(BATCH, SEQ_LEN, D_IN)


class TestLSTMForecaster:
    def test_output_shape(self, toy_batch):
        model = LSTMForecaster(D_IN)
        out = model(toy_batch)
        assert out.shape == (BATCH,)

    def test_bidirectional_runs(self, toy_batch):
        model = LSTMForecaster(D_IN, bidirectional=True)
        out = model(toy_batch)
        assert out.shape == (BATCH,)
        assert torch.isfinite(out).all()


class TestLSTMClassifier:
    def test_output_shape_and_logits(self, toy_batch):
        model = LSTMClassifier(D_IN)
        logits = model(toy_batch)
        assert logits.shape == (BATCH,)
        assert torch.isfinite(logits).all()


class TestBayesianLSTM:
    def test_forward_shape(self, toy_batch):
        model = BayesianLSTM(D_IN, dropout=0.4)
        out = model(toy_batch)
        assert out.shape == (BATCH,)

    def test_predict_posterior_keys_and_shapes(self, toy_batch):
        model = BayesianLSTM(D_IN, dropout=0.4)
        X_np = toy_batch.numpy()
        post = model.predict_posterior(X_np, n_samples=20, temperature=1.5)
        for key in ['mean', 'std', 'ci_lo', 'ci_hi', 'draws', 'temperature']:
            assert key in post
        assert post['mean'].shape == (BATCH,)
        assert post['ci_lo'].shape == (BATCH,)
        assert np.all(post['ci_lo'] <= post['ci_hi'])

    def test_temperature_scaling_widens_interval(self, toy_batch):
        model = BayesianLSTM(D_IN, dropout=0.4)
        X_np = toy_batch.numpy()
        post_narrow = model.predict_posterior(X_np, n_samples=30, temperature=1.0)
        post_wide   = model.predict_posterior(X_np, n_samples=30, temperature=2.0)
        width_narrow = (post_narrow['ci_hi'] - post_narrow['ci_lo']).mean()
        width_wide   = (post_wide['ci_hi'] - post_wide['ci_lo']).mean()
        assert width_wide >= width_narrow


class TestGRUForecaster:
    def test_output_shape(self, toy_batch):
        model = GRUForecaster(D_IN)
        out = model(toy_batch)
        assert out.shape == (BATCH,)


class TestMambaForecaster:
    def test_output_shape(self, toy_batch):
        model = MambaForecaster(D_IN, d_model=16, n_layers=2, d_state=4)
        out = model(toy_batch)
        assert out.shape == (BATCH,)
        assert torch.isfinite(out).all()

    def test_runs_with_varying_seq_len(self):
        model = MambaForecaster(D_IN, d_model=16, n_layers=2, d_state=4)
        for sl in [3, 4, 6]:
            x = torch.randn(2, sl, D_IN)
            out = model(x)
            assert out.shape == (2,)


class TestProtoNetEncoder:
    def test_embedding_shape(self, toy_batch):
        enc = ProtoNetEncoder(D_IN, hidden=32, proj_dim=16)
        emb = enc(toy_batch)
        assert emb.shape == (BATCH, 16)


class TestLSTMWithAttention:
    def test_forward_without_attention_weights(self, toy_batch):
        model = LSTMWithAttention(D_IN)
        pred = model(toy_batch)
        assert pred.shape == (BATCH,)

    def test_forward_with_attention_weights(self, toy_batch):
        model = LSTMWithAttention(D_IN)
        pred, weights = model(toy_batch, return_attn=True)
        assert pred.shape == (BATCH,)
        assert weights.shape == (BATCH, SEQ_LEN)
        # Attention weights should sum to ~1 across the time axis
        sums = weights.sum(dim=1)
        assert torch.allclose(sums, torch.ones(BATCH), atol=1e-4)