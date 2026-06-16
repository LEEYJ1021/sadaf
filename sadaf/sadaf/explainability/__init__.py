"""sadaf.explainability — attribution methods sub-package."""
from .gsshap import GSSHAP
from .intgrad import integrated_gradients, batch_integrated_gradients
from .permshap import permutation_shap, batch_permutation_shap
from .agreement import (
    spearman_agreement_matrix,
    average_agreement,
    rank_consensus_matrix,
    cluster_agreement_report,
)

__all__ = [
    "GSSHAP",
    "integrated_gradients",
    "batch_integrated_gradients",
    "permutation_shap",
    "batch_permutation_shap",
    "spearman_agreement_matrix",
    "average_agreement",
    "rank_consensus_matrix",
    "cluster_agreement_report",
]
