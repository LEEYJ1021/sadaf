"""
PATCH — sadaf/training/trainer.py :: domain_gap_report()
==========================================================
Apply this by replacing the existing domain_gap_report() function with
the version below. Nothing else in trainer.py needs to change.

What changed and why
---------------------
1. The original function reported `final_train` at best_aug_ep (the epoch
   that minimizes AUGMENTED validation loss) but reported `gap_real` using
   the train loss at best_real_ep (the epoch that minimizes REAL validation
   loss). Those are two different epochs whenever best_aug_ep != best_real_ep
   (true for every model in the March-2025 run). This patch adds an explicit
   `train_at_real_epoch` field so gap_real is always epoch-consistent with
   the value it's computed from, and keeps the original `final_train`
   (at best_aug_ep) available separately for the gap_aug diagnostic.

2. Field names are now unambiguous: `final_train_at_aug_epoch` and
   `train_at_real_epoch` replace the previously overloaded `final_train`.

3. No change to gap_aug's definition (it was already epoch-consistent:
   both best_val_aug and its paired train loss come from best_aug_ep).

Downstream usage: any code that read report["final_train"] for the
gap_real interpretation (e.g. the Fig 6 plotting script) should be
updated to read report["train_at_real_epoch"] instead.
"""

import numpy as np
from typing import Dict


def domain_gap_report(history: Dict, model_name: str = "") -> Dict[str, float]:
    """
    Compute train-val gap statistics from a history dict produced by
    train_model().

    Returns a dict with:
      best_epoch_aug           - epoch index (1-based) of best AUGMENTED-val checkpoint
      best_val_aug              - best augmented-dist val loss
      final_train_at_aug_epoch  - train loss AT best_epoch_aug (paired with best_val_aug)
      gap_aug                   - best_val_aug - final_train_at_aug_epoch

      best_epoch_real           - epoch index (1-based) of best REAL-val checkpoint
      best_val_real              - best real val loss (nan if not recorded)
      train_at_real_epoch       - train loss AT best_epoch_real (paired with best_val_real)
      gap_real                   - best_val_real - train_at_real_epoch  (nan if not recorded)

    Interpretation (unchanged sign convention, now epoch-consistent):
      gap_real > 0  =>  val_real > train  => classic overfitting signature
      gap_real < 0  =>  train > val_real  => NOT overfitting; for models with
                        training-time stochastic regularization (e.g. MC-dropout
                        in BayesianLSTM), this is more plausibly explained by
                        train-loss inflation than by underfitting, and should be
                        discussed as such rather than folded into an "overfitting
                        risk" ranking.
    """
    tr = np.array(history["train"])
    va = np.array(history["val"])
    vr = np.array(history["val_real"]) if history["val_real"] else None

    best_aug_ep = int(np.argmin(va))

    report = {
        "model": model_name,
        "best_epoch_aug": best_aug_ep + 1,
        "best_val_aug": float(va[best_aug_ep]),
        "final_train_at_aug_epoch": float(tr[best_aug_ep]),
        "gap_aug": float(va[best_aug_ep] - tr[best_aug_ep]),
    }

    if vr is not None:
        best_real_ep = int(np.argmin(vr))
        train_at_real_epoch = float(tr[best_real_ep])
        report["best_epoch_real"] = best_real_ep + 1
        report["best_val_real"] = float(vr[best_real_ep])
        report["train_at_real_epoch"] = train_at_real_epoch
        report["gap_real"] = float(vr[best_real_ep] - train_at_real_epoch)
    else:
        report["best_epoch_real"] = None
        report["best_val_real"] = float("nan")
        report["train_at_real_epoch"] = float("nan")
        report["gap_real"] = float("nan")

    return report
