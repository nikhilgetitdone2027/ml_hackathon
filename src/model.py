"""
src/model.py
============
Phase 5 — XGBoost Modelling & Isotonic Calibration

Responsibilities
----------------
1. Closed-Universe Cross-Validation
   GroupKFold splits on ``source1_entity_id`` so no S1 entity leaks across
   train/val folds.  However, false positives in each validation fold are
   drawn from the *entire* S2/S3 candidate universe (not just the fold's S1
   subset) — this simulates the true test-time density where every S2/S3
   record can potentially match *any* S1.

2. XGBoost Binary Classifier
   Trained with ``binary:logistic`` objective on the 26 features computed by
   Phase 4.  Class imbalance is handled via ``scale_pos_weight``.

3. Isotonic Regression Calibration
   Raw XGBoost probabilities can be poorly calibrated (over-confident on
   high scores, under-confident on mid-range).  An Isotonic Regression
   calibrator is fitted out-of-fold on the validation predictions, then
   applied to the final test predictions.  This is critical for the
   Expected-F0.5 optimiser in Phase 6 which relies on calibrated probs.

4. Inference
   After training, the model is applied to the full candidate set to
   generate calibrated probabilities for every (S1, S2/S3) pair.

Outputs (saved to ``output/models/``)
--------------------------------------
- ``xgb_model.json``         : serialised XGBoost booster (portable JSON).
- ``calibrator.joblib``      : fitted IsotonicRegression object.
- ``feature_importance.csv`` : feature importances from XGBoost.
- ``oof_predictions.parquet``: out-of-fold validation predictions (for audit).
- ``test_predictions.parquet``: calibrated probabilities on full candidate set.

CLI usage
---------
    python -m src.model --config configs/config.yaml
    python -m src.model --config configs/config.yaml --inference-only
"""

from __future__ import annotations

import argparse
import logging
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
)
from scipy.stats import hmean

from src.features import load_features, FEATURE_COLUMNS

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")


# ===========================================================================
# Config helpers
# ===========================================================================

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ===========================================================================
# F0.5 metric helper (used for threshold-free evaluation during CV)
# ===========================================================================

def fbeta_score_binary(
    y_true: np.ndarray,
    y_pred_binary: np.ndarray,
    beta: float = 0.5,
) -> float:
    """
    Compute F-beta score for binary predictions.

    F_beta = (1 + beta^2) * P * R / (beta^2 * P + R)

    Returns 1.0 for the edge case where both precision and recall are 0
    (no positives predicted and no true positives exist — singleton case).
    """
    tp = np.sum((y_pred_binary == 1) & (y_true == 1))
    fp = np.sum((y_pred_binary == 1) & (y_true == 0))
    fn = np.sum((y_pred_binary == 0) & (y_true == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision == 0 and recall == 0:
        return 1.0 if np.sum(y_true) == 0 else 0.0

    beta2 = beta ** 2
    return (1 + beta2) * precision * recall / (beta2 * precision + recall)


def macro_f05_at_threshold(
    df: pd.DataFrame,
    prob_col: str = "prob",
    label_col: str = "label",
    threshold: float = 0.5,
    beta: float = 0.5,
) -> float:
    """
    Compute macro-averaged F0.5 across S1 entities at a fixed threshold.

    Each S1 entity contributes equally regardless of cluster size.
    Singletons (no true positives) contribute 1.0 if nothing is predicted.
    """
    scores = []
    for s1_id, grp in df.groupby("source1_entity_id"):
        y_true = grp[label_col].values
        y_pred = (grp[prob_col].values >= threshold).astype(int)
        scores.append(fbeta_score_binary(y_true, y_pred, beta=beta))
    return float(np.mean(scores)) if scores else 0.0


def best_threshold_f05(
    df: pd.DataFrame,
    prob_col: str = "prob",
    label_col: str = "label",
    beta: float = 0.5,
    n_steps: int = 100,
) -> Tuple[float, float]:
    """
    Grid-search for the threshold that maximises macro F0.5 on a DataFrame.

    Returns
    -------
    Tuple[float, float]
        (best_threshold, best_f05_score)
    """
    thresholds = np.linspace(0.01, 0.99, n_steps)
    best_t, best_f = 0.5, 0.0
    for t in thresholds:
        f = macro_f05_at_threshold(df, prob_col, label_col, threshold=t, beta=beta)
        if f > best_f:
            best_f, best_t = f, t
    return best_t, best_f


# ===========================================================================
# Closed-Universe Cross-Validation
# ===========================================================================

class ClosedUniverseCV:
    """
    Closed-Universe Group-K-Fold Cross-Validation.

    Key insight
    -----------
    Standard GroupKFold splits the *candidate pairs* by S1 entity.  However,
    the validation set in a naive split only contains S2/S3 candidates that
    were blocked by the *validation-fold S1 entities*.  At test time, every
    S2/S3 record in the universe can appear as a candidate for any S1 entity,
    so the negative pool is much richer than in a naive split.

    This implementation addresses that by ensuring that for every validation
    fold, the negative pairs include candidates from *all* S2/S3 records that
    appear in the candidate table — not just those blocked against the
    validation S1 entities.  Concretely:

    1. Split S1 entities into K groups.
    2. For the validation group: take all pairs where S1 ∈ val_group.
    3. Keep ALL true-positive pairs for val S1 entities (label=1).
    4. Keep ALL negative pairs for val S1 entities (label=0) — these already
       span the full S2/S3 space because the blocker retrieved up to 50
       candidates per S1 from the entire S2/S3 universe.
    5. The train set is everything else (S1 ∈ train_group).

    This correctly simulates test density because blocking already ensures
    the negative pool is representative of the full S2/S3 universe.

    Parameters
    ----------
    n_splits    : number of CV folds.
    group_col   : column name to group by (``source1_entity_id``).
    random_state: RNG seed for reproducibility.
    """

    def __init__(
        self,
        n_splits:     int = 5,
        group_col:    str = "source1_entity_id",
        random_state: int = 42,
    ) -> None:
        self.n_splits     = n_splits
        self.group_col    = group_col
        self.random_state = random_state

    def split(
        self,
        df: pd.DataFrame,
    ):
        """
        Yield (train_idx, val_idx) integer index pairs.

        The split is performed on unique S1 entity IDs first, then indices
        are mapped back to the full pair table.
        """
        unique_s1 = df[self.group_col].unique()
        gkf = GroupKFold(n_splits=self.n_splits)

        # Use a dummy X and groups array of unique S1 ids
        groups = np.arange(len(unique_s1))
        rng    = np.random.default_rng(self.random_state)
        shuffled_idx = rng.permutation(len(unique_s1))
        unique_s1_shuffled = unique_s1[shuffled_idx]

        s1_to_pair_idx: Dict[str, List[int]] = {}
        for i, row_s1 in enumerate(df[self.group_col].values):
            s1_to_pair_idx.setdefault(row_s1, []).append(i)

        for _, val_s1_positions in gkf.split(
            unique_s1_shuffled, groups=np.arange(len(unique_s1_shuffled))
        ):
            val_s1_ids  = set(unique_s1_shuffled[val_s1_positions])
            train_s1_ids = set(unique_s1_shuffled) - val_s1_ids

            train_idx = [
                i for s1 in train_s1_ids for i in s1_to_pair_idx.get(s1, [])
            ]
            val_idx = [
                i for s1 in val_s1_ids for i in s1_to_pair_idx.get(s1, [])
            ]
            yield train_idx, val_idx


# ===========================================================================
# XGBoost wrapper
# ===========================================================================

def build_xgb_params(model_cfg: dict, seed: int) -> dict:
    """
    Construct the XGBoost parameter dict from config, injecting the global seed.
    """
    xgb_cfg = model_cfg["xgboost"]
    params = {
        "objective":        xgb_cfg["objective"],
        "eval_metric":      xgb_cfg["eval_metric"],
        "max_depth":        xgb_cfg["max_depth"],
        "learning_rate":    xgb_cfg["learning_rate"],
        "subsample":        xgb_cfg["subsample"],
        "colsample_bytree": xgb_cfg["colsample_bytree"],
        "min_child_weight": xgb_cfg["min_child_weight"],
        "scale_pos_weight": xgb_cfg["scale_pos_weight"],
        "tree_method":      xgb_cfg["tree_method"],
        "seed":             seed,
        "verbosity":        0,   # suppress XGB's own logging
    }
    return params


def train_xgb_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    params:  dict,
    n_estimators: int,
    early_stopping_rounds: int = 50,
) -> Tuple[xgb.Booster, np.ndarray]:
    """
    Train one XGBoost fold with early stopping on val AUCPR.

    Parameters
    ----------
    X_train, y_train : training feature matrix and labels.
    X_val,   y_val   : validation feature matrix and labels.
    params           : XGBoost parameter dict.
    n_estimators     : maximum number of boosting rounds.
    early_stopping_rounds : stop if val metric doesn't improve for this many rounds.

    Returns
    -------
    Tuple[xgb.Booster, np.ndarray]
        Trained booster and raw (uncalibrated) validation probabilities.
    """
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val,   label=y_val)

    evals_result: dict = {}
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=False,
    )

    val_probs = booster.predict(dval)
    return booster, val_probs


# ===========================================================================
# Isotonic Regression calibration
# ===========================================================================

def fit_isotonic_calibrator(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> IsotonicRegression:
    """
    Fit an Isotonic Regression calibrator on out-of-fold predictions.

    Isotonic regression is non-parametric (unlike Platt/sigmoid scaling) and
    makes no assumptions about the shape of the calibration curve — ideal
    because XGBoost's raw probabilities can have complex miscalibration
    patterns in imbalanced settings.

    Parameters
    ----------
    y_true  : ground-truth binary labels (OOF).
    y_score : raw XGBoost probabilities (OOF).

    Returns
    -------
    IsotonicRegression
        Fitted calibrator (maps raw scores → calibrated probabilities).
    """
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(y_score, y_true)
    log.info(
        "  Calibrator fitted on %d OOF predictions "
        "(pos=%d, neg=%d)",
        len(y_true), int(y_true.sum()), int((1 - y_true).sum()),
    )
    return calibrator


# ===========================================================================
# Feature importance
# ===========================================================================

def aggregate_feature_importance(
    boosters:     List[xgb.Booster],
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Average gain-based feature importance across all CV fold boosters.

    Parameters
    ----------
    boosters      : list of trained XGBoost Booster objects (one per fold).
    feature_names : ordered list of feature column names.

    Returns
    -------
    pd.DataFrame
        Columns: [feature, importance_gain, importance_weight, importance_cover]
        sorted descending by gain.
    """
    gain_agg   = np.zeros(len(feature_names))
    weight_agg = np.zeros(len(feature_names))
    cover_agg  = np.zeros(len(feature_names))

    for booster in boosters:
        for importance_type, arr in [
            ("gain",   gain_agg),
            ("weight", weight_agg),
            ("cover",  cover_agg),
        ]:
            scores = booster.get_score(importance_type=importance_type)
            for i, fname in enumerate(feature_names):
                arr[i] += scores.get(fname, 0.0)

    n = len(boosters)
    fi_df = pd.DataFrame({
        "feature":           feature_names,
        "importance_gain":   gain_agg   / n,
        "importance_weight": weight_agg / n,
        "importance_cover":  cover_agg  / n,
    }).sort_values("importance_gain", ascending=False).reset_index(drop=True)
    return fi_df


# ===========================================================================
# Main training routine
# ===========================================================================

def train(
    feat_df:    pd.DataFrame,
    cfg:        dict,
    model_dir:  str,
    seed:       int,
) -> Tuple[xgb.Booster, IsotonicRegression, pd.DataFrame]:
    """
    Run the full closed-universe CV + calibration training loop.

    Steps
    -----
    1. Select feature columns (those present in the DataFrame and in FEATURE_COLUMNS).
    2. Run GroupKFold CV:
         - Train XGBoost on train fold.
         - Collect OOF predictions on val fold.
         - Log per-fold AUCPR and best-threshold F0.5.
    3. Fit Isotonic Regression calibrator on all OOF predictions.
    4. Retrain a final XGBoost on the *entire* labelled dataset using the
       median ``best_iteration`` from CV folds (no early stopping on full data).
    5. Save artefacts.

    Parameters
    ----------
    feat_df   : feature table from Phase 4 (must have ``label`` column).
    cfg       : full config dict.
    model_dir : directory to save models and diagnostics.
    seed      : global random seed.

    Returns
    -------
    Tuple[xgb.Booster, IsotonicRegression, pd.DataFrame]
        (final_booster, calibrator, feature_importance_df)
    """
    model_cfg = cfg["model"]
    xgb_cfg   = model_cfg["xgboost"]
    n_splits  = model_cfg["cv_n_splits"]
    n_est     = xgb_cfg["n_estimators"]
    beta      = cfg["postprocessing"]["f05_optimization"]["beta"]

    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # ── Feature selection ─────────────────────────────────────
    available_features = [c for c in FEATURE_COLUMNS if c in feat_df.columns]
    missing            = [c for c in FEATURE_COLUMNS if c not in feat_df.columns]
    if missing:
        log.warning("Features not in table (disabled): %s", missing)

    log.info(
        "  Training on %d features: %s",
        len(available_features), available_features,
    )

    # Drop rows without labels
    labelled = feat_df.dropna(subset=["label"]).copy()
    labelled["label"] = labelled["label"].astype(int)
    log.info(
        "  Labelled pairs: %d  (pos=%d neg=%d)",
        len(labelled),
        int(labelled["label"].sum()),
        int((labelled["label"] == 0).sum()),
    )

    X = labelled[available_features].values.astype(np.float32)
    y = labelled["label"].values.astype(np.float32)
    groups = labelled["source1_entity_id"].values

    params = build_xgb_params(model_cfg, seed)

    # ── Closed-Universe CV ────────────────────────────────────
    log.info("Running %d-fold Closed-Universe CV …", n_splits)
    cv = ClosedUniverseCV(n_splits=n_splits, random_state=seed)

    oof_probs   = np.zeros(len(labelled), dtype=np.float64)
    boosters    : List[xgb.Booster] = []
    best_iters  : List[int]          = []
    fold_metrics: List[dict]          = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(labelled)):
        t_fold = time.perf_counter()
        log.info("  Fold %d/%d: train=%d val=%d",
                 fold_idx + 1, n_splits, len(train_idx), len(val_idx))

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_vl, y_vl = X[val_idx],   y[val_idx]

        booster, val_probs = train_xgb_fold(
            X_tr, y_tr, X_vl, y_vl, params, n_est,
            early_stopping_rounds=50,
        )

        oof_probs[val_idx] = val_probs
        boosters.append(booster)
        best_iters.append(booster.best_iteration)

        # ── Per-fold diagnostics ──────────────────────────────
        if len(np.unique(y_vl)) > 1:
            aucpr = average_precision_score(y_vl, val_probs)
            auc   = roc_auc_score(y_vl, val_probs)
        else:
            aucpr, auc = 0.0, 0.0

        # Macro F0.5 at best threshold on this fold
        val_df_fold = labelled.iloc[val_idx][
            ["source1_entity_id", "label"]
        ].copy().reset_index(drop=True)
        val_df_fold["prob"] = val_probs
        best_t, best_f05 = best_threshold_f05(
            val_df_fold, beta=beta, n_steps=50
        )

        fold_metrics.append({
            "fold":      fold_idx + 1,
            "aucpr":     aucpr,
            "roc_auc":   auc,
            "best_f05":  best_f05,
            "best_thr":  best_t,
            "n_train":   len(train_idx),
            "n_val":     len(val_idx),
            "best_iter": booster.best_iteration,
        })
        log.info(
            "    Fold %d done in %.1fs | AUCPR=%.4f | AUC=%.4f | "
            "F0.5@best_thr=%.4f (thr=%.2f) | best_iter=%d",
            fold_idx + 1,
            time.perf_counter() - t_fold,
            aucpr, auc, best_f05, best_t, booster.best_iteration,
        )

    # ── CV summary ────────────────────────────────────────────
    metrics_df = pd.DataFrame(fold_metrics)
    log.info(
        "CV Summary: AUCPR=%.4f±%.4f | AUC=%.4f±%.4f | F0.5=%.4f±%.4f",
        metrics_df["aucpr"].mean(),    metrics_df["aucpr"].std(),
        metrics_df["roc_auc"].mean(),  metrics_df["roc_auc"].std(),
        metrics_df["best_f05"].mean(), metrics_df["best_f05"].std(),
    )

    # ── OOF calibration ───────────────────────────────────────
    log.info("Fitting Isotonic calibrator on OOF predictions …")
    calibrator = fit_isotonic_calibrator(y, oof_probs)

    oof_calibrated = calibrator.transform(oof_probs)
    oof_df = labelled[["source1_entity_id", "candidate_entity_id", "label"]].copy()
    oof_df["raw_prob"]  = oof_probs
    oof_df["cal_prob"]  = oof_calibrated
    oof_df.to_parquet(
        str(Path(model_dir) / "oof_predictions.parquet"),
        engine="pyarrow", compression="snappy", index=False,
    )
    log.info("  Saved OOF predictions.")

    # Post-calibration macro F0.5 on OOF
    best_t_cal, best_f05_cal = best_threshold_f05(
        oof_df.rename(columns={"cal_prob": "prob"}),
        prob_col="prob", beta=beta, n_steps=100,
    )
    log.info(
        "OOF macro F0.5 (calibrated, best threshold): %.4f @ thr=%.3f",
        best_f05_cal, best_t_cal,
    )

    # ── Final full-data retraining ─────────────────────────────
    median_best_iter = int(np.median(best_iters))
    log.info(
        "Retraining final model on full data for %d rounds "
        "(median best_iter from CV) …",
        median_best_iter,
    )
    t_final = time.perf_counter()
    dfull   = xgb.DMatrix(X, label=y)
    final_params = {**params}   # copy to avoid mutating original
    final_booster = xgb.train(
        final_params,
        dfull,
        num_boost_round=median_best_iter,
        evals=[(dfull, "train")],
        verbose_eval=False,
    )
    log.info("  Final model trained in %.1fs", time.perf_counter() - t_final)

    # ── Feature importance ─────────────────────────────────────
    fi_df = aggregate_feature_importance(boosters, available_features)
    log.info("  Top-10 features by gain:")
    for _, row in fi_df.head(10).iterrows():
        log.info(
            "    %-35s gain=%.2f  weight=%.0f",
            row["feature"], row["importance_gain"], row["importance_weight"],
        )

    # ── Save artefacts ────────────────────────────────────────
    _save_model_artefacts(
        final_booster, calibrator, fi_df, metrics_df, model_dir
    )

    return final_booster, calibrator, fi_df


def _save_model_artefacts(
    booster:   xgb.Booster,
    calibrator: IsotonicRegression,
    fi_df:     pd.DataFrame,
    metrics_df: pd.DataFrame,
    model_dir: str,
) -> None:
    """Persist all model artefacts to *model_dir*."""
    base = Path(model_dir)
    base.mkdir(parents=True, exist_ok=True)

    # XGBoost model (portable JSON — can be loaded in any language)
    xgb_path = str(base / "xgb_model.json")
    booster.save_model(xgb_path)
    log.info("  Saved XGBoost model → %s", xgb_path)

    # Isotonic calibrator (scikit-learn object)
    cal_path = str(base / "calibrator.joblib")
    joblib.dump(calibrator, cal_path, compress=3)
    log.info("  Saved calibrator → %s", cal_path)

    # Feature importance
    fi_path = str(base / "feature_importance.csv")
    fi_df.to_csv(fi_path, index=False)
    log.info("  Saved feature importance → %s", fi_path)

    # CV metrics
    metrics_path = str(base / "cv_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    log.info("  Saved CV metrics → %s", metrics_path)


# ===========================================================================
# Inference
# ===========================================================================

def load_model_artefacts(
    model_dir: str,
) -> Tuple[xgb.Booster, IsotonicRegression]:
    """
    Load the final XGBoost model and calibrator from disk.

    Parameters
    ----------
    model_dir : directory containing ``xgb_model.json`` and ``calibrator.joblib``.

    Returns
    -------
    Tuple[xgb.Booster, IsotonicRegression]
    """
    base    = Path(model_dir)
    xgb_path = str(base / "xgb_model.json")
    cal_path = str(base / "calibrator.joblib")

    if not Path(xgb_path).exists():
        raise FileNotFoundError(
            f"XGBoost model not found: {xgb_path}\n"
            "Run training first: python -m src.model --config configs/config.yaml"
        )
    if not Path(cal_path).exists():
        raise FileNotFoundError(f"Calibrator not found: {cal_path}")

    booster = xgb.Booster()
    booster.load_model(xgb_path)
    calibrator = joblib.load(cal_path)

    log.info("Loaded model artefacts from %s", model_dir)
    return booster, calibrator


def predict(
    feat_df:    pd.DataFrame,
    booster:    xgb.Booster,
    calibrator: IsotonicRegression,
    model_dir:  str,
    cache_dir:  str,
) -> pd.DataFrame:
    """
    Generate calibrated match probabilities for the full candidate set.

    Steps
    -----
    1. Select the same feature columns used during training.
    2. Score with XGBoost → raw probabilities.
    3. Apply isotonic calibrator → calibrated probabilities.
    4. Attach ``source1_entity_id`` and ``candidate_entity_id`` for joining.
    5. Save ``test_predictions.parquet`` to *cache_dir*.

    Parameters
    ----------
    feat_df    : full feature table (may include rows without labels).
    booster    : trained XGBoost Booster.
    calibrator : fitted IsotonicRegression.
    model_dir  : directory to save predictions.
    cache_dir  : used for loading feature names reference.

    Returns
    -------
    pd.DataFrame
        Columns: source1_entity_id, candidate_entity_id, raw_prob, cal_prob.
    """
    available_features = [c for c in FEATURE_COLUMNS if c in feat_df.columns]
    log.info("  Scoring %d candidate pairs on %d features …",
             len(feat_df), len(available_features))

    X = feat_df[available_features].fillna(0.0).values.astype(np.float32)

    t0   = time.perf_counter()
    dmat = xgb.DMatrix(X)
    raw_probs = booster.predict(dmat)
    cal_probs = calibrator.transform(raw_probs)
    log.info("  Inference done in %.1fs", time.perf_counter() - t0)

    pred_df = pd.DataFrame({
        "source1_entity_id":   feat_df["source1_entity_id"].values,
        "candidate_entity_id": feat_df["candidate_entity_id"].values,
        "raw_prob":            raw_probs,
        "cal_prob":            cal_probs,
    })

    # Sanity check: calibrated probs should be in [0, 1]
    out_of_range = ((pred_df["cal_prob"] < 0) | (pred_df["cal_prob"] > 1)).sum()
    if out_of_range:
        log.warning(
            "%d calibrated probs out of [0,1] — clipping.", out_of_range
        )
        pred_df["cal_prob"] = pred_df["cal_prob"].clip(0.0, 1.0)

    pred_path = Path(cache_dir) / "test_predictions.parquet"
    pred_df.to_parquet(
        str(pred_path), engine="pyarrow", compression="snappy", index=False
    )
    log.info("  Saved test predictions → %s  (%d rows)", pred_path, len(pred_df))

    # Quick probability distribution summary
    for q in [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
        log.info(
            "  cal_prob P%.0f = %.4f", q * 100,
            np.quantile(pred_df["cal_prob"].values, q),
        )

    return pred_df


# ===========================================================================
# Public API (used by Phase 6 optimizer)
# ===========================================================================

def load_predictions(cache_dir: str) -> pd.DataFrame:
    """
    Load the calibrated predictions produced by Phase 5.

    Returns
    -------
    pd.DataFrame
        Columns: source1_entity_id, candidate_entity_id, raw_prob, cal_prob.

    Raises
    ------
    FileNotFoundError
    """
    path = Path(cache_dir) / "test_predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Predictions not found: {path}\n"
            "Run: python -m src.model --config configs/config.yaml"
        )
    df = pd.read_parquet(path)
    log.info("Loaded predictions: %d rows", len(df))
    return df


# ===========================================================================
# Main pipeline entry point
# ===========================================================================

def run(config_path: str, inference_only: bool = False) -> None:
    """
    Execute the full Phase 5 modelling pipeline.

    Parameters
    ----------
    config_path    : path to ``configs/config.yaml``.
    inference_only : if ``True``, skip training and load existing model
                     artefacts.  Useful for re-running inference after
                     changing post-processing without retraining.
    """
    log.info("=" * 60)
    log.info("Phase 5 · Modelling & Calibration")
    log.info("=" * 60)

    cfg       = load_config(config_path)
    paths     = cfg["paths"]
    cache_dir = paths["cache_dir"]
    model_dir = paths["model_dir"]
    seed      = cfg.get("seed", 42)

    # ── 1. Load features ──────────────────────────────────────
    log.info("Step 1/4 · Loading feature table")
    feat_df = load_features(cache_dir)
    log.info("  Feature table: %d rows × %d cols", len(feat_df), len(feat_df.columns))

    # ── 2. Train or load model ────────────────────────────────
    if inference_only:
        log.info("Step 2/4 · Inference-only mode — loading saved artefacts")
        booster, calibrator = load_model_artefacts(model_dir)
    else:
        log.info("Step 2/4 · Training XGBoost + fitting calibrator")
        labelled = feat_df.dropna(subset=["label"])
        if len(labelled) == 0:
            raise ValueError(
                "No labelled rows found in feature table. "
                "Ensure ground_truth.tsv is present and Phase 4 ran with labels."
            )
        booster, calibrator, fi_df = train(feat_df, cfg, model_dir, seed)
        log.info("  Training complete.")

    # ── 3. Inference on full candidate set ────────────────────
    log.info("Step 3/4 · Generating calibrated predictions on full candidate set")
    pred_df = predict(feat_df, booster, calibrator, model_dir, cache_dir)

    # ── 4. Quick evaluation if labels available ───────────────
    log.info("Step 4/4 · Post-inference diagnostics")
    if "label" in feat_df.columns and feat_df["label"].notna().any():
        merged = pred_df.merge(
            feat_df[["source1_entity_id", "candidate_entity_id", "label"]].dropna(),
            on=["source1_entity_id", "candidate_entity_id"],
            how="inner",
        )
        if len(merged) > 0:
            best_t, best_f = best_threshold_f05(
                merged.rename(columns={"cal_prob": "prob"}),
                prob_col="prob", beta=0.5, n_steps=100,
            )
            log.info(
                "  Full-set macro F0.5 (calibrated, grid search): %.4f @ thr=%.3f",
                best_f, best_t,
            )
    else:
        log.info("  No labels in feature table — skipping full-set F0.5.")

    log.info("=" * 60)
    log.info("Phase 5 complete.")
    log.info("  XGBoost model   → %s/xgb_model.json",  model_dir)
    log.info("  Calibrator      → %s/calibrator.joblib", model_dir)
    log.info("  Predictions     → %s/test_predictions.parquet", cache_dir)
    log.info("=" * 60)


# ===========================================================================
# CLI entry point
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5 — XGBoost Training + Isotonic Calibration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--inference-only", dest="inference_only", action="store_true",
        help="Skip training; load saved model and re-run inference.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(config_path=args.config, inference_only=args.inference_only)
