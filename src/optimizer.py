"""
src/optimizer.py
================
Phase 6 — Post-Processing & Expected F0.5 Maximisation

Responsibilities
----------------
1. Constraint Resolution
   S1 is a deduplicated reference set, so each S2/S3 record can belong to
   **at most ONE** S1 entity.  After the model scores every candidate pair,
   some S2/S3 records may be claimed by multiple S1 entities.  This step
   resolves conflicts by keeping only the highest-probability edge for each
   S2/S3 ID and discarding the rest.

2. Expected F0.5 Maximisation  (per S1 entity, adaptive threshold)
   For each S1 entity's *post-constraint* probability list, exhaustively
   evaluate top-k subsets (k = 0, 1, …, N) and pick the k that maximises
   the Expected F0.5 metric under Bernoulli probability assumptions:

       E[TP]        = Σ  p_i          (sum over top-k predictions)
       E[FP]        = k - E[TP]
       E[pos_total] = Σ  p_i          (sum over ALL candidates for this S1)
       E[FN]        = E[pos_total] - E[TP]
       E[Precision] = E[TP] / k       (undefined at k=0, treated as 1.0)
       E[Recall]    = E[TP] / E[pos_total]  (0/0 → 1.0 for singleton)
       E[F0.5]      = 1.25 · E[P] · E[R] / (0.25·E[P] + E[R])

   Special cases:
   - k = 0   → E[F0.5] = 1.0 if E[pos_total] ≈ 0, else computed normally.
   - Singleton (E[pos_total] ≈ 0): selecting k=0 gives E[F0.5] = 1.0.

   This eliminates the need for a global decision threshold and adapts to
   each entity's individual evidence level.

3. Output
   Produces ``output/matching_results.tsv`` in strict submission format:
       source1_entity_id \\t matched_ids
   where ``matched_ids`` is a comma-separated list (empty string = no match).

   Also saves ``output/cache/post_constraint_predictions.parquet`` for audit.

CLI usage
---------
    python -m src.optimizer --config configs/config.yaml
    python -m src.optimizer --config configs/config.yaml --beta 0.5
    python -m src.optimizer --config configs/config.yaml --skip-constraint
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from src.model import load_predictions
from src.preprocessor import load_preprocessed

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ===========================================================================
# Config helpers
# ===========================================================================

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ===========================================================================
# Step 1 — Constraint Resolution
# ===========================================================================

def resolve_constraints(
    pred_df: pd.DataFrame,
    prob_col: str = "cal_prob",
) -> pd.DataFrame:
    """
    Enforce the closed-universe constraint: each S2/S3 ID → at most ONE S1.

    Algorithm
    ---------
    1. Group all predictions by ``candidate_entity_id``.
    2. For each candidate claimed by more than one S1 entity, keep only the
       row with the **highest** calibrated probability.
    3. Ties (equal probabilities) are broken by lexicographic order of
       ``source1_entity_id`` (deterministic, reproducible).
    4. All other rows for that candidate are dropped (their S1 entity's
       candidate list will be shorter after this step — those S1s will
       default to k=0 for that candidate in the F0.5 optimiser).

    Parameters
    ----------
    pred_df  : DataFrame with at least:
               [source1_entity_id, candidate_entity_id, cal_prob]
    prob_col : name of the calibrated probability column.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame where each ``candidate_entity_id`` appears at
        most once as a predicted match (for exactly one S1 entity).

    Notes
    -----
    - Rows that are NOT the winner for a contested candidate are dropped from
      the output entirely.  The original DataFrame is not modified.
    - Single-owner candidates (no conflict) pass through unchanged.
    - This function operates on the *full* prediction table, including pairs
      with probabilities below any threshold.  The F0.5 optimiser in Step 2
      decides which surviving pairs to actually predict.

    Examples
    --------
    Suppose S2-ID-42 has two predictions:
        S1-001  p=0.85
        S1-007  p=0.72
    After constraint resolution, only the S1-001 row is kept.
    S1-007's candidate list no longer contains S2-ID-42.
    """
    log.info("Step 1/3 · Constraint Resolution")
    log.info("  Input: %d candidate pairs across %d unique candidates",
             len(pred_df), pred_df["candidate_entity_id"].nunique())
    t0 = time.perf_counter()

    # Sort so that within each candidate group the highest-prob row is first,
    # with ties broken by source1_entity_id (lexicographic).
    pred_sorted = pred_df.sort_values(
        by=[prob_col, "source1_entity_id"],
        ascending=[False, True],   # descending prob, ascending S1 id (tie-break)
    )

    # Keep only the first (= highest-prob) row per candidate
    resolved = pred_sorted.drop_duplicates(
        subset="candidate_entity_id", keep="first"
    ).copy()

    n_conflicts = len(pred_df) - len(resolved)
    n_contested = (
        pred_df.groupby("candidate_entity_id")["source1_entity_id"]
        .nunique()
        .gt(1)
        .sum()
    )

    log.info(
        "  Contested candidates: %d  |  Edges removed: %d  |  "
        "Remaining pairs: %d  (%.1fs)",
        n_contested, n_conflicts, len(resolved),
        time.perf_counter() - t0,
    )
    return resolved.reset_index(drop=True)


# ===========================================================================
# Step 2 — Expected F0.5 Maximisation (per S1 entity)
# ===========================================================================

def expected_f05(
    probs_sorted: np.ndarray,
    k:            int,
    total_exp_pos: float,
    beta:          float = 0.5,
) -> float:
    """
    Compute the Expected F-beta score for a top-k selection.

    All probabilities are treated as independent Bernoulli success rates
    (calibrated → calibrated probs are valid probability estimates).

    Parameters
    ----------
    probs_sorted   : 1-D array of calibrated probabilities, sorted descending.
                     Only the top-k are selected; the rest form the "not
                     predicted" pool from which E[FN] is estimated.
    k              : number of top candidates to select (0 ≤ k ≤ len(probs)).
    total_exp_pos  : E[total positives] = sum of ALL probabilities for this S1
                     (including those beyond rank k).  Used to estimate E[FN].
    beta           : F-beta parameter (0.5 for this competition).

    Returns
    -------
    float
        Expected F-beta score in [0, 1].

    Special Cases
    -------------
    - k = 0, total_exp_pos ≈ 0   → 1.0  (singleton: no preds = perfect)
    - k = 0, total_exp_pos > 0   → 0.0  (real entity: predicting nothing is bad)
    - k > 0, E[TP] = 0           → 0.0  (selecting zero-probability items)

    Examples
    --------
    >>> probs = np.array([0.95, 0.80, 0.10, 0.05])
    >>> total_exp = probs.sum()   # 1.90
    >>> expected_f05(probs, k=2, total_exp_pos=total_exp, beta=0.5)
    # E[TP]=1.75, E[FP]=0.25, E[FN]=0.15, P=0.875, R=0.921 → F0.5≈0.883
    """
    SINGLETON_EPS = 1e-6   # below this, entity is treated as a singleton

    if k == 0:
        # Predicting nothing
        if total_exp_pos < SINGLETON_EPS:
            return 1.0   # true singleton: perfect score
        # Recall = 0 when predicting nothing and there are real positives
        return 0.0

    top_k_probs = probs_sorted[:k]
    e_tp = float(np.sum(top_k_probs))

    if e_tp < SINGLETON_EPS:
        return 0.0   # selected k items but all have ~0 probability

    e_precision = e_tp / k
    e_recall    = e_tp / total_exp_pos if total_exp_pos > SINGLETON_EPS else 1.0

    if e_precision == 0.0 and e_recall == 0.0:
        return 0.0

    beta2 = beta ** 2
    return (
        (1.0 + beta2) * e_precision * e_recall
        / (beta2 * e_precision + e_recall)
    )


def optimise_per_entity(
    entity_probs: np.ndarray,
    beta:         float = 0.5,
    grid_steps:   int   = 200,
) -> Tuple[int, float]:
    """
    Find the top-k that maximises Expected F0.5 for a single S1 entity.

    Exhaustively evaluates k = 0, 1, …, len(entity_probs).  Runtime is
    O(N) with a precomputed cumulative sum, so even N=50 is negligible.

    Parameters
    ----------
    entity_probs : 1-D array of calibrated probabilities for THIS S1 entity's
                   candidates (order does not matter; will be sorted internally).
    beta         : F-beta parameter.
    grid_steps   : unused (kept for API compatibility with config field).
                   The exhaustive scan is always used because N ≤ top_n (≤50).

    Returns
    -------
    Tuple[int, float]
        (best_k, best_expected_f05)
        best_k = 0 means: predict no matches (singleton or all probs too low).

    Algorithm
    ---------
    1. Sort probabilities descending.
    2. Compute total_exp_pos = sum of all probabilities.
    3. For k = 0, 1, …, N: compute E[F0.5] and track the best.
    4. Return argmax k and its score.
    """
    if len(entity_probs) == 0:
        return 0, 1.0   # no candidates at all → singleton

    probs_sorted     = np.sort(entity_probs)[::-1].astype(np.float64)
    total_exp_pos    = float(np.sum(probs_sorted))
    n                = len(probs_sorted)

    best_k     = 0
    best_score = expected_f05(probs_sorted, 0, total_exp_pos, beta)

    for k in range(1, n + 1):
        score = expected_f05(probs_sorted, k, total_exp_pos, beta)
        if score > best_score:
            best_score = score
            best_k     = k

    return best_k, best_score


def optimise_all_entities(
    resolved_df:  pd.DataFrame,
    all_s1_ids:   List[str],
    beta:         float = 0.5,
    prob_col:     str   = "cal_prob",
    grid_steps:   int   = 200,
) -> pd.DataFrame:
    """
    Apply ``optimise_per_entity`` to every S1 entity.

    Parameters
    ----------
    resolved_df : post-constraint prediction DataFrame with columns:
                  [source1_entity_id, candidate_entity_id, cal_prob]
    all_s1_ids  : complete list of S1 entity IDs (ensures singletons are
                  represented even if the blocker retrieved 0 candidates).
    beta        : F-beta parameter (0.5 for this competition).
    prob_col    : column name for calibrated probabilities.
    grid_steps  : passed through to ``optimise_per_entity`` (unused internally).

    Returns
    -------
    pd.DataFrame
        Columns: source1_entity_id, matched_ids (list of str), best_k, best_ef05
        One row per S1 entity.
    """
    log.info("Step 2/3 · Expected F0.5 Maximisation across %d S1 entities",
             len(all_s1_ids))
    t0 = time.perf_counter()

    # Build a dict: s1_id → [(cand_id, prob), ...] sorted by prob desc
    entity_map: Dict[str, pd.DataFrame] = {
        s1_id: grp.sort_values(prob_col, ascending=False)
        for s1_id, grp in resolved_df.groupby("source1_entity_id")
    }

    records = []
    k_counts: Dict[int, int] = {}

    for s1_id in all_s1_ids:
        if s1_id not in entity_map:
            # No candidates survived constraint resolution → singleton
            records.append({
                "source1_entity_id": s1_id,
                "matched_ids":       [],
                "best_k":            0,
                "best_ef05":         1.0,
            })
            k_counts[0] = k_counts.get(0, 0) + 1
            continue

        grp       = entity_map[s1_id]
        probs     = grp[prob_col].values
        cand_ids  = grp["candidate_entity_id"].values

        # Sort together by prob descending (grp is already sorted above)
        order     = np.argsort(-probs)
        probs     = probs[order]
        cand_ids  = cand_ids[order]

        best_k, best_ef05 = optimise_per_entity(probs, beta=beta, grid_steps=grid_steps)

        matched = list(cand_ids[:best_k].astype(str))

        records.append({
            "source1_entity_id": s1_id,
            "matched_ids":       matched,
            "best_k":            best_k,
            "best_ef05":         best_ef05,
        })
        k_counts[best_k] = k_counts.get(best_k, 0) + 1

    result_df = pd.DataFrame(records)

    # Diagnostics
    total_matched = result_df["matched_ids"].apply(len).sum()
    singletons    = (result_df["best_k"] == 0).sum()
    mean_ef05     = result_df["best_ef05"].mean()

    log.info(
        "  Optimisation done in %.1fs  |  "
        "Singletons=%d  TotalMatched=%d  MeanExpF0.5=%.4f",
        time.perf_counter() - t0, singletons, total_matched, mean_ef05,
    )
    log.info("  k distribution: %s",
             dict(sorted(k_counts.items())))

    return result_df


# ===========================================================================
# Step 3 — Output Writing
# ===========================================================================

def write_matching_results(
    result_df:    pd.DataFrame,
    output_path:  str,
) -> None:
    """
    Write ``matching_results.tsv`` in strict submission format.

    Format (tab-separated, UTF-8, with header row):
        source1_entity_id \\t matched_ids

    Rules enforced here
    -------------------
    - ``matched_ids`` is a comma-separated list of S2/S3 entity IDs.
    - Entities with no matches (singletons / k=0) get an **empty string**
      in the ``matched_ids`` column (not "nan", not "[]").
    - No duplicate IDs within a single entity's ``matched_ids`` list.
    - Every S1 entity appears exactly once.

    Parameters
    ----------
    result_df   : output of ``optimise_all_entities()``.
    output_path : destination file path.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, row in result_df.iterrows():
        matched = row["matched_ids"]
        # Deduplicate while preserving order (should already be unique)
        seen  = set()
        dedup = []
        for mid in matched:
            mid_str = str(mid).strip()
            if mid_str and mid_str not in seen:
                seen.add(mid_str)
                dedup.append(mid_str)

        rows.append({
            "source1_entity_id": str(row["source1_entity_id"]),
            "matched_ids":       ",".join(dedup),
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, sep="\t", index=False, encoding="utf-8")

    non_empty = (out_df["matched_ids"] != "").sum()
    empty     = (out_df["matched_ids"] == "").sum()
    log.info(
        "Saved matching_results.tsv → %s  |  entities_with_matches=%d  singletons=%d",
        output_path, non_empty, empty,
    )


def save_post_constraint_predictions(
    resolved_df: pd.DataFrame,
    cache_dir:   str,
) -> None:
    """Persist the post-constraint probability table for audit / Phase 7 validation."""
    path = Path(cache_dir) / "post_constraint_predictions.parquet"
    resolved_df.to_parquet(str(path), engine="pyarrow",
                           compression="snappy", index=False)
    log.info("Saved post_constraint_predictions.parquet → %s  (%d rows)",
             path, len(resolved_df))


# ===========================================================================
# Validation helpers
# ===========================================================================

def validate_matching_results(
    matching_results_path: str,
    candidate_pairs_path:  str,
    all_s1_ids:            List[str],
) -> bool:
    """
    Run assertion checks on ``matching_results.tsv``.

    Checks
    ------
    1. Every S1 entity ID appears exactly once.
    2. No duplicate matched IDs within any S1 entity's row.
    3. Every matched ID exists in the corresponding S1's candidate list
       in ``candidate_pairs.tsv``.
    4. No S2/S3 ID is matched to more than one S1 entity
       (constraint resolution correctness).

    Parameters
    ----------
    matching_results_path : path to the generated ``matching_results.tsv``.
    candidate_pairs_path  : path to ``candidate_pairs.tsv`` from Phase 3.
    all_s1_ids            : complete set of expected S1 entity IDs.

    Returns
    -------
    bool
        True if all checks pass, False otherwise (errors are logged).
    """
    log.info("Validating matching_results.tsv …")
    passed = True

    # Load outputs
    mr_df  = pd.read_csv(matching_results_path, sep="\t", dtype=str,
                         encoding="utf-8").fillna("")
    cp_df  = pd.read_csv(candidate_pairs_path,  sep="\t", dtype=str,
                         encoding="utf-8").fillna("")

    # Build candidate set per S1
    cp_dict: Dict[str, set] = {}
    for _, row in cp_df.iterrows():
        s1  = str(row["source1_entity_id"]).strip()
        ids = [x.strip() for x in str(row.get("candidate_entity_ids","")).split(",") if x.strip()]
        cp_dict[s1] = set(ids)

    # Check 1: every S1 appears exactly once
    result_s1_ids = mr_df["source1_entity_id"].tolist()
    dup_s1 = len(result_s1_ids) - len(set(result_s1_ids))
    if dup_s1 > 0:
        log.error("CHECK 1 FAILED: %d duplicate source1_entity_id rows", dup_s1)
        passed = False
    else:
        log.info("  CHECK 1 PASSED: No duplicate S1 rows")

    missing_s1 = set(all_s1_ids) - set(result_s1_ids)
    if missing_s1:
        log.error("CHECK 1b FAILED: %d S1 IDs missing from results: %s",
                  len(missing_s1), list(missing_s1)[:5])
        passed = False
    else:
        log.info("  CHECK 1b PASSED: All %d S1 entities present", len(all_s1_ids))

    # Checks 2, 3, 4
    global_matched: Dict[str, str] = {}   # cand_id → s1_id (constraint check)
    dup_in_row_count   = 0
    not_in_cands_count = 0
    constraint_violations = 0

    for _, row in mr_df.iterrows():
        s1_id      = str(row["source1_entity_id"]).strip()
        raw_ids    = str(row["matched_ids"]).strip()
        matched    = [x.strip() for x in raw_ids.split(",") if x.strip()]

        # Check 2: no duplicate IDs within a row
        if len(matched) != len(set(matched)):
            dup_in_row_count += 1

        # Check 3: all IDs must be in candidate list
        cand_set = cp_dict.get(s1_id, set())
        for mid in set(matched):
            if cand_set and mid not in cand_set:
                not_in_cands_count += 1

        # Check 4: constraint — each S2/S3 appears at most once across all S1s
        for mid in set(matched):
            if mid in global_matched and global_matched[mid] != s1_id:
                constraint_violations += 1
            else:
                global_matched[mid] = s1_id

    if dup_in_row_count > 0:
        log.error("CHECK 2 FAILED: %d rows contain duplicate matched IDs",
                  dup_in_row_count)
        passed = False
    else:
        log.info("  CHECK 2 PASSED: No duplicate IDs within any S1 row")

    if not_in_cands_count > 0:
        log.error("CHECK 3 FAILED: %d matched IDs not in candidate_pairs.tsv",
                  not_in_cands_count)
        passed = False
    else:
        log.info("  CHECK 3 PASSED: All matched IDs traceable to candidate_pairs.tsv")

    if constraint_violations > 0:
        log.error("CHECK 4 FAILED: %d S2/S3 IDs matched to >1 S1 entity",
                  constraint_violations)
        passed = False
    else:
        log.info("  CHECK 4 PASSED: Constraint holds — no S2/S3 matched to >1 S1")

    if passed:
        log.info("All validation checks PASSED.")
    else:
        log.error("Validation FAILED — see errors above.")
    return passed


# ===========================================================================
# Training-time evaluation (if ground truth available)
# ===========================================================================

def evaluate_with_ground_truth(
    result_df:   pd.DataFrame,
    gt_path:     str,
    beta:        float = 0.5,
) -> float:
    """
    Compute the true macro F-beta score against ground truth.

    Used as a sanity check during development / hyperparameter tuning.
    At competition time, ground truth is not available for the test set.

    Parameters
    ----------
    result_df : output of ``optimise_all_entities()``.
    gt_path   : path to ground_truth.tsv.
    beta      : F-beta parameter.

    Returns
    -------
    float
        Macro-averaged F-beta score.
    """
    gt = pd.read_csv(gt_path, sep="\t", dtype=str,
                     encoding="utf-8", encoding_errors="ignore").fillna("")
    gt.columns = [c.strip() for c in gt.columns]

    # Build true-positive set per S1
    gt_dict: Dict[str, set] = {}
    for _, row in gt.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        raw   = str(row.get("matched_ids", "")).strip()
        ids   = {x.strip() for x in raw.split(",") if x.strip()}
        gt_dict[s1_id] = ids

    f_scores = []
    for _, row in result_df.iterrows():
        s1_id     = str(row["source1_entity_id"])
        predicted = set(str(x) for x in row["matched_ids"])
        true      = gt_dict.get(s1_id, set())

        tp = len(predicted & true)
        fp = len(predicted - true)
        fn = len(true - predicted)

        if not predicted and not true:
            f_scores.append(1.0)   # true singleton, correctly predicted
            continue

        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if p == 0 and r == 0:
            f_scores.append(0.0)
        else:
            beta2 = beta ** 2
            f_scores.append((1 + beta2) * p * r / (beta2 * p + r))

    macro_f = float(np.mean(f_scores)) if f_scores else 0.0
    log.info(
        "Ground-truth macro F%.1f = %.4f  (across %d S1 entities)",
        beta, macro_f, len(f_scores),
    )
    return macro_f


# ===========================================================================
# Main pipeline entry point
# ===========================================================================

def run(
    config_path:       str,
    beta_override:     Optional[float] = None,
    skip_constraint:   bool = False,
) -> None:
    """
    Execute the full Phase 6 post-processing pipeline.

    Parameters
    ----------
    config_path       : path to ``configs/config.yaml``.
    beta_override     : if provided, overrides the beta from config.
    skip_constraint   : if True, skip constraint resolution (for ablation).
    """
    log.info("=" * 60)
    log.info("Phase 6 · Post-Processing & F0.5 Optimisation")
    log.info("=" * 60)

    cfg       = load_config(config_path)
    paths     = cfg["paths"]
    pp_cfg    = cfg["postprocessing"]
    cache_dir = paths["cache_dir"]
    output_dir= paths["output_dir"]

    beta = beta_override if beta_override is not None else \
           pp_cfg["f05_optimization"]["beta"]
    grid_steps = pp_cfg["f05_optimization"]["prob_grid_steps"]

    log.info("  beta=%.2f  grid_steps=%d  skip_constraint=%s",
             beta, grid_steps, skip_constraint)

    # ── 0. Load inputs ────────────────────────────────────────
    log.info("Step 0/4 · Loading calibrated predictions + S1 ID list")
    pred_df = load_predictions(cache_dir)
    log.info("  Predictions loaded: %d rows", len(pred_df))

    # Full list of S1 entity IDs (needed for singletons with 0 candidates)
    s1_df, _, _ = load_preprocessed(cache_dir)
    all_s1_ids: List[str] = s1_df["entity_id"].astype(str).tolist()
    log.info("  Total S1 entities: %d", len(all_s1_ids))

    # ── 1. Constraint Resolution ───────────────────────────────
    if not skip_constraint and pp_cfg["constraint_resolution"]["enabled"]:
        resolved_df = resolve_constraints(pred_df, prob_col="cal_prob")
    else:
        log.info("Step 1/4 · Constraint resolution SKIPPED")
        resolved_df = pred_df.copy()

    # Persist post-constraint predictions for Phase 7 validation
    save_post_constraint_predictions(resolved_df, cache_dir)

    # ── 2. Expected F0.5 Maximisation ─────────────────────────
    result_df = optimise_all_entities(
        resolved_df = resolved_df,
        all_s1_ids  = all_s1_ids,
        beta        = beta,
        prob_col    = "cal_prob",
        grid_steps  = grid_steps,
    )

    # ── 3. Write output ────────────────────────────────────────
    log.info("Step 3/4 · Writing matching_results.tsv")
    matching_path = paths["matching_results"]
    write_matching_results(result_df, matching_path)

    # ── 4. Evaluate if GT available ───────────────────────────
    log.info("Step 4/4 · Post-hoc evaluation")
    gt_path = paths.get("ground_truth", "")
    if gt_path and Path(gt_path).exists():
        macro_f = evaluate_with_ground_truth(result_df, gt_path, beta=beta)
        log.info(
            "  Training-set macro F0.5 = %.4f  "
            "(note: this is optimistic — uses training labels)",
            macro_f,
        )
    else:
        log.info("  Ground truth not found — skipping evaluation.")

    # Validate outputs
    log.info("  Running output validation …")
    validate_matching_results(
        matching_results_path = matching_path,
        candidate_pairs_path  = paths["candidate_pairs"],
        all_s1_ids            = all_s1_ids,
    )

    log.info("=" * 60)
    log.info("Phase 6 complete.")
    log.info("  matching_results.tsv → %s", matching_path)
    log.info("=" * 60)


# ===========================================================================
# Public API (used by Phase 7 utils)
# ===========================================================================

def load_matching_results(output_dir: str) -> pd.DataFrame:
    """
    Load the final matching results TSV.

    Returns
    -------
    pd.DataFrame
        Columns: source1_entity_id, matched_ids (string, comma-separated).
    """
    path = Path(output_dir) / "matching_results.tsv"
    if not path.exists():
        raise FileNotFoundError(
            f"matching_results.tsv not found: {path}\n"
            "Run: python -m src.optimizer --config configs/config.yaml"
        )
    df = pd.read_csv(str(path), sep="\t", dtype=str,
                     encoding="utf-8").fillna("")
    log.info("Loaded matching_results.tsv: %d rows", len(df))
    return df


# ===========================================================================
# CLI entry point
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 6 — Post-Processing & Expected F0.5 Maximisation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--beta", type=float, default=None,
        help="Override the F-beta parameter (default: read from config).",
    )
    parser.add_argument(
        "--skip-constraint", dest="skip_constraint", action="store_true",
        help="Skip constraint resolution (S2/S3 → max-1 S1). For ablation only.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        config_path     = args.config,
        beta_override   = args.beta,
        skip_constraint = args.skip_constraint,
    )
