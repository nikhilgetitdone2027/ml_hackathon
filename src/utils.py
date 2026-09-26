"""
src/utils.py
============
Phase 7 — Output Generation, Validation & Pipeline Utilities

Responsibilities
----------------
This module provides three categories of functionality:

1. Output Validation (standalone CLI)
   Runs the full suite of assertions against the two required submission files:
   - ``output/candidate_pairs.tsv``   (blocking output)
   - ``output/matching_results.tsv``  (final predictions)

   Checks performed:
   a. Schema check    : correct columns, UTF-8 encoding, tab separator.
   b. Coverage check  : every S1 entity appears exactly once in each file.
   c. Subset check    : every ID in matching_results exists in candidate_pairs.
   d. Duplicate check : no duplicate matched IDs within any S1 row.
   e. Constraint check: no S2/S3 ID is matched to more than one S1 entity.
   f. Format check    : matched_ids is either empty or a valid comma-sep list
                        (no brackets, no quotes, no whitespace padding).
   g. Self-consistency: candidate_pairs row count == matching_results row count
                        (both contain exactly one row per S1 entity).

2. I/O Utilities (imported by other modules)
   - ``read_tsv`` / ``write_tsv``  : safe TSV read/write with BOM handling.
   - ``load_all_s1_ids``           : loads the canonical S1 ID list.
   - ``load_submission_files``     : loads both submission TSVs at once.
   - ``summarise_submission``      : prints a rich summary of the submission.

3. Smoke Test (built-in synthetic test)
   ``run_smoke_test()`` creates a 10-entity synthetic dataset in memory,
   runs it through the optimizer's Expected-F0.5 logic, and validates the
   output — no external data required.  Useful for CI or environment checks.

CLI usage
---------
    # Validate outputs after running the pipeline:
    python -m src.utils --validate --config configs/config.yaml

    # Run the built-in smoke test:
    python -m src.utils --smoke-test

    # Full summary of submission files:
    python -m src.utils --summary --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import yaml

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
# I/O Utilities
# ===========================================================================

def read_tsv(path: str, required_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Read a UTF-8 tab-separated file into a DataFrame.

    Handles BOM characters (``encoding_errors='ignore'``), strips whitespace
    from column names, and optionally validates that required columns exist.

    Parameters
    ----------
    path          : path to the TSV file.
    required_cols : if provided, raises ``ValueError`` if any column is missing.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError : if *path* does not exist.
    ValueError        : if a required column is missing.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(
        str(p), sep="\t", dtype=str,
        encoding="utf-8", encoding_errors="ignore",
        low_memory=False,
    )
    df.columns = [c.strip() for c in df.columns]
    df = df.fillna("")

    if required_cols:
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(
                f"File {path!r} is missing columns: {missing}\n"
                f"Found: {list(df.columns)}"
            )
    return df


def write_tsv(df: pd.DataFrame, path: str) -> None:
    """
    Write a DataFrame to a UTF-8 tab-separated file with a header row.

    Parameters
    ----------
    df   : DataFrame to write.
    path : destination path (parent directories are created if needed).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, encoding="utf-8")
    log.info("Written %d rows → %s", len(df), path)


def load_all_s1_ids(source1_path: str) -> List[str]:
    """
    Load the complete, ordered list of Source 1 entity IDs.

    Parameters
    ----------
    source1_path : path to the raw (or preprocessed) Source 1 TSV / Parquet.

    Returns
    -------
    List[str]
        All S1 entity IDs in file order.
    """
    p = Path(source1_path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(str(p))
    else:
        df = read_tsv(str(p), required_cols=["entity_id"])
    return df["entity_id"].astype(str).str.strip().tolist()


def load_submission_files(
    candidate_pairs_path: str,
    matching_results_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load both required submission files.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (candidate_pairs_df, matching_results_df)
    """
    cp = read_tsv(candidate_pairs_path,
                  required_cols=["source1_entity_id", "candidate_entity_ids"])
    mr = read_tsv(matching_results_path,
                  required_cols=["source1_entity_id", "matched_ids"])
    return cp, mr


# ===========================================================================
# Individual validation checks
# ===========================================================================

class ValidationError(Exception):
    """Raised when a hard assertion fails during output validation."""


def _check_schema(df: pd.DataFrame, required_cols: List[str], label: str) -> None:
    """Assert that *df* has all *required_cols*."""
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValidationError(
            f"[{label}] Missing columns: {missing}. Found: {list(df.columns)}"
        )
    log.info("  [%s] Schema OK — columns: %s", label, list(df.columns))


def _check_coverage(
    df: pd.DataFrame,
    all_s1_ids: List[str],
    label: str,
) -> None:
    """
    Assert every S1 entity appears exactly once in *df*.

    Checks:
    - No S1 entity is missing from the file.
    - No S1 entity appears more than once.
    """
    result_ids = df["source1_entity_id"].astype(str).str.strip().tolist()
    result_set = set(result_ids)
    expected_set = set(str(x) for x in all_s1_ids)

    missing = expected_set - result_set
    extra   = result_set - expected_set
    dups    = len(result_ids) - len(result_set)

    errors = []
    if missing:
        errors.append(
            f"{len(missing)} S1 entities missing: {sorted(missing)[:5]}..."
        )
    if extra:
        errors.append(
            f"{len(extra)} unexpected S1 entities: {sorted(extra)[:5]}..."
        )
    if dups > 0:
        errors.append(f"{dups} duplicate S1 entity rows")

    if errors:
        raise ValidationError(f"[{label}] Coverage check FAILED:\n  " +
                              "\n  ".join(errors))
    log.info("  [%s] Coverage OK — %d S1 entities, all present exactly once",
             label, len(all_s1_ids))


def _check_subset(
    mr_df: pd.DataFrame,
    cp_df: pd.DataFrame,
) -> int:
    """
    Assert every matched ID in matching_results exists in candidate_pairs.

    Returns
    -------
    int
        Number of violations found (0 = pass).
    """
    # Build per-S1 candidate sets from candidate_pairs
    cp_sets: Dict[str, Set[str]] = {}
    for _, row in cp_df.iterrows():
        s1  = str(row["source1_entity_id"]).strip()
        raw = str(row.get("candidate_entity_ids", "")).strip()
        cp_sets[s1] = {x.strip() for x in raw.split(",") if x.strip()}

    violations = 0
    bad_pairs: List[str] = []
    for _, row in mr_df.iterrows():
        s1  = str(row["source1_entity_id"]).strip()
        raw = str(row.get("matched_ids", "")).strip()
        if not raw:
            continue
        for mid in raw.split(","):
            mid = mid.strip()
            if mid and mid not in cp_sets.get(s1, set()):
                violations += 1
                if len(bad_pairs) < 5:
                    bad_pairs.append(f"S1={s1!r} matched={mid!r}")

    if violations > 0:
        raise ValidationError(
            f"Subset check FAILED: {violations} matched IDs not in candidate_pairs.\n"
            f"  Examples: {bad_pairs}"
        )
    log.info("  Subset check OK — all matched IDs traceable to candidate_pairs.tsv")
    return 0


def _check_no_duplicates(mr_df: pd.DataFrame) -> int:
    """
    Assert no duplicate matched IDs within any S1 entity's row.

    Returns
    -------
    int
        Number of S1 rows that contain duplicates (0 = pass).
    """
    dup_rows = 0
    examples: List[str] = []
    for _, row in mr_df.iterrows():
        s1  = str(row["source1_entity_id"]).strip()
        raw = str(row.get("matched_ids", "")).strip()
        ids = [x.strip() for x in raw.split(",") if x.strip()]
        if len(ids) != len(set(ids)):
            dup_rows += 1
            if len(examples) < 3:
                examples.append(f"S1={s1!r}")

    if dup_rows > 0:
        raise ValidationError(
            f"Duplicate check FAILED: {dup_rows} rows contain duplicate matched IDs.\n"
            f"  Examples: {examples}"
        )
    log.info("  Duplicate check OK — no duplicate IDs within any S1 row")
    return 0


def _check_constraint(mr_df: pd.DataFrame) -> int:
    """
    Assert each S2/S3 ID appears in at most one S1 entity's match list.

    This is the fundamental uniqueness constraint: S1 is deduplicated, so
    each S2/S3 record can only belong to one real-world entity.

    Returns
    -------
    int
        Number of S2/S3 IDs that appear in more than one S1's list (0 = pass).
    """
    cand_to_s1: Dict[str, str] = {}
    violations = 0
    examples: List[str] = []

    for _, row in mr_df.iterrows():
        s1  = str(row["source1_entity_id"]).strip()
        raw = str(row.get("matched_ids", "")).strip()
        for mid in raw.split(","):
            mid = mid.strip()
            if not mid:
                continue
            if mid in cand_to_s1 and cand_to_s1[mid] != s1:
                violations += 1
                if len(examples) < 5:
                    examples.append(
                        f"ID={mid!r} claimed by S1={s1!r} and S1={cand_to_s1[mid]!r}"
                    )
            else:
                cand_to_s1[mid] = s1

    if violations > 0:
        raise ValidationError(
            f"Constraint check FAILED: {violations} S2/S3 IDs matched to >1 S1.\n"
            f"  Examples: {examples}"
        )
    log.info("  Constraint check OK — no S2/S3 ID matched to more than one S1")
    return 0


def _check_format(mr_df: pd.DataFrame) -> None:
    """
    Assert the ``matched_ids`` column uses only valid comma-separated format.

    Valid:   ``""``  or  ``"id1,id2,id3"``
    Invalid: ``"[]"`` or ``"['id1','id2']"`` or ``"  id1 , id2 "`` (with padding)
    """
    bad = []
    for _, row in mr_df.iterrows():
        raw = str(row.get("matched_ids", ""))
        if raw in ("", "nan"):
            continue
        # Reject bracket/quote characters (JSON-style lists)
        if any(c in raw for c in "[]'\""):
            bad.append(f"S1={row['source1_entity_id']!r}: {raw[:60]!r}")
            continue
        # Reject leading/trailing whitespace around IDs
        parts = raw.split(",")
        for p in parts:
            if p != p.strip():
                bad.append(
                    f"S1={row['source1_entity_id']!r}: whitespace-padded ID {p!r}"
                )
                break

    if bad:
        raise ValidationError(
            f"Format check FAILED: {len(bad)} rows with invalid matched_ids format.\n"
            f"  Examples: {bad[:3]}"
        )
    log.info("  Format check OK — matched_ids column is clean comma-separated")


def _check_self_consistency(cp_df: pd.DataFrame, mr_df: pd.DataFrame) -> None:
    """Assert both files have the same number of rows (one per S1 entity)."""
    if len(cp_df) != len(mr_df):
        raise ValidationError(
            f"Self-consistency FAILED: candidate_pairs has {len(cp_df)} rows "
            f"but matching_results has {len(mr_df)} rows. Both must equal |S1|."
        )
    log.info("  Self-consistency OK — both files have %d rows", len(cp_df))


# ===========================================================================
# Master validation runner
# ===========================================================================

def validate_outputs(
    candidate_pairs_path:  str,
    matching_results_path: str,
    all_s1_ids:            List[str],
    raise_on_error:        bool = True,
) -> bool:
    """
    Run the full suite of output validation checks.

    Executes all checks in order; logs each result.  On the first failure,
    either raises ``ValidationError`` (default) or logs the error and returns
    ``False`` (if ``raise_on_error=False``).

    Parameters
    ----------
    candidate_pairs_path  : path to ``candidate_pairs.tsv``.
    matching_results_path : path to ``matching_results.tsv``.
    all_s1_ids            : complete list of S1 entity IDs from source1.
    raise_on_error        : if True, raise on first failure. If False, log all
                            failures and return False at the end.

    Returns
    -------
    bool
        True if all checks pass, False if any check fails (when not raising).
    """
    log.info("=" * 56)
    log.info("Output Validation Suite")
    log.info("  candidate_pairs  : %s", candidate_pairs_path)
    log.info("  matching_results : %s", matching_results_path)
    log.info("  S1 entities      : %d", len(all_s1_ids))
    log.info("=" * 56)
    t0 = time.perf_counter()

    failures: List[str] = []

    def _run_check(name: str, fn, *args, **kwargs) -> bool:
        try:
            fn(*args, **kwargs)
            log.info("  [PASS] %s", name)
            return True
        except (ValidationError, FileNotFoundError, ValueError) as exc:
            msg = f"[FAIL] {name}: {exc}"
            log.error(msg)
            failures.append(msg)
            if raise_on_error:
                raise
            return False

    # ── Load files ───────────────────────────────────────────
    try:
        cp_df, mr_df = load_submission_files(
            candidate_pairs_path, matching_results_path
        )
    except (FileNotFoundError, ValueError) as exc:
        log.error("Cannot load submission files: %s", exc)
        if raise_on_error:
            raise
        return False

    # ── Run all checks ────────────────────────────────────────
    _run_check(
        "Schema — candidate_pairs",
        _check_schema, cp_df,
        ["source1_entity_id", "candidate_entity_ids"],
        "candidate_pairs",
    )
    _run_check(
        "Schema — matching_results",
        _check_schema, mr_df,
        ["source1_entity_id", "matched_ids"],
        "matching_results",
    )
    _run_check(
        "Coverage — candidate_pairs",
        _check_coverage, cp_df, all_s1_ids, "candidate_pairs",
    )
    _run_check(
        "Coverage — matching_results",
        _check_coverage, mr_df, all_s1_ids, "matching_results",
    )
    _run_check(
        "Self-consistency (row counts match)",
        _check_self_consistency, cp_df, mr_df,
    )
    _run_check(
        "Subset (matched IDs in candidate_pairs)",
        _check_subset, mr_df, cp_df,
    )
    _run_check(
        "Duplicate IDs within S1 rows",
        _check_no_duplicates, mr_df,
    )
    _run_check(
        "Constraint (S2/S3 ID → at most 1 S1)",
        _check_constraint, mr_df,
    )
    _run_check(
        "Format (comma-sep, no brackets or padding)",
        _check_format, mr_df,
    )

    elapsed = time.perf_counter() - t0
    n_checks = 9
    n_failed = len(failures)
    n_passed = n_checks - n_failed

    log.info("=" * 56)
    if n_failed == 0:
        log.info("ALL %d CHECKS PASSED  (%.2fs)", n_checks, elapsed)
        log.info("=" * 56)
        return True
    else:
        log.error("%d/%d CHECKS FAILED  (%.2fs)", n_failed, n_checks, elapsed)
        for f in failures:
            log.error("  %s", f)
        log.info("=" * 56)
        return False


# ===========================================================================
# Submission summary
# ===========================================================================

def summarise_submission(
    candidate_pairs_path:  str,
    matching_results_path: str,
) -> None:
    """
    Print a detailed human-readable summary of the two submission files.

    Useful for a final sanity check before submitting to the leaderboard.

    Summary includes
    ----------------
    - Total S1 entities
    - Entities with at least one match vs. singletons
    - Distribution of match counts (min, median, p75, p90, max)
    - Total unique S2/S3 IDs matched
    - Candidate pair counts from blocking stage
    - Average candidates per S1 entity
    """
    log.info("=" * 56)
    log.info("Submission Summary")
    log.info("=" * 56)

    cp_df, mr_df = load_submission_files(
        candidate_pairs_path, matching_results_path
    )

    # ── Matching results stats ─────────────────────────────────
    mr_df["n_matched"] = mr_df["matched_ids"].apply(
        lambda x: len([i for i in str(x).split(",") if i.strip()])
    )

    total_s1        = len(mr_df)
    has_matches     = (mr_df["n_matched"] > 0).sum()
    singletons      = (mr_df["n_matched"] == 0).sum()
    total_matched   = mr_df["n_matched"].sum()
    match_counts    = mr_df["n_matched"]

    log.info("  matching_results.tsv")
    log.info("    Total S1 entities  : %d", total_s1)
    log.info("    With matches       : %d  (%.1f%%)",
             has_matches, 100 * has_matches / total_s1 if total_s1 else 0)
    log.info("    Singletons (k=0)   : %d  (%.1f%%)",
             singletons, 100 * singletons / total_s1 if total_s1 else 0)
    log.info("    Total IDs matched  : %d", total_matched)
    log.info("    Match count distrib:")
    log.info("      min=%.0f  p25=%.0f  median=%.0f  p75=%.0f  p90=%.0f  max=%.0f",
             match_counts.min(),
             match_counts.quantile(0.25),
             match_counts.median(),
             match_counts.quantile(0.75),
             match_counts.quantile(0.90),
             match_counts.max())

    # ── Candidate pairs stats ─────────────────────────────────
    cp_df["n_cands"] = cp_df["candidate_entity_ids"].apply(
        lambda x: len([i for i in str(x).split(",") if i.strip()])
    )
    log.info("")
    log.info("  candidate_pairs.tsv")
    log.info("    Total S1 entities  : %d", len(cp_df))
    log.info("    Mean candidates/S1 : %.1f", cp_df["n_cands"].mean())
    log.info("    Median candidates  : %.1f", cp_df["n_cands"].median())
    log.info("    Max candidates     : %d", cp_df["n_cands"].max())
    log.info("    S1 with 0 cands    : %d", (cp_df["n_cands"] == 0).sum())

    # ── Unique S2/S3 IDs across all predictions ───────────────
    all_matched = set()
    for raw in mr_df["matched_ids"]:
        for mid in str(raw).split(","):
            mid = mid.strip()
            if mid:
                all_matched.add(mid)
    log.info("")
    log.info("  Unique S2/S3 IDs in matching_results : %d", len(all_matched))
    log.info("=" * 56)


# ===========================================================================
# Built-in smoke test
# ===========================================================================

def run_smoke_test() -> bool:
    """
    Run a self-contained synthetic smoke test — no external data needed.

    Creates 10 S1 entities and 15 S2/S3 candidates in memory, applies the
    Expected F0.5 optimiser, writes the outputs, and validates them.

    Returns
    -------
    bool
        True if all checks pass.
    """
    import tempfile, os
    from src.optimizer import (
        resolve_constraints,
        optimise_all_entities,
        write_matching_results,
    )

    log.info("=" * 56)
    log.info("Running built-in smoke test (synthetic 10-entity dataset)")
    log.info("=" * 56)

    # ── Synthetic data ────────────────────────────────────────
    rng = np.random.default_rng(42)

    # 10 S1 entities
    s1_ids = [f"S1_{i:03d}" for i in range(10)]

    # 15 candidates (some shared across S1s to test constraint resolution)
    cand_ids = [f"C_{j:03d}" for j in range(15)]

    # True matches (ground truth): S1_000→C_000, S1_001→C_001,C_002, etc.
    true_matches = {
        "S1_000": {"C_000"},
        "S1_001": {"C_001", "C_002"},
        "S1_002": {"C_003"},
        "S1_003": set(),           # true singleton
        "S1_004": {"C_004"},
        "S1_005": {"C_005", "C_006", "C_007"},
        "S1_006": set(),           # true singleton
        "S1_007": {"C_008"},
        "S1_008": {"C_009"},
        "S1_009": {"C_010"},
    }

    # Build synthetic predictions: high prob for true matches, low for others
    rows = []
    for s1 in s1_ids:
        # Each S1 gets 5 candidates
        assigned_cands = list(true_matches.get(s1, set()))
        # Fill up to 5 with random non-true candidates
        other_cands = [c for c in cand_ids if c not in assigned_cands]
        fill = rng.choice(other_cands,
                          size=min(5 - len(assigned_cands), len(other_cands)),
                          replace=False).tolist()
        all_cands = assigned_cands + fill

        for cid in all_cands:
            is_true = cid in true_matches.get(s1, set())
            prob = float(rng.beta(9, 1) if is_true else rng.beta(1, 9))
            rows.append({
                "source1_entity_id":   s1,
                "candidate_entity_id": cid,
                "cal_prob":            prob,
            })

    # Add a deliberate constraint violation: C_011 claimed by two S1s
    rows.append({"source1_entity_id": "S1_000", "candidate_entity_id": "C_011", "cal_prob": 0.91})
    rows.append({"source1_entity_id": "S1_001", "candidate_entity_id": "C_011", "cal_prob": 0.75})

    pred_df = pd.DataFrame(rows)
    log.info("  Synthetic predictions: %d rows", len(pred_df))

    # ── Step 1: Constraint resolution ────────────────────────
    resolved = resolve_constraints(pred_df, prob_col="cal_prob")
    # C_011 should now only belong to S1_000 (higher prob 0.91 > 0.75)
    c011_owners = resolved[resolved["candidate_entity_id"] == "C_011"]["source1_entity_id"].tolist()
    assert c011_owners == ["S1_000"], (
        f"Constraint resolution FAILED: C_011 belongs to {c011_owners}, expected ['S1_000']"
    )
    log.info("  Constraint resolution: PASS (C_011 correctly assigned to S1_000)")

    # ── Step 2: Expected F0.5 optimisation ─────────────────────
    result_df = optimise_all_entities(
        resolved_df = resolved,
        all_s1_ids  = s1_ids,
        beta        = 0.5,
    )
    assert len(result_df) == len(s1_ids), (
        f"Result has {len(result_df)} rows, expected {len(s1_ids)}"
    )

    # True singletons should predict k=0
    for s1 in ["S1_003", "S1_006"]:
        k = result_df.loc[result_df["source1_entity_id"] == s1, "best_k"].values[0]
        assert k == 0, f"Singleton {s1} predicted k={k}, expected 0"
    log.info("  Singleton detection: PASS (S1_003, S1_006 correctly predict k=0)")

    # ── Step 3: Write & validate outputs ─────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        mr_path = os.path.join(tmpdir, "matching_results.tsv")
        write_matching_results(result_df, mr_path)

        # Build a synthetic candidate_pairs.tsv for validation
        cp_rows = []
        for s1 in s1_ids:
            sub = resolved[resolved["source1_entity_id"] == s1]
            cand_list = ",".join(sub["candidate_entity_id"].astype(str).tolist())
            cp_rows.append({
                "source1_entity_id":    s1,
                "candidate_entity_ids": cand_list,
            })
        cp_path = os.path.join(tmpdir, "candidate_pairs.tsv")
        pd.DataFrame(cp_rows).to_csv(cp_path, sep="\t", index=False)

        passed = validate_outputs(
            candidate_pairs_path  = cp_path,
            matching_results_path = mr_path,
            all_s1_ids            = s1_ids,
            raise_on_error        = False,
        )

    if passed:
        log.info("Smoke test: ALL CHECKS PASSED")
    else:
        log.error("Smoke test: SOME CHECKS FAILED")
    log.info("=" * 56)
    return passed


# ===========================================================================
# Main pipeline entry point
# ===========================================================================

def run(
    config_path:   str,
    do_validate:   bool = False,
    do_summary:    bool = False,
    do_smoke_test: bool = False,
) -> None:
    """
    Phase 7 CLI entry point.

    Modes (any combination can be combined):
    - ``--validate``   : run the full validation suite on pipeline outputs.
    - ``--summary``    : print a human-readable submission summary.
    - ``--smoke-test`` : run the built-in synthetic smoke test.
    """
    log.info("=" * 60)
    log.info("Phase 7 · Output Validation & Utilities")
    log.info("=" * 60)

    if do_smoke_test:
        ok = run_smoke_test()
        if not ok:
            log.error("Smoke test FAILED — check implementation before running on real data.")
            sys.exit(1)
        if not do_validate and not do_summary:
            return

    cfg = load_config(config_path)
    paths = cfg["paths"]

    # Determine S1 ID list
    # Prefer preprocessed Parquet (faster); fall back to raw TSV
    cache_dir   = paths["cache_dir"]
    s1_parquet  = Path(cache_dir) / "preprocessed_source1.parquet"
    if s1_parquet.exists():
        s1_df   = pd.read_parquet(str(s1_parquet))
        all_s1_ids = s1_df["entity_id"].astype(str).str.strip().tolist()
        log.info("  S1 IDs loaded from preprocessed cache (%d entities)", len(all_s1_ids))
    else:
        all_s1_ids = load_all_s1_ids(paths["source1"])
        log.info("  S1 IDs loaded from raw source1.tsv (%d entities)", len(all_s1_ids))

    if do_summary:
        summarise_submission(
            candidate_pairs_path  = paths["candidate_pairs"],
            matching_results_path = paths["matching_results"],
        )

    if do_validate:
        passed = validate_outputs(
            candidate_pairs_path  = paths["candidate_pairs"],
            matching_results_path = paths["matching_results"],
            all_s1_ids            = all_s1_ids,
            raise_on_error        = False,
        )
        if not passed:
            log.error("Validation FAILED. Fix the issues above before submitting.")
            sys.exit(1)
        log.info("Validation PASSED. Submission files are ready.")

    log.info("=" * 60)
    log.info("Phase 7 complete.")
    log.info("=" * 60)


# ===========================================================================
# CLI entry point
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 7 — Output Validation & Pipeline Utilities",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--validate", dest="do_validate", action="store_true",
        help="Run the full validation suite on pipeline outputs.",
    )
    parser.add_argument(
        "--summary", dest="do_summary", action="store_true",
        help="Print a rich summary of the submission files.",
    )
    parser.add_argument(
        "--smoke-test", dest="do_smoke_test", action="store_true",
        help="Run the built-in synthetic smoke test (no data required).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if not any([args.do_validate, args.do_summary, args.do_smoke_test]):
        print("No action specified. Use --validate, --summary, or --smoke-test.")
        print("Run with --help for usage.")
        sys.exit(0)
    run(
        config_path   = args.config,
        do_validate   = args.do_validate,
        do_summary    = args.do_summary,
        do_smoke_test = args.do_smoke_test,
    )
