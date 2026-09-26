"""
src/features.py
===============
Phase 4 — Graph & Competition Feature Engineering

For every (S1, S2/S3) candidate pair produced by the blocking layer this
module computes a rich feature vector that the XGBoost classifier (Phase 5)
will train on.

Feature groups
--------------
A. Base string similarities
   ├─ jaro_winkler_name      Jaro-Winkler on normalised business name
   ├─ jaro_winkler_addr      Jaro-Winkler on normalised address
   ├─ lev_norm_name          Levenshtein ratio (0-1) on name
   ├─ lev_norm_addr          Levenshtein ratio (0-1) on address
   ├─ lev_norm_search        Levenshtein ratio on full search_text
   ├─ jaccard_name           Jaccard on name token sets
   ├─ jaccard_addr           Jaccard on address token sets
   ├─ jaccard_search         Jaccard on search_text token sets
   └─ sem_score              MiniLM cosine similarity (from blocking cache)

B. Number Veto
   ├─ pin_conflict            1 if both sides have a PIN and they differ
   ├─ street_conflict         1 if both sides have a street number and differ
   └─ number_veto             1 if pin_conflict OR street_conflict

C. Competition features  (graph: S2/S3 node → competing S1 nodes)
   ├─ margin_sem             sem_score − next-best S1's sem_score (same cand)
   ├─ margin_tfidf           tfidf_score − next-best S1's tfidf_score
   ├─ margin_jw              jaro_winkler_name − next-best S1's JW score
   ├─ rank_among_s1          rank of this S1 among all S1s that retrieved cand
   └─ n_competing_s1         how many S1s retrieved this S2/S3 candidate

D. Transitivity / cluster coherence  (graph: S1 node → its candidate cluster)
   ├─ trans_mean_sem          mean sem_score of S1's other candidates
   ├─ trans_std_sem           std of S1's sem_score distribution
   ├─ trans_mean_jw           mean JW of S1's other name candidates
   ├─ trans_n_candidates      total candidate count for this S1
   └─ trans_cross_source_sim  mean pairwise JW between S2-side and S3-side
                              candidates of the same S1 (cluster consistency)

E. Meta / surface features
   ├─ name_len_ratio          min/max of len(s1_name)/len(cand_name)
   ├─ addr_len_ratio          same for addresses
   ├─ country_match           1 if normalised country strings are identical
   ├─ common_token_count      |tokens(s1_name) ∩ tokens(cand_name)|
   └─ tfidf_score             raw TF-IDF cosine from blocking (as feature)

Output
------
Parquet file at ``output/cache/features.parquet``
Columns: source1_entity_id, candidate_entity_id, label (if GT available),
         + all feature columns listed above.

CLI usage
---------
    python -m src.features --config configs/config.yaml
    python -m src.features --config configs/config.yaml --no-labels  # inference
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import yaml
from rapidfuzz import fuzz as rfuzz
from rapidfuzz import distance as rdist
from tqdm import tqdm

from src.preprocessor import load_preprocessed
from src.blocker import load_candidates

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
# A. Base string similarity helpers
# ===========================================================================

def jaro_winkler(s1: str, s2: str, prefix_weight: float = 0.1) -> float:
    """
    Jaro-Winkler similarity in [0, 1].

    Uses rapidfuzz for a fast C-extension implementation.
    The prefix weight (p) rewards strings sharing a common prefix — well
    suited to business names where the first word is highly discriminative.

    Parameters
    ----------
    s1, s2         : pre-normalised strings.
    prefix_weight  : Jaro-Winkler p parameter (default 0.1, standard value).

    Examples
    --------
    >>> jaro_winkler("acme corporation", "acme corp")
    0.9...
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    # rapidfuzz returns similarity in [0, 100]; normalise to [0, 1]
    return rfuzz.jaro_winkler(s1, s2, prefix_weight=prefix_weight) / 100.0


def levenshtein_norm(s1: str, s2: str) -> float:
    """
    Normalised Levenshtein similarity in [0, 1].

    Normalised as: 1 - (edit_distance / max(len(s1), len(s2))).
    Returns 1.0 for two empty strings, 0.0 for (empty, non-empty).
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    max_len = max(len(s1), len(s2))
    dist = rdist.Levenshtein.distance(s1, s2)
    return 1.0 - dist / max_len


def jaccard_tokens(s1: str, s2: str) -> float:
    """
    Jaccard similarity over the token-set representations of two strings.

    J(A, B) = |A ∩ B| / |A ∪ B|

    Whitespace-tokenises both strings; returns 1.0 if both are empty.

    Examples
    --------
    >>> jaccard_tokens("acme corporation limited", "acme corp ltd")
    0.2  # only 'acme' in common out of {'acme','corporation','limited','corp','ltd'}
    """
    if not s1 and not s2:
        return 1.0
    t1: Set[str] = set(s1.split())
    t2: Set[str] = set(s2.split())
    if not t1 and not t2:
        return 1.0
    intersection = len(t1 & t2)
    union = len(t1 | t2)
    return intersection / union if union else 0.0


def common_token_count(s1: str, s2: str) -> int:
    """Absolute count of tokens shared between two strings."""
    t1 = set(s1.split())
    t2 = set(s2.split())
    return len(t1 & t2)


def len_ratio(s1: str, s2: str) -> float:
    """
    Length similarity ratio: min(len) / max(len).

    Returns 1.0 when both are empty, 0.0 when one is empty and the other not.
    """
    l1, l2 = len(s1), len(s2)
    if l1 == 0 and l2 == 0:
        return 1.0
    if l1 == 0 or l2 == 0:
        return 0.0
    return min(l1, l2) / max(l1, l2)


# ===========================================================================
# B. Number Veto helpers
# ===========================================================================

def _extract_numbers(text: str, pattern: re.Pattern) -> Set[str]:
    """Return all non-overlapping matches of *pattern* in *text* as a set."""
    return set(pattern.findall(text))


def number_veto_flags(
    s1_addr:  str,
    s2_addr:  str,
    s1_pin:   str,
    s2_pin:   str,
    s1_street: str,
    s2_street: str,
    pin_pattern:    re.Pattern,
    street_pattern: re.Pattern,
) -> Tuple[int, int, int]:
    """
    Compute three binary Number Veto flags for one candidate pair.

    Logic
    -----
    - ``pin_conflict``    : both sides have a PIN code *and* they differ.
    - ``street_conflict`` : both sides have a street number *and* they differ.
    - ``number_veto``     : 1 if *either* conflict fires.

    A missing number (empty string) is treated as "unknown" — it does NOT
    trigger a conflict.  This is intentional: a noisy S2/S3 record may simply
    have a missing postcode without actually being a different location.

    Parameters
    ----------
    s1_addr, s2_addr   : normalised address strings (for fallback pattern match).
    s1_pin, s2_pin     : pre-extracted PIN codes (from preprocessor).
    s1_street, s2_street: pre-extracted street numbers (from preprocessor).
    pin_pattern        : compiled PIN regex (used if stored columns are empty).
    street_pattern     : compiled street-number regex.

    Returns
    -------
    Tuple[int, int, int]
        (pin_conflict, street_conflict, number_veto)
    """
    # Use pre-extracted values; fall back to inline regex on the address string
    p1 = s1_pin    if s1_pin    else (pin_pattern.search(s1_addr)   or type('', (), {'group': lambda *_: ''})()).group(0) if s1_pin == "" else s1_pin
    p2 = s2_pin    if s2_pin    else (pin_pattern.search(s2_addr)   or type('', (), {'group': lambda *_: ''})()).group(0) if s2_pin == "" else s2_pin

    # Simpler inline fallback using walrus
    if not s1_pin:
        m = pin_pattern.search(s1_addr)
        p1 = m.group(0) if m else ""
    else:
        p1 = s1_pin

    if not s2_pin:
        m = pin_pattern.search(s2_addr)
        p2 = m.group(0) if m else ""
    else:
        p2 = s2_pin

    if not s1_street:
        m = street_pattern.search(s1_addr)
        st1 = m.group(0) if m else ""
    else:
        st1 = s1_street

    if not s2_street:
        m = street_pattern.search(s2_addr)
        st2 = m.group(0) if m else ""
    else:
        st2 = s2_street

    # Conflict = both present AND different
    pin_conflict    = int(bool(p1 and p2 and p1 != p2))
    street_conflict = int(bool(st1 and st2 and st1 != st2))
    veto            = int(bool(pin_conflict or street_conflict))

    return pin_conflict, street_conflict, veto


# ===========================================================================
# C. Competition features (graph: candidate → competing S1s)
# ===========================================================================

def compute_competition_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute competition (margin) features for every candidate pair.

    For each unique S2/S3 candidate ID, rank all S1 entities that retrieved
    it by their ``sem_score``.  Then for the current (S1, cand) pair compute:

    - ``margin_sem``       = current_sem   - second_best_sem   (0 if only one S1)
    - ``margin_tfidf``     = current_tfidf - second_best_tfidf
    - ``rank_among_s1``    = 1-based rank of this S1 for this candidate
    - ``n_competing_s1``   = total number of S1s that retrieved this candidate

    A large positive margin means this S1 is the "clear winner" for this S2/S3
    record — a very strong signal for a true match.

    Parameters
    ----------
    df : DataFrame with at least:
         [source1_entity_id, candidate_entity_id, sem_score, tfidf_score]

    Returns
    -------
    pd.DataFrame
        Input DataFrame with four new columns appended.
    """
    log.info("  Computing competition features …")
    t0 = time.perf_counter()

    df = df.copy()

    # Rank S1s per candidate by sem_score (descending)
    df["rank_among_s1"] = (
        df.groupby("candidate_entity_id")["sem_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    df["n_competing_s1"] = df.groupby("candidate_entity_id")[
        "source1_entity_id"
    ].transform("count")

    # For margin: shift the sorted scores within each candidate group
    df_sorted = df.sort_values(
        ["candidate_entity_id", "sem_score"], ascending=[True, False]
    )

    # Second-best sem_score for each candidate
    second_best_sem = (
        df_sorted.groupby("candidate_entity_id")["sem_score"]
        .transform(lambda x: x.shift(-1).fillna(x))  # second row; if only one, itself
    )
    df["margin_sem"] = (df["sem_score"] - second_best_sem.reindex(df.index)).fillna(0.0)

    # Clamp: margin is only meaningful when this S1 is ranked #1
    df["margin_sem"] = df["margin_sem"].clip(lower=0.0)

    # Same for tfidf
    df_sorted_tfidf = df.sort_values(
        ["candidate_entity_id", "tfidf_score"], ascending=[True, False]
    )
    second_best_tfidf = (
        df_sorted_tfidf.groupby("candidate_entity_id")["tfidf_score"]
        .transform(lambda x: x.shift(-1).fillna(x))
    )
    df["margin_tfidf"] = (
        df["tfidf_score"] - second_best_tfidf.reindex(df.index)
    ).fillna(0.0).clip(lower=0.0)

    log.info(
        "  Competition features done in %.1fs", time.perf_counter() - t0
    )
    return df


# ===========================================================================
# D. Transitivity / cluster coherence features
# ===========================================================================

def compute_transitivity_features(
    df: pd.DataFrame,
    s2_ids: Set[str],
    s3_ids: Set[str],
    max_pairs: int = 500,
) -> pd.DataFrame:
    """
    Compute cluster-coherence (transitivity) features for every candidate pair.

    For each S1 entity, its retrieved candidates form a cluster.  If this is
    a real entity cluster, we expect:
    - The S2 members to be similar to each other.
    - The S3 members to be similar to each other.
    - Cross-source (S2 ↔ S3) pairs to also be similar.

    Features computed per S1 entity (then merged back to pair level):

    - ``trans_n_candidates``    total number of candidates for this S1.
    - ``trans_mean_sem``        mean sem_score of all other pairs in the cluster.
    - ``trans_std_sem``         std of sem_scores in the cluster.
    - ``trans_mean_jw``         mean JW score of all other pairs in cluster
                                (computed from already-computed jw column).
    - ``trans_cross_source_sim`` mean pairwise JW between the *names* of
                                S2-side candidates and S3-side candidates
                                (capped at ``max_pairs`` cross pairs).

    Parameters
    ----------
    df        : candidate DataFrame with sem_score and jaro_winkler_name columns.
    s2_ids    : set of entity IDs belonging to Source 2.
    s3_ids    : set of entity IDs belonging to Source 3.
    max_pairs : cap on the number of cross-source pairs sampled per S1 to keep
                runtime manageable for very large clusters.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with transitivity feature columns appended.
    """
    log.info("  Computing transitivity features …")
    t0 = time.perf_counter()

    # ── Per-S1 aggregate stats ─────────────────────────────────
    grp = df.groupby("source1_entity_id")

    agg = grp.agg(
        trans_n_candidates=("candidate_entity_id", "count"),
        trans_mean_sem=("sem_score", "mean"),
        trans_std_sem=("sem_score", "std"),
        trans_mean_jw=("jaro_winkler_name", "mean"),
    ).reset_index()
    agg["trans_std_sem"] = agg["trans_std_sem"].fillna(0.0)

    # ── Cross-source pairwise JW ───────────────────────────────
    # Build a quick name lookup for candidates
    cand_name_map: Dict[str, str] = (
        df[["candidate_entity_id", "cand_norm_name"]]
        .drop_duplicates("candidate_entity_id")
        .set_index("candidate_entity_id")["cand_norm_name"]
        .to_dict()
    )

    cross_sim_records = []
    rng = np.random.default_rng(seed=42)

    for s1_id, group in grp:
        cand_ids = group["candidate_entity_id"].tolist()

        s2_names = [
            cand_name_map.get(cid, "")
            for cid in cand_ids if cid in s2_ids
        ]
        s3_names = [
            cand_name_map.get(cid, "")
            for cid in cand_ids if cid in s3_ids
        ]

        if not s2_names or not s3_names:
            cross_sim_records.append(
                {"source1_entity_id": s1_id, "trans_cross_source_sim": 0.0}
            )
            continue

        # Build all cross pairs, sample if too many
        pairs = [(n2, n3) for n2 in s2_names for n3 in s3_names]
        if len(pairs) > max_pairs:
            idxs = rng.choice(len(pairs), size=max_pairs, replace=False)
            pairs = [pairs[i] for i in idxs]

        sims = [jaro_winkler(n2, n3) for n2, n3 in pairs]
        cross_sim_records.append({
            "source1_entity_id":      s1_id,
            "trans_cross_source_sim": float(np.mean(sims)),
        })

    cross_sim_df = pd.DataFrame(cross_sim_records)

    # ── Merge back ─────────────────────────────────────────────
    result = df.merge(agg,           on="source1_entity_id", how="left")
    result = result.merge(cross_sim_df, on="source1_entity_id", how="left")

    # Fill any NaN that crept in from the merge
    trans_cols = [
        "trans_n_candidates", "trans_mean_sem", "trans_std_sem",
        "trans_mean_jw", "trans_cross_source_sim",
    ]
    result[trans_cols] = result[trans_cols].fillna(0.0)

    log.info(
        "  Transitivity features done in %.1fs", time.perf_counter() - t0
    )
    return result


# ===========================================================================
# E. Label attachment (training mode)
# ===========================================================================

def attach_labels(
    df: pd.DataFrame,
    ground_truth_path: str,
) -> pd.DataFrame:
    """
    Attach binary match labels to the candidate pair DataFrame.

    For training, the ground truth maps each S1 entity to its correct S2/S3
    matches.  Pairs in the candidate set that appear in the ground truth get
    ``label = 1``; all others get ``label = 0``.

    Parameters
    ----------
    df                 : candidate pair DataFrame with source1_entity_id,
                         candidate_entity_id columns.
    ground_truth_path  : path to the ground-truth TSV.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with a ``label`` column (int, 0 or 1).
    """
    log.info("  Attaching labels from %s …", ground_truth_path)
    gt = pd.read_csv(ground_truth_path, sep="\t", dtype=str,
                     encoding="utf-8", encoding_errors="ignore")
    gt.columns = [c.strip() for c in gt.columns]

    # Build a set of true-positive (s1_id, cand_id) tuples
    true_pairs: Set[Tuple[str, str]] = set()
    for _, row in gt.iterrows():
        s1_id    = str(row["source1_entity_id"]).strip()
        raw_ids  = str(row["matched_ids"]).strip()
        if not raw_ids or raw_ids.lower() in ("nan", "none", ""):
            continue
        for mid in raw_ids.split(","):
            mid = mid.strip()
            if mid:
                true_pairs.add((s1_id, mid))

    log.info("  Total true pairs in GT: %d", len(true_pairs))

    df = df.copy()
    df["label"] = df.apply(
        lambda r: int(
            (str(r["source1_entity_id"]), str(r["candidate_entity_id"]))
            in true_pairs
        ),
        axis=1,
    )

    pos = df["label"].sum()
    neg = len(df) - pos
    log.info(
        "  Labels attached: %d positives, %d negatives (ratio 1:%.1f)",
        pos, neg, neg / pos if pos else float("inf"),
    )
    return df


# ===========================================================================
# Main feature computation function
# ===========================================================================

def compute_all_features(
    candidates: pd.DataFrame,
    s1:         pd.DataFrame,
    s2:         pd.DataFrame,
    s3:         pd.DataFrame,
    cfg:        dict,
) -> pd.DataFrame:
    """
    Compute the full feature matrix for all candidate pairs.

    Parameters
    ----------
    candidates : output of Phase 3 blocker — columns:
                 [source1_entity_id, candidate_entity_id,
                  tfidf_score, sem_score, max_score]
    s1         : preprocessed Source 1 DataFrame.
    s2         : preprocessed Source 2 DataFrame.
    s3         : preprocessed Source 3 DataFrame.
    cfg        : full config dict.

    Returns
    -------
    pd.DataFrame
        One row per candidate pair with all feature columns.
    """
    feat_cfg   = cfg["features"]
    num_cfg    = feat_cfg["number_veto"]
    jw_p       = feat_cfg["jaro_winkler_prefix_weight"]
    trans_max  = feat_cfg["transitivity_features"]["max_pairs"]

    pin_pattern    = re.compile(num_cfg["pin_regex"],        re.IGNORECASE)
    street_pattern = re.compile(num_cfg["street_num_regex"], re.IGNORECASE)

    # ── Build fast lookup dicts for S1, S2+S3 ─────────────────
    s2s3 = pd.concat([s2, s3], ignore_index=True).drop_duplicates("entity_id")
    s2_id_set: Set[str] = set(s2["entity_id"].astype(str))
    s3_id_set: Set[str] = set(s3["entity_id"].astype(str))

    def _build_lookup(df: pd.DataFrame) -> Dict[str, dict]:
        return {
            str(r["entity_id"]): {
                "norm_name":    r.get("norm_name", ""),
                "norm_address": r.get("norm_address", ""),
                "norm_country": r.get("norm_country", ""),
                "search_text":  r.get("search_text", ""),
                "pin_code":     r.get("pin_code", ""),
                "street_number":r.get("street_number", ""),
            }
            for _, r in df.iterrows()
        }

    log.info("  Building lookup tables …")
    s1_lookup   = _build_lookup(s1)
    s2s3_lookup = _build_lookup(s2s3)

    # ── Row-wise feature computation ───────────────────────────
    log.info("  Computing row-level features for %d pairs …", len(candidates))
    t0 = time.perf_counter()

    records = []
    for row in tqdm(candidates.itertuples(index=False),
                    total=len(candidates), desc="features", unit="pair"):
        s1_id   = str(row.source1_entity_id)
        cand_id = str(row.candidate_entity_id)

        s1_rec   = s1_lookup.get(s1_id,   {})
        cand_rec = s2s3_lookup.get(cand_id, {})

        s1_name   = s1_rec.get("norm_name",    "")
        cand_name = cand_rec.get("norm_name",   "")
        s1_addr   = s1_rec.get("norm_address", "")
        cand_addr = cand_rec.get("norm_address","")
        s1_srch   = s1_rec.get("search_text",  "")
        cand_srch = cand_rec.get("search_text", "")
        s1_cntry  = s1_rec.get("norm_country", "")
        cand_cntry= cand_rec.get("norm_country","")

        # ── A. Base similarities ─────────────────────────────
        jw_name  = jaro_winkler(s1_name,  cand_name,  prefix_weight=jw_p)
        jw_addr  = jaro_winkler(s1_addr,  cand_addr,  prefix_weight=jw_p)
        lev_name = levenshtein_norm(s1_name,  cand_name)
        lev_addr = levenshtein_norm(s1_addr,  cand_addr)
        lev_srch = levenshtein_norm(s1_srch,  cand_srch)
        jac_name = jaccard_tokens(s1_name,  cand_name)
        jac_addr = jaccard_tokens(s1_addr,  cand_addr)
        jac_srch = jaccard_tokens(s1_srch,  cand_srch)

        # ── B. Number Veto ──────────────────────────────────
        pin_c, st_c, veto = number_veto_flags(
            s1_addr,  cand_addr,
            s1_rec.get("pin_code", ""),      cand_rec.get("pin_code", ""),
            s1_rec.get("street_number", ""), cand_rec.get("street_number", ""),
            pin_pattern, street_pattern,
        )

        # ── E. Meta / surface ───────────────────────────────
        ctk    = common_token_count(s1_name, cand_name)
        lr_nm  = len_ratio(s1_name,  cand_name)
        lr_ad  = len_ratio(s1_addr,  cand_addr)
        cntry  = int(s1_cntry == cand_cntry and bool(s1_cntry))

        records.append({
            "source1_entity_id":    s1_id,
            "candidate_entity_id":  cand_id,
            # blocking scores (pass-through + used as features)
            "tfidf_score":          getattr(row, "tfidf_score", 0.0),
            "sem_score":            getattr(row, "sem_score",   0.0),
            # A. base similarities
            "jaro_winkler_name":    jw_name,
            "jaro_winkler_addr":    jw_addr,
            "lev_norm_name":        lev_name,
            "lev_norm_addr":        lev_addr,
            "lev_norm_search":      lev_srch,
            "jaccard_name":         jac_name,
            "jaccard_addr":         jac_addr,
            "jaccard_search":       jac_srch,
            # B. Number Veto
            "pin_conflict":         pin_c,
            "street_conflict":      st_c,
            "number_veto":          veto,
            # E. meta
            "common_token_count":   ctk,
            "name_len_ratio":       lr_nm,
            "addr_len_ratio":       lr_ad,
            "country_match":        cntry,
            # store candidate name for transitivity lookup
            "cand_norm_name":       cand_name,
        })

    feat_df = pd.DataFrame(records)
    log.info("  Row features done in %.1fs", time.perf_counter() - t0)

    # ── C. Competition features ────────────────────────────────
    if feat_cfg["competition_features"]["enabled"]:
        feat_df = compute_competition_features(feat_df)

    # ── D. Transitivity features ───────────────────────────────
    if feat_cfg["transitivity_features"]["enabled"]:
        feat_df = compute_transitivity_features(
            feat_df, s2_id_set, s3_id_set, max_pairs=trans_max
        )

    # Drop the helper column used only for transitivity
    if "cand_norm_name" in feat_df.columns:
        feat_df = feat_df.drop(columns=["cand_norm_name"])

    return feat_df


# ===========================================================================
# Feature list (used by model.py)
# ===========================================================================

FEATURE_COLUMNS: List[str] = [
    # A. base similarities
    "tfidf_score",
    "sem_score",
    "jaro_winkler_name",
    "jaro_winkler_addr",
    "lev_norm_name",
    "lev_norm_addr",
    "lev_norm_search",
    "jaccard_name",
    "jaccard_addr",
    "jaccard_search",
    # B. Number Veto
    "pin_conflict",
    "street_conflict",
    "number_veto",
    # C. Competition
    "margin_sem",
    "margin_tfidf",
    "rank_among_s1",
    "n_competing_s1",
    # D. Transitivity
    "trans_n_candidates",
    "trans_mean_sem",
    "trans_std_sem",
    "trans_mean_jw",
    "trans_cross_source_sim",
    # E. Meta
    "common_token_count",
    "name_len_ratio",
    "addr_len_ratio",
    "country_match",
]


# ===========================================================================
# Main pipeline entry point
# ===========================================================================

def run(config_path: str, attach_label: bool = True) -> None:
    """
    Execute the full Phase 4 feature engineering pipeline.

    Parameters
    ----------
    config_path   : path to ``configs/config.yaml``.
    attach_label  : if ``True`` (training mode), load ground truth and add
                    a ``label`` column.  Set ``False`` for inference.
    """
    log.info("=" * 60)
    log.info("Phase 4 · Feature Engineering")
    log.info("=" * 60)

    cfg       = load_config(config_path)
    paths     = cfg["paths"]
    cache_dir = paths["cache_dir"]
    out_path  = Path(cache_dir) / "features.parquet"

    # ── 1. Load inputs ────────────────────────────────────────
    log.info("Step 1/4 · Loading preprocessed frames + candidates")
    s1, s2, s3 = load_preprocessed(cache_dir)
    candidates  = load_candidates(cache_dir)

    log.info(
        "  S1=%d  S2=%d  S3=%d  Candidate pairs=%d",
        len(s1), len(s2), len(s3), len(candidates),
    )

    # ── 2. Compute features ───────────────────────────────────
    log.info("Step 2/4 · Computing features")
    feat_df = compute_all_features(candidates, s1, s2, s3, cfg)

    # ── 3. Attach labels (training mode) ─────────────────────
    if attach_label:
        gt_path = paths.get("ground_truth", "")
        if gt_path and Path(gt_path).exists():
            log.info("Step 3/4 · Attaching ground-truth labels")
            feat_df = attach_labels(feat_df, gt_path)
        else:
            log.warning(
                "Ground truth not found at %s — skipping label attachment.", gt_path
            )
            feat_df["label"] = np.nan
    else:
        log.info("Step 3/4 · Inference mode — no labels attached")
        feat_df["label"] = np.nan

    # ── 4. Validate & save ────────────────────────────────────
    log.info("Step 4/4 · Validating and saving feature table")
    _validate_features(feat_df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_parquet(str(out_path), engine="pyarrow",
                       compression="snappy", index=False)

    log.info("Saved feature table → %s  (%d rows × %d cols)",
             out_path, len(feat_df), len(feat_df.columns))

    # ── Summary stats ─────────────────────────────────────────
    if "label" in feat_df.columns and feat_df["label"].notna().any():
        pos = int(feat_df["label"].sum())
        neg = len(feat_df) - pos
        log.info(
            "Label distribution: %d positive, %d negative (1:%.1f)",
            pos, neg, neg / pos if pos else float("inf"),
        )

    log.info("=" * 60)
    log.info("Phase 4 complete.")
    log.info("  Feature table → %s", out_path)
    log.info("=" * 60)


def _validate_features(df: pd.DataFrame) -> None:
    """
    Sanity-check the feature DataFrame before saving.

    Checks
    ------
    - All FEATURE_COLUMNS are present (some may be missing if disabled in cfg).
    - No column is entirely NaN.
    - No Inf values in numeric columns.
    """
    present_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]

    if missing_features:
        log.warning("Missing feature columns (may be disabled): %s", missing_features)

    num_df = df[present_features].select_dtypes(include=[np.number])

    all_nan_cols = [c for c in num_df.columns if num_df[c].isna().all()]
    if all_nan_cols:
        log.warning("Columns that are entirely NaN: %s", all_nan_cols)

    inf_cols = [c for c in num_df.columns
                if np.isinf(num_df[c].values).any()]
    if inf_cols:
        log.warning("Inf values detected in: %s — replacing with 0", inf_cols)
        df[inf_cols] = df[inf_cols].replace([np.inf, -np.inf], 0.0)

    log.info(
        "  Validation: %d feature cols present, %d missing, %d inf-cols fixed",
        len(present_features), len(missing_features), len(inf_cols),
    )


# ===========================================================================
# Public API (used by model.py)
# ===========================================================================

def load_features(cache_dir: str) -> pd.DataFrame:
    """
    Load the feature table produced by Phase 4.

    Returns
    -------
    pd.DataFrame
        Full feature table (one row per candidate pair).

    Raises
    ------
    FileNotFoundError
        If ``features.parquet`` does not exist in *cache_dir*.
    """
    path = Path(cache_dir) / "features.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Feature table not found: {path}\n"
            "Run 'python -m src.features --config configs/config.yaml' first."
        )
    df = pd.read_parquet(path)
    log.info("Loaded features.parquet: %d rows × %d cols", len(df), len(df.columns))
    return df


# ===========================================================================
# CLI entry point
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4 — Graph & Competition Feature Engineering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--no-labels", dest="attach_label", action="store_false",
        help="Skip label attachment (inference mode).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(config_path=args.config, attach_label=args.attach_label)
