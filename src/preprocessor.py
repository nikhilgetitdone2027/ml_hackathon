"""
src/preprocessor.py
===================
Phase 2 — Text Preprocessing & Abbreviation Mining

Responsibilities
----------------
1. Unicode normalisation   : NFD → strip diacritics → lowercase.
2. Punctuation removal     : collapse non-alphanumeric characters to spaces.
3. Token-replacement mining: align matched S1↔S2/S3 pairs token-by-token to
   discover high-frequency substitution pairs (e.g. "corp" → "corporation"),
   then apply them to every record.
4. Number extraction       : pull `pin_code` and `street_number` into
   dedicated columns via configurable regex, used downstream for Number Veto.

Outputs
-------
- Preprocessed DataFrames stored as Parquet in output/cache/:
    preprocessed_source1.parquet
    preprocessed_source2.parquet
    preprocessed_source3.parquet
- Token-replacement dictionary: output/token_replacement_dict.json

CLI usage
---------
    python -m src.preprocessor --config configs/config.yaml
    python -m src.preprocessor --config configs/config.yaml --no-mine   # skip mining
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    """Load and return the YAML configuration file as a plain dict."""
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


# ===========================================================================
# Core text normalisation
# ===========================================================================

# Compiled once at module load for performance
_RE_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_RE_MULTI_SPACE = re.compile(r"\s+")


def strip_accents(text: str) -> str:
    """
    Unicode NFD decomposition → remove combining diacritical marks.

    Works correctly on French (é → e, ç → c, etc.) and all other
    Latin-script languages without requiring a lookup table.

    Examples
    --------
    >>> strip_accents("Société Générale")
    'Societe Generale'
    >>> strip_accents("MÜNCHEN")
    'MUNCHEN'
    """
    nfd = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")


def basic_normalise(text: str) -> str:
    """
    Apply the full normalisation pipeline to a single string:
      1. Coerce to str (handles NaN / None gracefully).
      2. Lowercase.
      3. Strip diacritical accents.
      4. Replace all non-alphanumeric characters with a space.
      5. Collapse multiple spaces and strip leading/trailing whitespace.

    Parameters
    ----------
    text : str | None | float
        Raw field value.

    Returns
    -------
    str
        Normalised string, always a str (never NaN).

    Examples
    --------
    >>> basic_normalise("Acme Corp., Ltd.")
    'acme corp ltd'
    >>> basic_normalise("Société Anonyme S.A.")
    'societe anonyme sa'
    >>> basic_normalise(None)
    ''
    """
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)  # type: ignore[arg-type]
    text = text.lower()
    text = strip_accents(text)
    text = _RE_NON_ALNUM.sub(" ", text)
    text = _RE_MULTI_SPACE.sub(" ", text).strip()
    return text


# ===========================================================================
# Number extraction
# ===========================================================================

def _compile_patterns(cfg: dict) -> Tuple[re.Pattern, re.Pattern]:
    """
    Compile the PIN and street-number regex patterns from config.

    Both patterns are compiled with IGNORECASE so they match
    upper- and lower-case letter suffixes (e.g. "12B" or "12b").
    """
    pin_pat    = re.compile(cfg["features"]["number_veto"]["pin_regex"],
                            re.IGNORECASE)
    street_pat = re.compile(cfg["features"]["number_veto"]["street_num_regex"],
                            re.IGNORECASE)
    return pin_pat, pin_pat  # kept symmetric; caller unpacks differently


def extract_pin_code(address: str, pin_pattern: re.Pattern) -> str:
    """
    Extract the first match of the PIN/postal-code pattern from an address.

    Returns an empty string when nothing is found so the Number Veto
    logic can safely compare two strings.

    Parameters
    ----------
    address     : normalised address string.
    pin_pattern : compiled regex for 4–7 digit codes.

    Examples
    --------
    >>> import re
    >>> pat = re.compile(r'\\b\\d{4,7}\\b')
    >>> extract_pin_code("12 main street 94105 san francisco", pat)
    '94105'
    >>> extract_pin_code("no code here", pat)
    ''
    """
    m = pin_pattern.search(address)
    return m.group(0) if m else ""


def extract_street_number(address: str, street_pattern: re.Pattern) -> str:
    """
    Extract the first street-number token from an address string.

    Street numbers appear early in an address and are 1-4 digits optionally
    followed by a single letter (e.g. "42", "7B", "123a").

    Parameters
    ----------
    address        : normalised address string.
    street_pattern : compiled regex for street numbers.

    Examples
    --------
    >>> import re
    >>> pat = re.compile(r'\\b\\d{1,4}[a-z]?\\b')
    >>> extract_street_number("42b baker street london", pat)
    '42b'
    """
    m = street_pattern.search(address)
    return m.group(0) if m else ""


def add_number_columns(df: pd.DataFrame,
                       pin_pattern: re.Pattern,
                       street_pattern: re.Pattern,
                       address_col: str = "norm_address") -> pd.DataFrame:
    """
    Vectorised application of number-extraction functions.

    Adds two columns to *df* in-place and returns the frame:
    - ``pin_code``      : postal / ZIP code string (or "").
    - ``street_number`` : street number string (or "").

    Parameters
    ----------
    df             : DataFrame that already contains *address_col*.
    pin_pattern    : compiled PIN regex.
    street_pattern : compiled street-number regex.
    address_col    : name of the normalised address column to search in.
    """
    df["pin_code"]      = df[address_col].apply(
        lambda x: extract_pin_code(x, pin_pattern)
    )
    df["street_number"] = df[address_col].apply(
        lambda x: extract_street_number(x, street_pattern)
    )
    return df


# ===========================================================================
# Abbreviation / token-replacement mining
# ===========================================================================

def _tokenise(text: str) -> List[str]:
    """Split a normalised string on whitespace, returning a list of tokens."""
    return text.split()


def _collect_substitution_pairs(
    s1_df: pd.DataFrame,
    other_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    min_freq: int,
    top_k: int,
) -> Dict[str, str]:
    """
    Mine token-level substitution pairs from ground-truth matched record pairs.

    Algorithm
    ---------
    1. For each (S1_id, matched_id) pair in ground truth:
         a. Look up the normalised name tokens for the S1 record.
         b. Look up the normalised name tokens for the matched S2/S3 record.
         c. For every S1-token that does *not* appear verbatim in the S2/S3
            token set, and vice-versa, record (s2_token → s1_token) as a
            candidate substitution — meaning "s2_token is an abbreviation of
            s1_token".
    2. Count all candidate pairs across the entire training set.
    3. Keep pairs where:
         - frequency >= min_freq
         - the s2 token is strictly shorter than the s1 token (abbreviation
           direction only; avoids circular replacements)
         - the s2 token is a prefix of the s1 token (high-confidence signal)
    4. Return the top-k pairs by frequency as a {abbreviated: canonical} dict.

    Parameters
    ----------
    s1_df    : preprocessed Source 1 DataFrame (must have ``entity_id``,
               ``norm_name`` columns).
    other_df : combined preprocessed Source 2 + 3 DataFrame (same columns).
    gt_df    : ground-truth DataFrame with columns ``source1_entity_id`` and
               ``matched_ids`` (comma-separated IDs).
    min_freq : minimum pair frequency to retain.
    top_k    : maximum number of substitution pairs to keep.

    Returns
    -------
    dict
        ``{abbreviated_token: canonical_token}`` mapping, ready for expansion.

    Notes
    -----
    - Addresses are intentionally excluded from mining because street-number
      tokens and postcode tokens would pollute the substitution dictionary.
    - The direction is always (noisy → canonical), so we replace noisy tokens
      with their canonical S1 equivalents before encoding / similarity scoring.
    """
    # Build lookup dicts: id → token list
    s1_lookup: Dict[str, List[str]] = {
        row.entity_id: _tokenise(row.norm_name)
        for row in s1_df[["entity_id", "norm_name"]].itertuples(index=False)
    }
    other_lookup: Dict[str, List[str]] = {
        row.entity_id: _tokenise(row.norm_name)
        for row in other_df[["entity_id", "norm_name"]].itertuples(index=False)
    }

    pair_counter: Counter = Counter()

    for _, row in gt_df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        raw_matches = str(row["matched_ids"]).strip()

        if not raw_matches or raw_matches.lower() in ("nan", "none", ""):
            continue  # singleton — no matches

        matched_ids = [m.strip() for m in raw_matches.split(",") if m.strip()]
        s1_tokens_set = set(s1_lookup.get(s1_id, []))

        for mid in matched_ids:
            other_tokens = other_lookup.get(mid, [])
            other_tokens_set = set(other_tokens)

            # Tokens that appear only on the noisy side (candidates for expansion)
            noisy_only = other_tokens_set - s1_tokens_set
            # Tokens that appear only on the canonical side
            canonical_only = s1_tokens_set - other_tokens_set

            # For each (noisy, canonical) pair where noisy looks like an abbreviation
            for noisy_tok in noisy_only:
                for canon_tok in canonical_only:
                    if (
                        len(noisy_tok) >= 2                     # ignore single chars
                        and len(noisy_tok) < len(canon_tok)     # noisy is shorter
                        and canon_tok.startswith(noisy_tok)     # prefix match
                    ):
                        pair_counter[(noisy_tok, canon_tok)] += 1

    # Filter by minimum frequency and select top-k
    filtered = {
        abbr: canon
        for (abbr, canon), freq in pair_counter.most_common(top_k)
        if freq >= min_freq
    }
    log.info("  Mined %d token-substitution pairs (min_freq=%d, top_k=%d)",
             len(filtered), min_freq, top_k)
    return filtered


def apply_token_replacements(text: str,
                              replacement_dict: Dict[str, str]) -> str:
    """
    Replace abbreviated tokens in *text* with their canonical equivalents.

    Replacement is performed left-to-right on whitespace-split tokens.
    Only exact whole-token matches are replaced (no substring replacement),
    preventing accidental corruption of street names.

    Parameters
    ----------
    text             : normalised string to expand.
    replacement_dict : ``{abbreviated: canonical}`` mapping.

    Returns
    -------
    str
        String with abbreviated tokens replaced.

    Examples
    --------
    >>> apply_token_replacements("acme corp ltd", {"corp": "corporation", "ltd": "limited"})
    'acme corporation limited'
    """
    if not replacement_dict:
        return text
    tokens = text.split()
    return " ".join(replacement_dict.get(tok, tok) for tok in tokens)


# ===========================================================================
# DataFrame-level preprocessing
# ===========================================================================

TEXT_COLS = ["business_name", "business_address"]


def preprocess_dataframe(
    df: pd.DataFrame,
    replacement_dict: Optional[Dict[str, str]] = None,
    pin_pattern: Optional[re.Pattern] = None,
    street_pattern: Optional[re.Pattern] = None,
) -> pd.DataFrame:
    """
    Apply the full preprocessing pipeline to a source DataFrame.

    Steps
    -----
    1. Make a copy (non-destructive).
    2. Fill NaN in text columns with "".
    3. Normalise ``business_name``  → ``norm_name``.
    4. Normalise ``business_address`` → ``norm_address``.
    5. Apply token replacements to ``norm_name`` (if dict provided).
    6. Normalise ``country`` → ``norm_country``.
    7. Build ``search_text`` = ``norm_name`` + " " + ``norm_address``
       (the field that gets vectorised by the blockers).
    8. Extract ``pin_code`` and ``street_number`` (if patterns provided).

    Parameters
    ----------
    df               : raw DataFrame with columns as in the TSV spec.
    replacement_dict : mined abbreviation dict to apply.  Pass ``None`` or
                       ``{}`` to skip expansion.
    pin_pattern      : compiled PIN regex.  ``None`` → skip extraction.
    street_pattern   : compiled street-number regex.  ``None`` → skip.

    Returns
    -------
    pd.DataFrame
        Original columns preserved; new columns added.
    """
    df = df.copy()

    # Coerce text columns to str, fill blanks
    for col in TEXT_COLS + ["country"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    # ── Normalise ────────────────────────────────────────────
    df["norm_name"]    = df["business_name"].apply(basic_normalise)
    df["norm_address"] = df["business_address"].apply(basic_normalise)
    df["norm_country"] = df["country"].apply(basic_normalise)

    # ── Token expansion (abbreviation replacement) ───────────
    if replacement_dict:
        df["norm_name"] = df["norm_name"].apply(
            lambda t: apply_token_replacements(t, replacement_dict)
        )

    # ── Combined search text (used by both blockers) ─────────
    df["search_text"] = df["norm_name"] + " " + df["norm_address"]
    df["search_text"] = df["search_text"].str.strip()

    # ── Number extraction ────────────────────────────────────
    if pin_pattern is not None and street_pattern is not None:
        df = add_number_columns(df, pin_pattern, street_pattern)
    else:
        df["pin_code"]      = ""
        df["street_number"] = ""

    # Ensure entity_id is always a string for consistent joins
    df["entity_id"] = df["entity_id"].astype(str).str.strip()

    return df


# ===========================================================================
# I/O helpers
# ===========================================================================

def read_tsv(path: str) -> pd.DataFrame:
    """
    Read a TSV file into a DataFrame.

    Handles:
    - BOM (``encoding_errors='ignore'``).
    - Leading/trailing whitespace in column names.
    - Duplicate column names (raises early with a clear message).
    """
    log.info("  Reading %s", path)
    df = pd.read_csv(path, sep="\t", dtype=str,
                     encoding="utf-8", encoding_errors="ignore",
                     low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    required = {"entity_id", "business_name", "business_address", "country"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"File {path!r} is missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )
    log.info("    Loaded %d rows, %d columns", len(df), len(df.columns))
    return df


def read_ground_truth(path: str) -> pd.DataFrame:
    """
    Read the ground-truth TSV.

    Expected columns: ``source1_entity_id``, ``matched_ids``
    where ``matched_ids`` is a comma-separated list of S2/S3 IDs
    (or empty / "nan" for singletons).
    """
    log.info("  Reading ground truth from %s", path)
    gt = pd.read_csv(path, sep="\t", dtype=str,
                     encoding="utf-8", encoding_errors="ignore")
    gt.columns = [c.strip() for c in gt.columns]

    required = {"source1_entity_id", "matched_ids"}
    missing  = required - set(gt.columns)
    if missing:
        raise ValueError(
            f"Ground truth file is missing columns: {missing}\n"
            f"Found: {list(gt.columns)}"
        )
    log.info("    Loaded %d ground-truth rows", len(gt))
    return gt


def save_parquet(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame as Parquet with snappy compression."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
    log.info("  Saved %d rows → %s", len(df), path)


def save_replacement_dict(d: Dict[str, str], path: str) -> None:
    """Serialise the replacement dictionary to JSON (human-readable)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=2, ensure_ascii=False, sort_keys=True)
    log.info("  Saved %d substitution pairs → %s", len(d), path)


def load_replacement_dict(path: str) -> Dict[str, str]:
    """Load a previously mined replacement dictionary from JSON."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ===========================================================================
# Main pipeline entry point
# ===========================================================================

def run(config_path: str, mine: bool = True) -> None:
    """
    Execute the full Phase 2 preprocessing pipeline.

    Parameters
    ----------
    config_path : path to ``configs/config.yaml``.
    mine        : if ``True`` (default), mine abbreviations from ground truth
                  and apply them.  Set ``False`` to skip mining (e.g. when
                  ground truth is unavailable at inference time — the cached
                  dict from training is loaded instead).
    """
    log.info("=" * 60)
    log.info("Phase 2 · Preprocessing")
    log.info("=" * 60)

    cfg = load_config(config_path)
    paths   = cfg["paths"]
    pp_cfg  = cfg["preprocessing"]
    num_cfg = cfg["features"]["number_veto"]

    # Ensure output directories exist
    cache_dir = Path(paths["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir   = Path(paths["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load raw TSVs ─────────────────────────────────────
    log.info("Step 1/5 · Loading raw source files")
    s1_raw = read_tsv(paths["source1"])
    s2_raw = read_tsv(paths["source2"])
    s3_raw = read_tsv(paths["source3"])

    # ── 2. Compile regex patterns ────────────────────────────
    log.info("Step 2/5 · Compiling number-extraction patterns")
    pin_pattern    = re.compile(num_cfg["pin_regex"],    re.IGNORECASE)
    street_pattern = re.compile(num_cfg["street_num_regex"], re.IGNORECASE)

    # ── 3. Mine / load token-replacement dictionary ──────────
    dict_path = str(out_dir / "token_replacement_dict.json")

    if mine:
        if not Path(paths["ground_truth"]).exists():
            log.warning(
                "Ground truth file not found at %s — skipping abbreviation mining.",
                paths["ground_truth"],
            )
            replacement_dict: Dict[str, str] = {}
        else:
            log.info("Step 3/5 · Mining abbreviation dictionary from ground truth")
            gt_df = read_ground_truth(paths["ground_truth"])

            # We need partially-normalised (but not yet expanded) names for mining,
            # so we do a quick normalise-only pass before the full pipeline.
            s1_norm_quick = s1_raw.copy()
            s1_norm_quick["entity_id"] = s1_norm_quick["entity_id"].astype(str).str.strip()
            s1_norm_quick["norm_name"] = s1_norm_quick["business_name"].apply(basic_normalise)

            s2_norm_quick = s2_raw.copy()
            s2_norm_quick["entity_id"] = s2_norm_quick["entity_id"].astype(str).str.strip()
            s2_norm_quick["norm_name"] = s2_norm_quick["business_name"].apply(basic_normalise)

            s3_norm_quick = s3_raw.copy()
            s3_norm_quick["entity_id"] = s3_norm_quick["entity_id"].astype(str).str.strip()
            s3_norm_quick["norm_name"] = s3_norm_quick["business_name"].apply(basic_normalise)

            other_combined = pd.concat(
                [s2_norm_quick, s3_norm_quick], ignore_index=True
            )

            replacement_dict = _collect_substitution_pairs(
                s1_df    = s1_norm_quick,
                other_df = other_combined,
                gt_df    = gt_df,
                min_freq = pp_cfg["min_token_freq"],
                top_k    = pp_cfg["top_k_substitutions"],
            )
            save_replacement_dict(replacement_dict, dict_path)

            # Log the top-20 substitutions for inspection
            preview_n = min(20, len(replacement_dict))
            if preview_n:
                log.info("  Top substitution pairs (abbreviated → canonical):")
                for abbr, canon in list(replacement_dict.items())[:preview_n]:
                    log.info("    %-20s  →  %s", abbr, canon)
    else:
        # Inference mode: load the dict mined during training
        if Path(dict_path).exists():
            log.info("Step 3/5 · Loading pre-mined abbreviation dictionary from %s", dict_path)
            replacement_dict = load_replacement_dict(dict_path)
        else:
            log.warning("No pre-mined dictionary found at %s — skipping expansion.", dict_path)
            replacement_dict = {}

    # ── 4. Full preprocessing pass ───────────────────────────
    log.info("Step 4/5 · Preprocessing all source DataFrames")

    log.info("  Processing Source 1 …")
    s1 = preprocess_dataframe(s1_raw, replacement_dict, pin_pattern, street_pattern)

    log.info("  Processing Source 2 …")
    s2 = preprocess_dataframe(s2_raw, replacement_dict, pin_pattern, street_pattern)

    log.info("  Processing Source 3 …")
    s3 = preprocess_dataframe(s3_raw, replacement_dict, pin_pattern, street_pattern)

    # ── 5. Validate & save ───────────────────────────────────
    log.info("Step 5/5 · Validating and persisting preprocessed frames")

    _validate_preprocessed(s1, label="Source 1")
    _validate_preprocessed(s2, label="Source 2")
    _validate_preprocessed(s3, label="Source 3")

    save_parquet(s1, str(cache_dir / "preprocessed_source1.parquet"))
    save_parquet(s2, str(cache_dir / "preprocessed_source2.parquet"))
    save_parquet(s3, str(cache_dir / "preprocessed_source3.parquet"))

    log.info("=" * 60)
    log.info("Phase 2 complete.")
    log.info("  Replacement dict  : %s", dict_path)
    log.info("  Cache directory   : %s", cache_dir)
    log.info("=" * 60)


def _validate_preprocessed(df: pd.DataFrame, label: str) -> None:
    """
    Sanity-check a preprocessed DataFrame.

    Raises
    ------
    AssertionError
        If any mandatory column is missing, or if ``entity_id`` contains
        duplicates (which would corrupt downstream joins).
    """
    required_cols = {
        "entity_id", "norm_name", "norm_address",
        "norm_country", "search_text", "pin_code", "street_number",
    }
    missing = required_cols - set(df.columns)
    assert not missing, (
        f"[{label}] Preprocessed frame is missing columns: {missing}"
    )

    dup_count = df["entity_id"].duplicated().sum()
    if dup_count:
        log.warning("[%s] %d duplicate entity_ids detected!", label, dup_count)

    blank_search = (df["search_text"].str.strip() == "").sum()
    if blank_search:
        log.warning("[%s] %d rows have an empty search_text!", label, blank_search)

    log.info(
        "  [%s] OK — %d rows | blanks=%d | dup_ids=%d",
        label, len(df), blank_search, dup_count,
    )


# ===========================================================================
# Public API (used by other modules)
# ===========================================================================

def load_preprocessed(cache_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the three cached preprocessed DataFrames.

    Used by downstream modules (blocker, features, model) so they don't
    need to repeat the preprocessing step.

    Parameters
    ----------
    cache_dir : path to the output/cache directory.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (source1, source2, source3) DataFrames.

    Raises
    ------
    FileNotFoundError
        If any of the three Parquet files do not exist.
    """
    base = Path(cache_dir)
    paths = {
        "s1": base / "preprocessed_source1.parquet",
        "s2": base / "preprocessed_source2.parquet",
        "s3": base / "preprocessed_source3.parquet",
    }
    for key, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(
                f"Preprocessed cache not found: {p}\n"
                "Run 'python -m src.preprocessor --config configs/config.yaml' first."
            )
    s1 = pd.read_parquet(paths["s1"])
    s2 = pd.read_parquet(paths["s2"])
    s3 = pd.read_parquet(paths["s3"])
    log.info(
        "Loaded preprocessed frames: S1=%d  S2=%d  S3=%d",
        len(s1), len(s2), len(s3),
    )
    return s1, s2, s3


# ===========================================================================
# CLI entry point
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2 — Preprocessing & Abbreviation Mining",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--no-mine", dest="mine", action="store_false",
        help="Skip abbreviation mining and load the cached dict instead.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(config_path=args.config, mine=args.mine)
