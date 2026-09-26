"""
src/blocker.py
==============
Phase 3 — Dual Blocking Layer

Responsibilities
----------------
1. Blocker A (TF-IDF)   : Character n-gram TF-IDF vectorisation of
   ``search_text`` + scikit-learn NearestNeighbors (cosine distance).
   Strengths: fast, handles abbreviations well, deterministic.

2. Blocker B (Semantic)  : Sentence embeddings via
   ``paraphrase-multilingual-MiniLM-L12-v2`` + FAISS Flat index (inner
   product on L2-normalised vectors = cosine similarity).
   Strengths: multilingual zero-shot, captures paraphrase-level similarity,
   robust to French (unseen in training).

3. Merge               : Union of both candidate sets per S1 entity,
   ranked by ``max(tfidf_score, semantic_score)``, hard-capped at
   ``blocking.top_n_per_entity`` (default 50).

4. Output              : ``output/candidate_pairs.tsv``
   Format  (tab-separated, header row):
       source1_entity_id \\t candidate_entity_ids
   where ``candidate_entity_ids`` is a comma-separated list.
   Singletons (S1 entities with zero candidates) get an empty string.

Also persists:
- ``output/cache/faiss_index_s2s3.bin``  — FAISS index for S2+S3
- ``output/cache/embeddings_s1.npy``     — S1 MiniLM embeddings
- ``output/cache/embeddings_s2s3.npy``   — S2+S3 MiniLM embeddings
- ``output/cache/candidates_raw.parquet`` — merged candidate table
  (source1_entity_id, candidate_entity_id, tfidf_score, sem_score, max_score)

CLI usage
---------
    python -m src.blocker --config configs/config.yaml
    python -m src.blocker --config configs/config.yaml --no-cache   # force re-encode
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd
import yaml
from scipy.sparse import csr_matrix
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

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
# Blocker A — TF-IDF character n-gram + NearestNeighbors
# ===========================================================================

class TFIDFBlocker:
    """
    Sparse TF-IDF character n-gram blocker.

    Fits a TF-IDF vectoriser on the combined corpus (S1 + S2 + S3) so that
    the IDF weights reflect the full token distribution, then uses a
    ball-tree / brute-force cosine NearestNeighbors to retrieve the top-k
    S2/S3 candidates for every S1 record.

    Parameters
    ----------
    analyzer    : ``'char_wb'`` (default) uses padded character n-grams,
                  which handles word boundaries better than ``'char'``.
    ngram_range : tuple of (min_n, max_n) for character n-grams.
    max_features: vocabulary size cap (speeds up fitting on large corpora).
    top_k       : candidates to retrieve per S1 entity.
    """

    def __init__(
        self,
        analyzer:     str        = "char_wb",
        ngram_range:  Tuple[int,int] = (2, 4),
        max_features: int        = 200_000,
        top_k:        int        = 30,
    ) -> None:
        self.top_k = top_k
        self._vectoriser = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=True,          # log(1+tf) dampens very frequent tokens
            strip_accents=None,         # already handled in preprocessor
            lowercase=False,            # already lowercase
        )
        self._nn: Optional[NearestNeighbors] = None
        self._s2s3_ids: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def fit(self, s1_texts: List[str], s2s3_texts: List[str]) -> None:
        """
        Fit the vectoriser on the joint corpus, then index the S2+S3 matrix.

        The NearestNeighbors index is built on S2/S3 vectors only; S1 vectors
        are used as queries at search time.
        """
        log.info("  [TF-IDF] Fitting vectoriser on %d + %d = %d texts …",
                 len(s1_texts), len(s2s3_texts), len(s1_texts) + len(s2s3_texts))
        t0 = time.perf_counter()
        all_texts = s1_texts + s2s3_texts
        self._vectoriser.fit(all_texts)

        log.info("  [TF-IDF] Vocabulary size: %d", len(self._vectoriser.vocabulary_))

        log.info("  [TF-IDF] Transforming S2+S3 corpus …")
        X_s2s3: csr_matrix = self._vectoriser.transform(s2s3_texts)

        log.info("  [TF-IDF] Building NearestNeighbors index (metric=cosine) …")
        # 'brute' is fastest for sparse matrices; sklearn uses optimised BLAS
        self._nn = NearestNeighbors(
            n_neighbors=min(self.top_k, X_s2s3.shape[0]),
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        )
        self._nn.fit(X_s2s3)
        log.info("  [TF-IDF] Index built in %.1fs", time.perf_counter() - t0)

    # ------------------------------------------------------------------
    def query(
        self,
        s1_texts: List[str],
        s2s3_ids: List[str],
    ) -> pd.DataFrame:
        """
        Retrieve top-k S2/S3 neighbours for every S1 text.

        Returns
        -------
        pd.DataFrame with columns:
            ``s1_idx``  — positional index into *s1_texts*
            ``cand_id`` — S2/S3 entity_id of the candidate
            ``tfidf_score`` — cosine similarity (1 - distance)
        """
        assert self._nn is not None, "Call fit() before query()."
        log.info("  [TF-IDF] Querying %d S1 records …", len(s1_texts))
        t0 = time.perf_counter()

        X_s1: csr_matrix = self._vectoriser.transform(s1_texts)
        distances, indices = self._nn.kneighbors(X_s1)  # shape: (n_s1, top_k)

        # distances are cosine distances → convert to similarities
        similarities = 1.0 - distances

        s2s3_id_arr = np.array(s2s3_ids)
        rows = []
        for s1_idx in range(len(s1_texts)):
            for rank, (idx, sim) in enumerate(
                zip(indices[s1_idx], similarities[s1_idx])
            ):
                if sim <= 0.0:
                    continue  # skip zero-similarity pairs
                rows.append({
                    "s1_idx":       s1_idx,
                    "cand_id":      s2s3_id_arr[idx],
                    "tfidf_score":  float(sim),
                })

        log.info("  [TF-IDF] Query done in %.1fs — %d raw candidates",
                 time.perf_counter() - t0, len(rows))
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["s1_idx", "cand_id", "tfidf_score"]
        )


# ===========================================================================
# Blocker B — Sentence-Transformer + FAISS
# ===========================================================================

class SemanticBlocker:
    """
    Multilingual dense-vector semantic blocker.

    Uses ``paraphrase-multilingual-MiniLM-L12-v2`` (Apache 2.0, 117M params)
    to encode ``search_text`` into 384-dimensional embeddings, then performs
    approximate nearest-neighbour search via FAISS.

    Key design choices
    ------------------
    - Embeddings are L2-normalised before indexing; inner product then equals
      cosine similarity (avoids a separate cosine FAISS index type).
    - A FAISS ``IndexFlatIP`` is used by default for exact search.  For very
      large datasets (>500k records) the config can switch to ``IVF`` for
      sub-linear search at a small recall cost.
    - Embeddings are cached as ``.npy`` files; re-runs skip the expensive
      encode step unless ``--no-cache`` is passed.
    """

    def __init__(
        self,
        model_name:           str  = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        batch_size:           int  = 256,
        normalize_embeddings: bool = True,
        faiss_index_type:     str  = "Flat",
        top_k:                int  = 30,
        cache_dir:            str  = "output/cache",
        seed:                 int  = 42,
    ) -> None:
        self.model_name           = model_name
        self.batch_size           = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.faiss_index_type     = faiss_index_type
        self.top_k                = top_k
        self.cache_dir            = Path(cache_dir)
        self.seed                 = seed

        self._model:      Optional[SentenceTransformer] = None
        self._index:      Optional[faiss.Index]         = None
        self._s2s3_ids:   Optional[np.ndarray]          = None
        self._embed_dim:  int                            = 384  # MiniLM-L12-v2

    # ------------------------------------------------------------------
    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            log.info("  [Semantic] Loading model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    # ------------------------------------------------------------------
    def _encode(
        self,
        texts: List[str],
        cache_path: Optional[Path] = None,
        use_cache:  bool = True,
    ) -> np.ndarray:
        """
        Encode a list of texts to L2-normalised float32 embeddings.

        If *cache_path* is provided and the file exists (and *use_cache* is
        True), the cached array is loaded instead of re-encoding.
        """
        if use_cache and cache_path is not None and cache_path.exists():
            log.info("  [Semantic] Loading cached embeddings from %s", cache_path)
            emb = np.load(str(cache_path))
            log.info("  [Semantic] Loaded embeddings shape: %s", emb.shape)
            return emb

        model = self._load_model()
        log.info("  [Semantic] Encoding %d texts (batch_size=%d) …",
                 len(texts), self.batch_size)
        t0 = time.perf_counter()

        emb = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        emb = emb.astype(np.float32)
        log.info("  [Semantic] Encoded in %.1fs — shape %s",
                 time.perf_counter() - t0, emb.shape)

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(cache_path), emb)
            log.info("  [Semantic] Saved embeddings → %s", cache_path)

        return emb

    # ------------------------------------------------------------------
    def _build_faiss_index(self, emb_s2s3: np.ndarray) -> faiss.Index:
        """
        Build (or load) the FAISS index over S2+S3 embeddings.

        For ``IndexFlatIP`` exact search is used.  The index is persisted to
        disk and reloaded on subsequent runs.
        """
        dim = emb_s2s3.shape[1]
        index_path = self.cache_dir / "faiss_index_s2s3.bin"

        if index_path.exists():
            log.info("  [Semantic] Loading FAISS index from %s", index_path)
            index = faiss.read_index(str(index_path))
            log.info("  [Semantic] Index loaded — %d vectors", index.ntotal)
            return index

        log.info("  [Semantic] Building FAISS %s index (dim=%d, n=%d) …",
                 self.faiss_index_type, dim, emb_s2s3.shape[0])
        t0 = time.perf_counter()

        if self.faiss_index_type == "Flat":
            # Exact inner-product search on normalised embeddings = cosine
            index = faiss.IndexFlatIP(dim)
        elif self.faiss_index_type == "IVF":
            nlist = min(4096, max(64, emb_s2s3.shape[0] // 100))
            quantiser = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantiser, dim, nlist,
                                       faiss.METRIC_INNER_PRODUCT)
            index.train(emb_s2s3)
            index.nprobe = min(64, nlist)
        else:
            raise ValueError(
                f"Unknown faiss_index_type: {self.faiss_index_type!r}. "
                "Choose 'Flat' or 'IVF'."
            )

        index.add(emb_s2s3)  # type: ignore[attr-defined]
        log.info("  [Semantic] Index built in %.1fs", time.perf_counter() - t0)

        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_path))
        log.info("  [Semantic] Saved FAISS index → %s", index_path)
        return index

    # ------------------------------------------------------------------
    def fit(
        self,
        s2s3_texts: List[str],
        s2s3_ids:   List[str],
        use_cache:  bool = True,
    ) -> None:
        """
        Encode S2+S3 texts and build the FAISS index.

        Parameters
        ----------
        s2s3_texts : list of preprocessed ``search_text`` strings for S2+S3.
        s2s3_ids   : corresponding entity IDs (same order as *s2s3_texts*).
        use_cache  : load cached embeddings and index if available.
        """
        self._s2s3_ids = np.array(s2s3_ids)

        emb_s2s3 = self._encode(
            s2s3_texts,
            cache_path=self.cache_dir / "embeddings_s2s3.npy",
            use_cache=use_cache,
        )
        self._embed_dim = emb_s2s3.shape[1]

        # Invalidate stale FAISS index if embeddings were freshly computed
        index_path = self.cache_dir / "faiss_index_s2s3.bin"
        if not use_cache and index_path.exists():
            index_path.unlink()
            log.info("  [Semantic] Removed stale FAISS index.")

        self._index = self._build_faiss_index(emb_s2s3)

    # ------------------------------------------------------------------
    def query(
        self,
        s1_texts: List[str],
        use_cache: bool = True,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Retrieve top-k S2/S3 neighbours for every S1 text.

        Returns
        -------
        Tuple of:
        - pd.DataFrame with columns:
            ``s1_idx``, ``cand_id``, ``sem_score``
        - np.ndarray of shape (n_s1, embed_dim) — S1 embeddings (reused
          downstream for feature computation without re-encoding).
        """
        assert self._index is not None, "Call fit() before query()."
        assert self._s2s3_ids is not None

        emb_s1 = self._encode(
            s1_texts,
            cache_path=self.cache_dir / "embeddings_s1.npy",
            use_cache=use_cache,
        )

        log.info("  [Semantic] FAISS search: %d queries, top-%d …",
                 len(s1_texts), self.top_k)
        t0 = time.perf_counter()

        k = min(self.top_k, self._index.ntotal)
        scores, indices = self._index.search(emb_s1, k)  # type: ignore[attr-defined]

        log.info("  [Semantic] Search done in %.1fs", time.perf_counter() - t0)

        rows = []
        for s1_idx in range(len(s1_texts)):
            for rank in range(k):
                idx  = indices[s1_idx, rank]
                sim  = float(scores[s1_idx, rank])
                if idx < 0 or sim <= 0.0:
                    continue
                rows.append({
                    "s1_idx":    s1_idx,
                    "cand_id":   self._s2s3_ids[idx],
                    "sem_score": sim,
                })

        df = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["s1_idx", "cand_id", "sem_score"]
        )
        log.info("  [Semantic] %d raw semantic candidates", len(df))
        return df, emb_s1


# ===========================================================================
# Candidate merging
# ===========================================================================

def merge_candidates(
    tfidf_df:    pd.DataFrame,
    semantic_df: pd.DataFrame,
    s1_ids:      List[str],
    top_n:       int,
) -> pd.DataFrame:
    """
    Union TF-IDF and semantic candidates, dedup, rank, and cap at *top_n*.

    Ranking key: ``max(tfidf_score, sem_score)`` — keeps the strongest
    signal from either blocker.

    Parameters
    ----------
    tfidf_df    : output of ``TFIDFBlocker.query()``
                  columns: [s1_idx, cand_id, tfidf_score]
    semantic_df : output of ``SemanticBlocker.query()``
                  columns: [s1_idx, cand_id, sem_score]
    s1_ids      : list of S1 entity IDs (index matches s1_idx).
    top_n       : hard cap on candidates per S1 entity.

    Returns
    -------
    pd.DataFrame with columns:
        source1_entity_id, candidate_entity_id,
        tfidf_score, sem_score, max_score
    """
    log.info("Merging TF-IDF (%d) + semantic (%d) candidates …",
             len(tfidf_df), len(semantic_df))

    # ── Align on (s1_idx, cand_id) with an outer join ───────
    merged = pd.merge(
        tfidf_df.rename(columns={"cand_id": "candidate_entity_id"}),
        semantic_df.rename(columns={"cand_id": "candidate_entity_id"}),
        on=["s1_idx", "candidate_entity_id"],
        how="outer",
    )

    # Fill missing scores with 0 (the blocker simply didn't find this pair)
    merged["tfidf_score"] = merged["tfidf_score"].fillna(0.0)
    merged["sem_score"]   = merged["sem_score"].fillna(0.0)

    # Ranking score
    merged["max_score"] = merged[["tfidf_score", "sem_score"]].max(axis=1)

    # Map s1_idx → source1_entity_id
    s1_id_arr = np.array(s1_ids)
    merged["source1_entity_id"] = s1_id_arr[merged["s1_idx"].astype(int)]

    # ── Per-entity sort + cap ─────────────────────────────────
    merged = merged.sort_values(
        ["source1_entity_id", "max_score"], ascending=[True, False]
    )
    merged = (
        merged
        .groupby("source1_entity_id", sort=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    # Drop the integer index column
    merged = merged.drop(columns=["s1_idx"])

    log.info("After merge & cap(top_%d): %d candidate pairs across %d S1 entities",
             top_n, len(merged), merged["source1_entity_id"].nunique())
    return merged


# ===========================================================================
# Output writers
# ===========================================================================

def write_candidate_pairs_tsv(
    candidates: pd.DataFrame,
    s1_ids:     List[str],
    output_path: str,
) -> None:
    """
    Write ``candidate_pairs.tsv`` in the required submission format.

    Format (tab-separated, UTF-8, with header):
        source1_entity_id \\t candidate_entity_ids

    - ``candidate_entity_ids`` is a comma-separated list of S2/S3 IDs.
    - Singletons (S1 entities with zero candidates) get an empty string.
    - All S1 entities appear exactly once (even if they have no candidates).

    Parameters
    ----------
    candidates  : merged candidate DataFrame (from ``merge_candidates``).
    s1_ids      : complete list of S1 entity IDs — ensures all S1 entities
                  appear in the output, including singletons.
    output_path : destination file path.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Group candidates into comma-separated lists
    grouped = (
        candidates
        .groupby("source1_entity_id")["candidate_entity_id"]
        .apply(lambda ids: ",".join(ids.astype(str).tolist()))
        .reset_index()
        .rename(columns={"candidate_entity_id": "candidate_entity_ids"})
    )

    # Ensure every S1 entity is represented (singletons → empty string)
    all_s1 = pd.DataFrame({"source1_entity_id": s1_ids})
    output_df = all_s1.merge(grouped, on="source1_entity_id", how="left")
    output_df["candidate_entity_ids"] = (
        output_df["candidate_entity_ids"].fillna("")
    )

    output_df.to_csv(
        output_path, sep="\t", index=False, encoding="utf-8"
    )

    total_candidates = (output_df["candidate_entity_ids"] != "").sum()
    singleton_count  = (output_df["candidate_entity_ids"] == "").sum()
    log.info(
        "Saved candidate_pairs.tsv → %s  |  entities_with_candidates=%d  singletons=%d",
        output_path, total_candidates, singleton_count,
    )


def save_candidates_parquet(candidates: pd.DataFrame, cache_dir: str) -> None:
    """Persist the full merged candidate table (with scores) for Phase 4."""
    path = Path(cache_dir) / "candidates_raw.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(str(path), engine="pyarrow",
                          compression="snappy", index=False)
    log.info("Saved candidates_raw.parquet → %s  (%d rows)", path, len(candidates))


# ===========================================================================
# Blocking recall evaluation (training-time only)
# ===========================================================================

def compute_blocking_recall(
    candidates: pd.DataFrame,
    ground_truth_path: str,
) -> float:
    """
    Compute pair-level blocking recall against the ground truth.

    This is a diagnostic metric only — it measures what fraction of the
    true positive pairs were retrieved by the blocker.  At test time, the
    ground truth is unavailable.

    Parameters
    ----------
    candidates        : merged candidate DataFrame.
    ground_truth_path : path to the ground-truth TSV.

    Returns
    -------
    float
        Blocking recall in [0, 1].
    """
    gt = pd.read_csv(ground_truth_path, sep="\t", dtype=str)
    gt.columns = [c.strip() for c in gt.columns]

    total_positives = 0
    retrieved       = 0

    cand_set: Dict = (
        candidates
        .groupby("source1_entity_id")["candidate_entity_id"]
        .apply(set)
        .to_dict()
    )

    for _, row in gt.iterrows():
        s1_id      = str(row["source1_entity_id"]).strip()
        raw_ids    = str(row["matched_ids"]).strip()
        if not raw_ids or raw_ids.lower() in ("nan", "none", ""):
            continue
        true_ids   = {m.strip() for m in raw_ids.split(",") if m.strip()}
        retrieved_ids = cand_set.get(s1_id, set())

        total_positives += len(true_ids)
        retrieved        += len(true_ids & retrieved_ids)

    recall = retrieved / total_positives if total_positives else 1.0
    log.info(
        "Blocking recall: %d / %d = %.4f (%.2f%%)",
        retrieved, total_positives, recall, recall * 100,
    )
    return recall


# ===========================================================================
# Main pipeline entry point
# ===========================================================================

def run(config_path: str, use_cache: bool = True) -> None:
    """
    Execute the full Phase 3 blocking pipeline.

    Parameters
    ----------
    config_path : path to ``configs/config.yaml``.
    use_cache   : if ``True`` (default), reuse cached embeddings and FAISS
                  index.  Pass ``False`` (``--no-cache`` CLI flag) to force
                  re-encoding — necessary after preprocessing changes.
    """
    log.info("=" * 60)
    log.info("Phase 3 · Blocking")
    log.info("=" * 60)

    cfg         = load_config(config_path)
    paths       = cfg["paths"]
    block_cfg   = cfg["blocking"]
    tfidf_cfg   = block_cfg["tfidf"]
    sem_cfg     = block_cfg["semantic"]
    cache_dir   = paths["cache_dir"]
    output_dir  = paths["output_dir"]
    top_n       = block_cfg["top_n_per_entity"]
    seed        = cfg.get("seed", 42)

    # ── 1. Load preprocessed DataFrames ──────────────────────
    log.info("Step 1/6 · Loading preprocessed source files")
    s1, s2, s3 = load_preprocessed(cache_dir)

    # Combine S2 + S3 as the candidate pool
    s2s3 = pd.concat([s2, s3], ignore_index=True)
    # Defensive: drop any duplicate entity_ids (shouldn't happen, but guard it)
    s2s3 = s2s3.drop_duplicates(subset="entity_id").reset_index(drop=True)

    s1_ids   : List[str] = s1["entity_id"].tolist()
    s2s3_ids : List[str] = s2s3["entity_id"].tolist()
    s1_texts  = s1["search_text"].tolist()
    s2s3_texts = s2s3["search_text"].tolist()

    log.info("  S1=%d  |  S2+S3=%d  |  top_n=%d", len(s1), len(s2s3), top_n)

    # ── 2. Blocker A: TF-IDF ─────────────────────────────────
    log.info("Step 2/6 · Blocker A — TF-IDF character n-gram")
    tfidf_blocker = TFIDFBlocker(
        analyzer    = tfidf_cfg["analyzer"],
        ngram_range = tuple(tfidf_cfg["ngram_range"]),  # type: ignore[arg-type]
        max_features= tfidf_cfg["max_features"],
        top_k       = tfidf_cfg["top_k"],
    )
    tfidf_blocker.fit(s1_texts, s2s3_texts)
    tfidf_candidates = tfidf_blocker.query(s1_texts, s2s3_ids)

    # ── 3. Blocker B: Semantic + FAISS ───────────────────────
    log.info("Step 3/6 · Blocker B — Semantic (MiniLM-L12-v2) + FAISS")
    sem_blocker = SemanticBlocker(
        model_name           = sem_cfg["model_name"],
        batch_size           = sem_cfg["batch_size"],
        normalize_embeddings = sem_cfg["normalize_embeddings"],
        faiss_index_type     = sem_cfg["faiss_index_type"],
        top_k                = sem_cfg["top_k"],
        cache_dir            = cache_dir,
        seed                 = seed,
    )
    sem_blocker.fit(s2s3_texts, s2s3_ids, use_cache=use_cache)
    sem_candidates, _ = sem_blocker.query(s1_texts, use_cache=use_cache)

    # ── 4. Merge candidates ───────────────────────────────────
    log.info("Step 4/6 · Merging candidates from both blockers")
    merged_candidates = merge_candidates(
        tfidf_df    = tfidf_candidates,
        semantic_df = sem_candidates,
        s1_ids      = s1_ids,
        top_n       = top_n,
    )

    # ── 5. Save outputs ───────────────────────────────────────
    log.info("Step 5/6 · Saving outputs")
    save_candidates_parquet(merged_candidates, cache_dir)
    write_candidate_pairs_tsv(
        candidates  = merged_candidates,
        s1_ids      = s1_ids,
        output_path = paths["candidate_pairs"],
    )

    # ── 6. Evaluate blocking recall (if GT available) ─────────
    log.info("Step 6/6 · Evaluating blocking recall")
    gt_path = paths.get("ground_truth", "")
    if gt_path and Path(gt_path).exists():
        recall = compute_blocking_recall(merged_candidates, gt_path)
        if recall < 0.95:
            log.warning(
                "Blocking recall %.2f%% is below 95%%. "
                "Consider increasing top_n_per_entity or top_k values.",
                recall * 100,
            )
    else:
        log.info("  Ground truth not found — skipping recall evaluation.")

    log.info("=" * 60)
    log.info("Phase 3 complete.")
    log.info("  candidate_pairs.tsv  → %s", paths["candidate_pairs"])
    log.info("  candidates_raw       → %s/candidates_raw.parquet", cache_dir)
    log.info("=" * 60)


# ===========================================================================
# Public API (used by downstream modules)
# ===========================================================================

def load_candidates(cache_dir: str) -> pd.DataFrame:
    """
    Load the raw merged candidate table produced by Phase 3.

    Used by Phase 4 (feature engineering) so it can iterate over candidate
    pairs without re-running the blocking step.

    Returns
    -------
    pd.DataFrame with columns:
        source1_entity_id, candidate_entity_id,
        tfidf_score, sem_score, max_score
    """
    path = Path(cache_dir) / "candidates_raw.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Candidate table not found: {path}\n"
            "Run 'python -m src.blocker --config configs/config.yaml' first."
        )
    df = pd.read_parquet(path)
    log.info("Loaded candidates_raw: %d rows", len(df))
    return df


# ===========================================================================
# CLI entry point
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3 — Dual Blocking Layer (TF-IDF + Semantic + FAISS)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--no-cache", dest="use_cache", action="store_false",
        help="Force re-encoding of embeddings and rebuilding of FAISS index.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(config_path=args.config, use_cache=args.use_cache)
