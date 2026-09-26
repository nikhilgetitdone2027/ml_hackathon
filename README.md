# Amazon ML Hackathon 2026 — Business Entity Resolution

> **Graph-Aware Hybrid Dual-Encoder Pipeline**
> Match deduplicated reference business records (Source 1) to noisy fragments in Source 2 and Source 3.

---

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Architecture Overview](#architecture-overview)
3. [Repository Structure](#repository-structure)
4. [Setup & Installation](#setup--installation)
5. [Data Preparation](#data-preparation)
6. [Running the Pipeline](#running-the-pipeline)
7. [Phase-by-Phase Description](#phase-by-phase-description)
8. [Evaluation Metric](#evaluation-metric)
9. [Hard Constraints & Compliance](#hard-constraints--compliance)
10. [Configuration Reference](#configuration-reference)
11. [Reproducibility](#reproducibility)

---

## Problem Statement

Given three TSV files sharing columns `entity_id | business_name | business_address | country`:

| Source | Role |
|--------|------|
| **Source 1 (S1)** | Deduplicated **reference** records — each entity is unique. |
| **Source 2 (S2)** | Noisy / fragmented records that **may** map to an S1 entity. |
| **Source 3 (S3)** | Additional noisy records (different data vendor) — same task. |

**Goal:** Produce, for every S1 entity, the list of S2/S3 IDs that refer to the same real-world business.

**Key constraint:** An S2 or S3 record can belong to **at most one** S1 entity (S1 is deduplicated).
Entities with no matches must output an **empty list** (contributes a perfect 1.0 to macro F0.5).

---

## Architecture Overview

```
+------------------------------------------------------------------+
|  Phase 1 . PREPROCESSING                                         |
|  Unicode normalise -> mine abbreviation dict -> extract numbers   |
+------------------------+-----------------------------------------+
                         |
+------------------------v-----------------------------------------+
|  Phase 2 . BLOCKING  (recall target >= 98%)                      |
|  +----------------------+   +--------------------------------+   |
|  | Blocker A            |   | Blocker B                      |   |
|  | TF-IDF char n-gram   |   | MiniLM-L12-v2 + FAISS          |   |
|  | + NearestNeighbors   |   | semantic ANN search            |   |
|  +----------+-----------+   +----------------+---------------+   |
|             +--------------+-----------------+                   |
|                      Union & top-50                              |
|              -> candidate_pairs.tsv                              |
+------------------------+-----------------------------------------+
                         |
+------------------------v-----------------------------------------+
|  Phase 3 . FEATURE ENGINEERING                                   |
|  Jaro-Winkler . Levenshtein . Jaccard . MiniLM cosine            |
|  Number Veto . Competition margin . Transitivity cluster score   |
+------------------------+-----------------------------------------+
                         |
+------------------------v-----------------------------------------+
|  Phase 4 . MODELLING                                             |
|  XGBoost (binary:logistic) with closed-universe group-CV         |
|  + Isotonic Regression probability calibration (OOF)             |
+------------------------+-----------------------------------------+
                         |
+------------------------v-----------------------------------------+
|  Phase 5 . POST-PROCESSING                                       |
|  Constraint resolution (S2/S3 -> max-1 S1)                      |
|  Expected F0.5 maximisation per S1 (no static threshold)         |
|              -> matching_results.tsv                             |
+------------------------------------------------------------------+
```

---

## Repository Structure

```
amazon-ml-2026/
├── configs/
│   └── config.yaml            # Master configuration (all hyperparams)
├── data/
│   ├── source1.tsv            # Reference records  (place here)
│   ├── source2.tsv            # Noisy records S2   (place here)
│   ├── source3.tsv            # Noisy records S3   (place here)
│   └── ground_truth.tsv       # Training labels    (place here)
├── notebooks/
│   └── eda.ipynb              # Exploratory analysis (optional)
├── output/
│   ├── cache/                 # Embeddings & index cache
│   ├── models/                # Saved XGBoost + calibrator
│   ├── candidate_pairs.tsv    # Blocking output
│   └── matching_results.tsv   # Final predictions
├── scripts/
│   └── run_pipeline.sh        # End-to-end orchestration script
├── src/
│   ├── __init__.py
│   ├── preprocessor.py        # Phase 2: text normalisation + dict mining
│   ├── blocker.py             # Phase 3: TF-IDF + FAISS dual blocking
│   ├── features.py            # Phase 4: pair feature computation
│   ├── model.py               # Phase 5: XGBoost + calibration
│   ├── optimizer.py           # Phase 6: constraint resolution + F0.5 opt
│   └── utils.py               # Phase 7: I/O, validation, assertion checks
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python >= 3.10
- (Optional) CUDA 11.8+ for GPU-accelerated FAISS / PyTorch

### 1. Clone / copy project
```bash
git clone <repo-url>
cd amazon-ml-2026
```

### 2. Create a virtual environment
```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** The first run will download `paraphrase-multilingual-MiniLM-L12-v2` (~120 MB)
> from Hugging Face. Subsequent runs use the local cache.

---

## Data Preparation

Place the four TSV files (tab-separated, UTF-8, with header row) into `data/`:

| File | Columns | Notes |
|------|---------|-------|
| `source1.tsv` | `entity_id`, `business_name`, `business_address`, `country` | Reference; no duplicates |
| `source2.tsv` | same | May contain noisy / abbreviated text |
| `source3.tsv` | same | Different vendor; different noise patterns |
| `ground_truth.tsv` | `source1_entity_id`, `matched_ids` | `matched_ids` is comma-separated S2/S3 IDs |

All file paths are configurable in `configs/config.yaml`.

---

## Running the Pipeline

### Full pipeline (recommended)
```bash
bash scripts/run_pipeline.sh
```

### Individual stages
```bash
# Phase 2: Preprocess
python -m src.preprocessor --config configs/config.yaml

# Phase 3: Blocking -> candidate_pairs.tsv
python -m src.blocker --config configs/config.yaml

# Phase 4: Feature engineering
python -m src.features --config configs/config.yaml

# Phase 5: Train / infer model
python -m src.model --config configs/config.yaml

# Phase 6: Post-process -> matching_results.tsv
python -m src.optimizer --config configs/config.yaml

# Phase 7: Validate outputs
python -m src.utils --validate --config configs/config.yaml
```

---

## Phase-by-Phase Description

### Phase 2 — Preprocessing (`src/preprocessor.py`)
- **Unicode normalisation**: NFD decomposition -> strip combining diacritics -> lowercase.
- **Punctuation removal**: replaces all non-alphanumeric characters with a single space.
- **Abbreviation mining**: aligns matched S1-S2/S3 pairs token-by-token to discover
  high-frequency substitutions (e.g. `Corp -> Corporation`, `St -> Street`).
  Stored as `output/token_replacement_dict.json`.
- **Number extraction**: separates `pin_code` and `street_number` into dedicated columns
  via configurable regex patterns, enabling the downstream **Number Veto** feature.

### Phase 3 — Blocking (`src/blocker.py`)
- **Blocker A (TF-IDF)**: Vectorises concatenated `business_name + business_address` with
  character `(2,4)`-gram TF-IDF. Retrieves top-30 S2/S3 candidates per S1 using cosine NearestNeighbors.
- **Blocker B (Semantic)**: Encodes with `paraphrase-multilingual-MiniLM-L12-v2`; retrieves
  top-30 via FAISS Flat (L2 on normalised embeddings = cosine).
  Multilingual model -> zero-shot French generalisation.
- **Merge**: Union, deduped, capped at **top-50** per S1 entity (by max-of-two similarity scores).

### Phase 4 — Feature Engineering (`src/features.py`)

| Feature | Description |
|---------|-------------|
| `jaro_winkler` | Jaro-Winkler on normalised business name |
| `levenshtein_norm` | Normalised edit distance (0-1) |
| `jaccard_token` | Jaccard on token sets of name+address |
| `minilm_cosine` | Pre-computed cosine from FAISS (reused) |
| `number_veto` | 1 if PIN/street numbers conflict, 0 otherwise |
| `margin_s2s3` | Current S1 score minus next-best S1 score for same S2/S3 ID |
| `transitivity_score` | Mean cross-source similarity among S1's cluster candidates |

### Phase 5 — Modelling (`src/model.py`)
- **Closed-Universe CV**: `GroupKFold` on `source1_entity_id`, but FP candidates are drawn
  from the *full* S2/S3 universe (not only the fold's S1 subset), matching true test-set density.
- **XGBoost**: `binary:logistic`, 800 trees, `scale_pos_weight` tuned for class imbalance.
- **Calibration**: Isotonic regression fitted OOF, then applied to test predictions.

### Phase 6 — Post-Processing (`src/optimizer.py`)
- **Constraint resolution**: For any S2/S3 ID claimed by >1 S1 entities, keep only the
  highest-probability edge. Ties broken by lexicographic `entity_id` order.
- **Expected F0.5 maximisation**: For each S1, iterate top-k subsets (k = 0 ... N).
  Compute Expected-F0.5 using calibrated probabilities as Bernoulli ground-truth estimates.
  Select the k that maximises the expectation.

### Phase 7 — Output Validation (`src/utils.py`)
- Assertion: every ID in `matching_results.tsv` must appear in `candidate_pairs.tsv`.
- Assertion: no duplicate IDs within a single S1's matched list.
- Assertion: every S1 entity has exactly one row in `matching_results.tsv`.

---

## Evaluation Metric

**Macro-averaged F0.5** across all S1 entities.

```
F_0.5 = (1 + 0.5^2) * P * R  /  (0.5^2 * P + R)
```

- beta = 0.5 -> **precision weighted 2x over recall**.
- Singletons (no true matches) contribute **1.0** if the predicted list is empty.
- Score is **macro-averaged** — each S1 entity contributes equally regardless of cluster size.

---

## Hard Constraints & Compliance

| Constraint | Implementation |
|------------|---------------|
| No external data / APIs | All computation is local; no geocoding, no web requests |
| Models <= 8B params, MIT/Apache 2.0 | `MiniLM-L12-v2` (117M params, Apache 2.0) |
| Zero-shot France generalisation | Multilingual sentence encoder; no country-specific rules |
| No OCR / Vision models | Pure text; no image inputs anywhere |

---

## Configuration Reference

All tunable parameters live in `configs/config.yaml`. Key knobs:

| Key | Default | Effect |
|-----|---------|--------|
| `blocking.top_n_per_entity` | 50 | Hard cap on candidates per S1 |
| `blocking.tfidf.top_k` | 30 | TF-IDF candidates before merge |
| `blocking.semantic.top_k` | 30 | FAISS candidates before merge |
| `model.xgboost.n_estimators` | 800 | More trees -> better fit, slower training |
| `model.xgboost.scale_pos_weight` | 10 | Increase if positive pairs are very rare |
| `postprocessing.f05_optimization.beta` | 0.5 | Tune precision/recall balance |

---

## Reproducibility

- Global `seed: 42` passed to all random states (NumPy, XGBoost, FAISS IVF, scikit-learn).
- FAISS index and embedding cache saved to `output/cache/` — re-runs skip re-encoding.
- XGBoost model and calibrator saved to `output/models/` in JSON + joblib formats.
- Pin exact dependency versions for full hash reproducibility:
  ```bash
  pip-compile requirements.txt --generate-hashes -o requirements.lock
  pip install --require-hashes -r requirements.lock
  ```
