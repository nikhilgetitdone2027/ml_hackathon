# TODO
**Project:** Amazon ML Hackathon 2026 — Business Entity Resolution
**Last Updated:** 2026-09-26

Legend: ✅ Done | 🔲 Pending | 🔄 In Progress | ❌ Blocked

---

## Phase 1 — Repository Setup ✅

- [x] Create directory structure (`data/`, `src/`, `configs/`, `output/`, `scripts/`, `notebooks/`)
- [x] Write `requirements.txt` with pinned versions and license annotations
- [x] Write `configs/config.yaml` — all hyperparameters for all phases
- [x] Write `README.md` — full docs with architecture ASCII diagram
- [x] Write `scripts/run_pipeline.sh` — bash orchestrator with `set -euo pipefail`
- [x] Write `src/__init__.py`
- [x] Write stub files for phases 2–7 (raise `NotImplementedError`)
- [x] Write `.gitignore`

---

## Phase 2 — Preprocessing ✅

- [x] `basic_normalise()` — NFD → strip diacritics → lowercase → de-punct
- [x] `strip_accents()` — Unicode decomposition
- [x] `_collect_substitution_pairs()` — mine abbreviations from GT aligned pairs
- [x] `apply_token_replacements()` — whole-token-only expansion
- [x] `extract_pin_code()` — regex postal code extraction
- [x] `extract_street_number()` — regex street number extraction
- [x] `add_number_columns()` — vectorised application to DataFrame
- [x] `preprocess_dataframe()` — full pipeline on a DataFrame
- [x] `run()` — CLI entry point with `--no-mine` flag
- [x] `load_preprocessed()` — public API for downstream modules
- [x] `_validate_preprocessed()` — assertion checks
- [x] Output: 3 Parquet files in `output/cache/`
- [x] Output: `output/token_replacement_dict.json`
- [x] AST parse verified ✅

---

## Phase 3 — Blocking Layer ✅

- [x] `TFIDFBlocker` class — char n-gram TF-IDF + NearestNeighbors (cosine)
  - [x] `fit()` — joint-corpus vectoriser + S2/S3 index
  - [x] `query()` — top-k candidates per S1
- [x] `SemanticBlocker` class — MiniLM-L12-v2 + FAISS
  - [x] `_encode()` — with `.npy` embedding cache
  - [x] `_build_faiss_index()` — Flat or IVF, persisted to `.bin`
  - [x] `fit()` — encode S2+S3, build FAISS index
  - [x] `query()` — FAISS search, returns DataFrame + S1 embeddings
- [x] `merge_candidates()` — union, dedup, rank by max score, cap at top-N
- [x] `write_candidate_pairs_tsv()` — submission-format output
- [x] `save_candidates_parquet()` — scored table for Phase 4
- [x] `compute_blocking_recall()` — training-time GT diagnostic
- [x] `run()` — full pipeline with `--no-cache` flag
- [x] `load_candidates()` — public API
- [x] AST parse verified ✅

---

## Phase 4 — Feature Engineering ✅

- [x] **Group A — Base similarities** (9 features)
  - [x] `jaro_winkler()` — name & address
  - [x] `levenshtein_norm()` — name, address, search_text
  - [x] `jaccard_tokens()` — name, address, search_text
- [x] **Group B — Number Veto** (3 features)
  - [x] `number_veto_flags()` — pin_conflict, street_conflict, number_veto
- [x] **Group C — Competition features** (4 features)
  - [x] `compute_competition_features()` — margin_sem, margin_tfidf, rank, n_competing
- [x] **Group D — Transitivity features** (5 features)
  - [x] `compute_transitivity_features()` — mean/std sem, mean JW, n_candidates, cross-source sim
- [x] **Group E — Meta/surface features** (5 features)
  - [x] `common_token_count()`, `len_ratio()`, country_match, tfidf_score passthrough
- [x] `FEATURE_COLUMNS` list — 26 features, canonical order
- [x] `attach_labels()` — GT label attachment for training
- [x] `compute_all_features()` — orchestrator with lookup dicts
- [x] `_validate_features()` — NaN/Inf checks
- [x] `run()` — with `--no-labels` flag
- [x] `load_features()` — public API
- [x] AST parse verified ✅

---

## Phase 5 — Modelling & Calibration ✅

- [x] `fbeta_score_binary()` — F-beta for binary arrays
- [x] `macro_f05_at_threshold()` — macro-averaged F0.5 per S1 entity
- [x] `best_threshold_f05()` — grid search for best threshold
- [x] `ClosedUniverseCV` class
  - [x] `split()` — GroupKFold on S1 IDs, full S2/S3 negative pool
- [x] `build_xgb_params()` — config → XGBoost param dict
- [x] `train_xgb_fold()` — single fold: DMatrix, early stopping, OOF probs
- [x] `fit_isotonic_calibrator()` — OOF Isotonic Regression
- [x] `aggregate_feature_importance()` — avg gain/weight/cover across K folds
- [x] `train()` — full CV + calibration + final retrain loop
- [x] `_save_model_artefacts()` — JSON model, joblib calibrator, CSVs
- [x] `load_model_artefacts()` — public load function
- [x] `predict()` — inference + clipping + parquet save
- [x] `load_predictions()` — public API for Phase 6
- [x] `run()` — with `--inference-only` flag
- [x] AST parse verified ✅

---

## Phase 6 — Post-Processing & F0.5 Optimisation ✅

- [x] `resolve_constraints()` — S2/S3 → at most ONE S1 (keep max-prob edge, tie-break by S1 ID)
- [x] `expected_f05()` — E[F-beta] for a top-k subset via Bernoulli formula
- [x] `optimise_per_entity()` — exhaustive scan k=0..N, pick argmax expected F0.5
- [x] `optimise_all_entities()` — orchestrator; logs k-distribution and singleton count
- [x] `write_matching_results()` — strict TSV format, dedup, empty strings for singletons
- [x] `save_post_constraint_predictions()` — audit parquet for Phase 7
- [x] `validate_matching_results()` — inline 4-check validation (IDs in candidates, no dups, constraint holds, all S1 present)
- [x] `evaluate_with_ground_truth()` — true macro F0.5 when GT available
- [x] `load_matching_results()` — public API for Phase 7
- [x] `run()` — CLI with `--beta` and `--skip-constraint` flags
- [x] AST parse verified ✅

---

## Phase 7 — Output Generation & Validation 🔲

- [ ] `write_matching_results()` — strict TSV format, empty lists handled
- [ ] `validate_outputs()` — assertion: all IDs in results exist in candidates
- [ ] `validate_no_duplicates()` — no duplicate IDs per S1 row
- [ ] `validate_all_s1_present()` — every S1 entity has exactly one row
- [ ] End-to-end smoke test on synthetic 10-entity dataset
- [ ] `run()` — `--validate` CLI flag

---

## Post-Implementation (After Phase 7)

- [ ] Install dependencies and run end-to-end on actual data
- [ ] Tune `blocking.top_n_per_entity` based on observed recall
- [ ] Tune `model.xgboost.scale_pos_weight` based on actual class ratio
- [ ] Grid-search `blocking.tfidf.top_k` and `blocking.semantic.top_k`
- [ ] Ablation study: which feature groups contribute most to F0.5
- [ ] Cross-validate F0.5 with French test-set simulation
- [ ] Pin dependency hashes in `requirements.lock`
