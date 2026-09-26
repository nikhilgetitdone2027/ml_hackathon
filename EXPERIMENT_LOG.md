# EXPERIMENT LOG
**Project:** Amazon ML Hackathon 2026 — Business Entity Resolution
**Last Updated:** 2026-09-26

Format: Experiments are append-only. Each entry documents the hypothesis, setup, results, and conclusion.
Results from actual data runs will be added here as they are executed.

---

## EXP-001 · AST Validation — Phase 2 (preprocessor.py)
**Date:** 2026-09-26
**Type:** Code Validation (no data)
**Status:** PASSED

**Hypothesis:** `src/preprocessor.py` is syntactically correct and exports the expected public API.

**Method:** `ast.parse()` + function name extraction.

**Results:**
```
AST parse: OK
  strip_accents                        FOUND
  basic_normalise                      FOUND
  extract_pin_code                     FOUND
  extract_street_number                FOUND
  add_number_columns                   FOUND
  apply_token_replacements             FOUND
  preprocess_dataframe                 FOUND
  run                                  FOUND
  load_preprocessed                    FOUND
```

**Conclusion:** All 9 required functions present. File is syntactically valid.

---

## EXP-002 · AST Validation — Phase 3 (blocker.py)
**Date:** 2026-09-26
**Type:** Code Validation (no data)
**Status:** PASSED

**Hypothesis:** `src/blocker.py` is syntactically correct and exports the expected public API.

**Method:** `ast.parse()` + class/function name extraction.

**Results:**
```
AST parse: OK
  class TFIDFBlocker - FOUND
  class SemanticBlocker - FOUND
  func  merge_candidates - FOUND
  func  write_candidate_pairs_tsv - FOUND
  func  save_candidates_parquet - FOUND
  func  compute_blocking_recall - FOUND
  func  run - FOUND
  func  load_candidates - FOUND
```

**Conclusion:** Both blocker classes and all 6 module-level functions present.

---

## EXP-003 · AST Validation — Phase 4 (features.py)
**Date:** 2026-09-26
**Type:** Code Validation (no data)
**Status:** PASSED

**Hypothesis:** `src/features.py` is syntactically correct and `FEATURE_COLUMNS` (26 features) is defined.

**Method:** `ast.parse()` + `grep` for `FEATURE_COLUMNS` at line 689.

**Results:**
```
AST parse: OK
  func jaro_winkler - FOUND
  func levenshtein_norm - FOUND
  func jaccard_tokens - FOUND
  func common_token_count - FOUND
  func len_ratio - FOUND
  func number_veto_flags - FOUND
  func compute_competition_features - FOUND
  func compute_transitivity_features - FOUND
  func attach_labels - FOUND
  func compute_all_features - FOUND
  func run - FOUND
  func load_features - FOUND
  func _validate_features - FOUND

FEATURE_COLUMNS at line 689: FOUND (26 features confirmed by grep)
```

**Conclusion:** All 13 functions present; FEATURE_COLUMNS defined with 26 features.

---

## EXP-004 · AST Validation — Phase 5 (model.py)
**Date:** 2026-09-26
**Type:** Code Validation (no data)
**Status:** PASSED

**Hypothesis:** `src/model.py` is syntactically correct and exports the expected public API.

**Method:** `ast.parse()` + class/function name extraction.

**Results:**
```
AST parse: OK

Classes:
  ClosedUniverseCV - FOUND

Functions:
  fbeta_score_binary - FOUND
  macro_f05_at_threshold - FOUND
  best_threshold_f05 - FOUND
  build_xgb_params - FOUND
  train_xgb_fold - FOUND
  fit_isotonic_calibrator - FOUND
  aggregate_feature_importance - FOUND
  train - FOUND
  _save_model_artefacts - FOUND
  load_model_artefacts - FOUND
  predict - FOUND
  load_predictions - FOUND
  run - FOUND
```

**Conclusion:** ClosedUniverseCV class + 12 functions all present.

---

## EXP-005 · AST Validation — Phase 6 (optimizer.py)
**Date:** 2026-09-26
**Type:** Code Validation (no data)
**Status:** PASSED

**Hypothesis:** `src/optimizer.py` is syntactically correct and exports the expected public API.

**Method:** `ast.parse()` + function name extraction.

**Results:**
```
AST parse: OK
  resolve_constraints - FOUND
  expected_f05 - FOUND
  optimise_per_entity - FOUND
  optimise_all_entities - FOUND
  write_matching_results - FOUND
  save_post_constraint_predictions - FOUND
  validate_matching_results - FOUND
  evaluate_with_ground_truth - FOUND
  run - FOUND
  load_matching_results - FOUND
```

**Conclusion:** All 10 required functions present. File is syntactically valid.
Notable: `validate_matching_results()` now lives in Phase 6 (inline) rather
than only in Phase 7 — provides earlier error detection in the pipeline.

---


## EXP-006 . AST Validation -- Phase 7 (utils.py)
**Date:** 2026-09-26
**Type:** Code Validation (no data)
**Status:** PASSED

**Results:**
`
AST parse: OK
Classes:   ValidationError - FOUND
Functions: read_tsv, write_tsv, load_all_s1_ids, load_submission_files,
           _check_schema, _check_coverage, _check_subset,
           _check_no_duplicates, _check_constraint, _check_format,
           _check_self_consistency, validate_outputs, summarise_submission,
           run_smoke_test, run -- all FOUND
`

**Conclusion:** All 16 symbols present. scripts/run_pipeline.ps1 also created -- resolves ISS-007.

---

## EXP-007 . Smoke Test -- Synthetic 10-Entity Dataset (PLANNED)
**Date:** TBD
**Type:** Functional test (no data files required)
**Status:** PLANNED

Run with: python -m src.utils --smoke-test

**Results:** _TO BE FILLED_

---

## EXP-008 . End-to-End Pipeline Run (PENDING -- awaiting data)
**Date:** TBD
**Type:** Full pipeline execution
**Status:** PENDING

**Results:** _TO BE FILLED_

