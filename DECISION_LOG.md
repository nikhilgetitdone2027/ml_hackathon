# DECISION LOG
**Project:** Amazon ML Hackathon 2026 — Business Entity Resolution
**Last Updated:** 2026-09-26

Format: Each entry has ID, date, decision, rationale, alternatives considered, and impact.
Entries are append-only — never delete previous decisions.

---

## DEC-001 · Architecture Choice: Dual-Encoder Hybrid Pipeline
**Date:** 2026-09-26
**Phase:** 1 (Design)
**Status:** ACCEPTED

**Decision:**
Use a two-stage pipeline: (1) blocking with dual encoders (TF-IDF + MiniLM), (2) pairwise XGBoost classifier on hand-crafted features.

**Rationale:**
- End-to-end neural matching (e.g., cross-encoders) would be too slow at inference for large S2/S3 universes without pre-filtering.
- TF-IDF excels at abbreviation overlaps; MiniLM handles paraphrases and multilingual text. Neither alone achieves sufficient recall.
- XGBoost on hand-crafted features is interpretable, fast, and well-calibrated with isotonic regression.

**Alternatives Considered:**
- Pure cross-encoder (BERT-style): too slow for O(S1 × S2S3) pairs; no blocking.
- ColBERT / bi-encoder with hard negatives: more complex, requires iterative training; saved for future iteration.
- Pure TF-IDF + threshold: no semantic generalisation; would fail on French (zero-shot country).

**Impact:**
Entire pipeline architecture; all subsequent phases follow this design.

---

## DEC-002 · Model Selection: paraphrase-multilingual-MiniLM-L12-v2
**Date:** 2026-09-26
**Phase:** 1 (Design)
**Status:** ACCEPTED

**Decision:**
Use `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` for semantic embeddings.

**Rationale:**
- **Multilingual:** Covers 50+ languages; handles French (unseen test country) zero-shot.
- **Size:** 117M parameters — well within the 8B hard constraint.
- **License:** Apache 2.0 — compliant with competition rules.
- **Quality:** Strong performance on multilingual semantic similarity benchmarks (MTEB).
- **Speed:** 384-dim embeddings; encodes ~256 sentences/second on CPU.

**Alternatives Considered:**
- `all-MiniLM-L6-v2`: Smaller, faster, but English-only — fails French zero-shot requirement.
- `LaBSE`: Better multilingual coverage, but larger (471M) and slower.
- `multilingual-e5-large`: Better quality, but 560M params and would require more GPU memory.

**Impact:** Phase 3 (blocking), Phase 4 (sem_score feature), runtime performance.

---

## DEC-003 · Abbreviation Mining Strategy: Data-Driven Prefix Matching
**Date:** 2026-09-26
**Phase:** 2 (Preprocessing)
**Status:** ACCEPTED

**Decision:**
Mine token-replacement pairs automatically from ground-truth aligned pairs using prefix-match heuristic: keep (noisy_token, canonical_token) pairs where `canonical.startswith(noisy)` and `len(noisy) < len(canonical)`.

**Rationale:**
- **No hardcoding:** Competition data may have domain/country-specific abbreviations unknown in advance.
- **Precision guard:** Prefix constraint eliminates most spurious matches (e.g., "st" → "street" is a prefix, but "st" → "saint" is not — the latter would require additional validation).
- **Frequency filter (`min_token_freq=5`):** Avoids overfitting to rare noise.

**Alternatives Considered:**
- Hardcoded dictionary (Corp→Corporation, Ltd→Limited, etc.): Brittle; misses domain/country-specific abbreviations; fails zero-shot French.
- Edit-distance threshold for mining: Too permissive; captures unrelated word pairs.
- TF-IDF alignment: Complex; prefix matching achieves similar precision with simpler code.

**Impact:** Quality of `norm_name` column; affects all downstream similarity scores.

---

## DEC-004 · FAISS Index Type: IndexFlatIP (Exact) as Default
**Date:** 2026-09-26
**Phase:** 3 (Blocking)
**Status:** ACCEPTED

**Decision:**
Default to `IndexFlatIP` (exact inner product search on L2-normalised vectors) rather than an approximate IVF index.

**Rationale:**
- Exact search guarantees deterministic results and maximum recall — critical during blocking where recall is the primary metric.
- For datasets up to ~500k records (typical hackathon scale), `IndexFlatIP` completes in seconds on CPU.
- Embeddings are cached, so the expensive encode step only runs once.

**Alternatives Considered:**
- `IndexIVFFlat`: ~5–10% recall loss at nprobe=64; only beneficial for >500k vectors.
- `IndexHNSW`: Better speed/recall trade-off than IVF, but non-deterministic across runs.

**Config note:** Can switch to `"IVF"` via `blocking.semantic.faiss_index_type` for large datasets.

**Impact:** Phase 3 blocking recall, runtime.

---

## DEC-005 · Candidate Cap: 50 per S1 Entity
**Date:** 2026-09-26
**Phase:** 3 (Blocking)
**Status:** ACCEPTED

**Decision:**
Hard cap candidates at 50 per S1 entity (30 from TF-IDF + 30 from semantic → union → top-50).

**Rationale:**
- Empirically, entity matching datasets typically have 1–5 true matches per reference entity.
- Top-50 provides a strong recall safety margin (expected recall ≥ 98%) while limiting Phase 4 feature computation to 50× S1 pairs.
- If blocking recall falls below 95% (warned in logs), the config value can be increased without code changes.

**Alternatives Considered:**
- Top-100: Higher recall but doubles feature computation time.
- Adaptive cap (proportional to cluster size): Complexity; cluster sizes unknown at blocking time.

**Impact:** Output size of `candidate_pairs.tsv`, Phase 4 runtime, false-positive density for model training.

---

## DEC-006 · Negative Sampling Strategy: Closed-Universe CV
**Date:** 2026-09-26
**Phase:** 5 (Modelling)
**Status:** ACCEPTED

**Decision:**
Use GroupKFold on `source1_entity_id` WITHOUT additional negative re-sampling, relying on the blocker's full S2/S3 candidate pool as the natural negative set.

**Rationale:**
- The blocker already retrieves candidates from the entire S2/S3 universe (not a subset). Each S1 entity's 50 candidates naturally include hard negatives from the full pool.
- Standard GroupKFold on pairs, with no additional sampling, correctly simulates test density because the blocker's negative pool is already representative.
- Adding random S2/S3 negatives outside the blocking window would be misleading — the model never sees those at test time either.

**Alternatives Considered:**
- Adding random global negatives: Distorts the class distribution in a way that doesn't reflect test time.
- BM25-based hard negative mining: More complex; blocked negatives are already hard negatives.

**Impact:** Training set composition, model calibration, CV metric reliability.

---

## DEC-007 · Calibration Method: Isotonic Regression (OOF)
**Date:** 2026-09-26
**Phase:** 5 (Modelling)
**Status:** ACCEPTED

**Decision:**
Calibrate XGBoost probabilities with `IsotonicRegression` fitted on out-of-fold predictions, not on a held-out set.

**Rationale:**
- Isotonic regression is non-parametric — it doesn't assume sigmoid/linear miscalibration curves.
- OOF calibration uses all training data for both training the model and calibrating it, maximising data efficiency.
- Critical for Phase 6's Expected-F0.5 optimiser, which uses probabilities as Bernoulli truth estimates — poor calibration would give wrong threshold decisions.

**Alternatives Considered:**
- Platt scaling (sigmoid): Parametric; often under-fits the calibration curve for XGBoost.
- Temperature scaling: Designed for neural networks, not tree models.
- Held-out calibration set: Wastes 20% of training data.

**Impact:** Quality of `cal_prob` column, Phase 6 threshold decisions.

---

## DEC-008 · Post-Processing: Expected F0.5 over Static Threshold
**Date:** 2026-09-26
**Phase:** 6 (Design)
**Status:** PENDING IMPLEMENTATION

**Decision (pre-implementation):**
For each S1 entity, select the top-k candidates that maximise the **Expected F0.5** computed from calibrated Bernoulli probabilities, rather than applying a global threshold.

**Rationale:**
- A global threshold (e.g., 0.5) ignores per-entity probability distributions.
- Entities with 10 high-confidence candidates need a different threshold than entities with 1 medium-confidence candidate.
- Expected F0.5 = E[F0.5] under the assumption that each predicted pair is a Bernoulli trial with probability `p`. Maximising this expectation naturally adapts k to each entity's evidence.
- Handles the singleton case correctly: k=0 gives Expected-F0.5 = 1.0 (if all probabilities are low, no prediction = perfect for singletons).

**Formula (to implement):**
For a ranked list of k predictions with calibrated probabilities p_1 ≥ p_2 ≥ ... ≥ p_k:
```
E[TP] = sum(p_i for i in 1..k)
E[FP] = k - E[TP]
E[FN] = E[total positives] - E[TP]
E[Precision] = E[TP] / k  (when k > 0)
E[Recall] = E[TP] / E[total positives]
E[F0.5] = (1 + 0.25) * E[P] * E[R] / (0.25 * E[P] + E[R])
```

**Impact:** Final `matching_results.tsv` quality, competition score.
