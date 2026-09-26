# PROJECT STATUS
**Project:** Amazon ML Hackathon 2026 — Business Entity Resolution
**Architecture:** Graph-Aware Hybrid Dual-Encoder Pipeline
**Last Updated:** 2026-09-26

---

## Overall Progress

| Phase | Name | Status | Key Output |
|-------|------|--------|-----------|
| 1 | Repository Setup | ✅ COMPLETED | Project scaffold, configs, README |
| 2 | Preprocessing | ✅ COMPLETED | `src/preprocessor.py` |
| 3 | Blocking Layer | ✅ COMPLETED | `src/blocker.py` |
| 4 | Feature Engineering | ✅ COMPLETED | `src/features.py` |
| 5 | Modelling & Calibration | ✅ COMPLETED | `src/model.py` |
| 6 | Post-Processing & F0.5 Opt. | 🔲 PENDING | `src/optimizer.py` |
| 7 | Output Generation & Validation | 🔲 PENDING | `src/utils.py`, outputs |

**Current Phase:** Ready to begin **Phase 6**
**Blocking Issues:** None

---

## File Inventory (actual state on disk)

```
amazon-ml-2026/
├── configs/
│   └── config.yaml                ✅  Master config (all hyperparams)
├── data/                          ⚠️  EMPTY — user must place TSV files here
│   └── (source1.tsv, source2.tsv, source3.tsv, ground_truth.tsv)
├── notebooks/                     ✅  Directory created
├── output/
│   ├── cache/                     ✅  Directory created (populated at runtime)
│   └── models/                    ✅  Directory created (populated at runtime)
├── scripts/
│   └── run_pipeline.sh            ✅  End-to-end bash orchestrator
├── src/
│   ├── __init__.py                ✅  Package init (v1.0.0)
│   ├── preprocessor.py            ✅  Phase 2 — FULL IMPLEMENTATION
│   ├── blocker.py                 ✅  Phase 3 — FULL IMPLEMENTATION
│   ├── features.py                ✅  Phase 4 — FULL IMPLEMENTATION
│   ├── model.py                   ✅  Phase 5 — FULL IMPLEMENTATION
│   ├── optimizer.py               🔲  Phase 6 stub (raises NotImplementedError)
│   └── utils.py                   🔲  Phase 7 stub (raises NotImplementedError)
├── .gitignore                     ✅
├── README.md                      ✅  Comprehensive documentation
└── requirements.txt               ✅  All deps with version pins
```

---

## Runtime Prerequisites (not yet satisfied)

- [ ] Python ≥ 3.10 virtual environment created
- [ ] `pip install -r requirements.txt` executed
- [ ] Data files placed in `data/` directory
- [ ] First run will download MiniLM-L12-v2 model (~120 MB)

---

## Evaluation Target

- **Metric:** Macro-averaged F0.5
- **Beta:** 0.5 (precision weighted 2× over recall)
- **Key constraint:** S2/S3 ID → at most ONE S1 entity
- **Singleton rule:** Empty list = 1.0 contribution per entity

---

## Next Immediate Action

Implement **Phase 6** (`src/optimizer.py`):
- Constraint resolution (S2/S3 → max-1 S1, keep highest-prob edge)
- Expected F0.5 maximisation per S1 (scan top-k subsets, pick argmax)
