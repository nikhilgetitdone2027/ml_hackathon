# PROJECT STATUS
**Project:** Amazon ML Hackathon 2026 — Business Entity Resolution
**Architecture:** Graph-Aware Hybrid Dual-Encoder Pipeline
**Last Updated:** 2026-09-26 (updated after Phase 7 — PIPELINE COMPLETE)

---

## Overall Progress

| Phase | Name | Status | Key Output |
|-------|------|--------|-----------|
| 1 | Repository Setup | ✅ COMPLETED | Project scaffold, configs, README |
| 2 | Preprocessing | ✅ COMPLETED | `src/preprocessor.py` |
| 3 | Blocking Layer | ✅ COMPLETED | `src/blocker.py` |
| 4 | Feature Engineering | ✅ COMPLETED | `src/features.py` |
| 5 | Modelling & Calibration | ✅ COMPLETED | `src/model.py` |
| 6 | Post-Processing & F0.5 Opt. | ✅ COMPLETED | `src/optimizer.py` |
| 7 | Output Generation & Validation | ✅ COMPLETED | `src/utils.py`, `scripts/run_pipeline.ps1` |

**Current Phase:** 🎉 **ALL PHASES COMPLETE** — ready for data + end-to-end run
**Blocking Issues:** ISS-003 (data files), ISS-004 (pip install)

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
│   ├── run_pipeline.sh            ✅  End-to-end Bash orchestrator (Linux/Mac)
│   └── run_pipeline.ps1           ✅  End-to-end PowerShell runner (Windows)
├── src/
│   ├── __init__.py                ✅  Package init (v1.0.0)
│   ├── preprocessor.py            ✅  Phase 2 — FULL IMPLEMENTATION
│   ├── blocker.py                 ✅  Phase 3 — FULL IMPLEMENTATION
│   ├── features.py                ✅  Phase 4 — FULL IMPLEMENTATION
│   ├── model.py                   ✅  Phase 5 — FULL IMPLEMENTATION
│   ├── optimizer.py               ✅  Phase 6 — FULL IMPLEMENTATION
│   └── utils.py                   ✅  Phase 7 — FULL IMPLEMENTATION
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

**All phases complete.** To run the pipeline:

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place data files in data/
#    source1.tsv, source2.tsv, source3.tsv, ground_truth.tsv

# 3. Run smoke test (no data required)
python -m src.utils --smoke-test

# 4. Run full pipeline
.\scripts\run_pipeline.ps1

# 5. Validate outputs
python -m src.utils --validate --summary --config configs/config.yaml
```
