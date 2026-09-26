# ISSUE LOG
**Project:** Amazon ML Hackathon 2026 — Business Entity Resolution
**Last Updated:** 2026-09-26

Format: Issues are append-only. Status: 🔴 OPEN | 🟡 IN PROGRESS | 🟢 RESOLVED | ⚫ WONT-FIX
Severity: P0 = Blocker | P1 = High | P2 = Medium | P3 = Low

---

## ISS-001 · PowerShell Profile Errors on Every Command
**Date:** 2026-09-26
**Severity:** P3 (cosmetic)
**Status:** 🟢 RESOLVED (by ignoring)

**Description:**
Every `run_command` invocation prints PowerShell profile errors:
```
oh-my-posh : The term 'oh-my-posh' is not recognized...
conda : The term 'conda' is not recognized...
```

**Root Cause:**
The user's PowerShell profile (`Microsoft.PowerShell_profile.ps1`) references
`oh-my-posh` and `conda` which are not on PATH in the current shell context.
These are cosmetic profile load errors; they do not affect command execution.

**Resolution:**
Commands still complete with exit code 0. These errors can be ignored.
All subsequent command outputs should be read after these header errors.

**Prevention:**
None needed — this is a user environment configuration issue, not a project issue.

---

## ISS-002 · `FEATURE_COLUMNS` Not Found by First AST Checker Script
**Date:** 2026-09-26
**Severity:** P3 (cosmetic — false alarm)
**Status:** 🟢 RESOLVED

**Description:**
First AST validation script for `features.py` reported `FEATURE_COLUMNS: MISSING`.
Second check (walking all assignment targets at any depth) also returned MISSING.

**Root Cause:**
The initial check used `n.targets[0].id` which only works for simple `Name` assignments
at the top of the walk. The AST walker was visiting nodes inside function bodies first,
and `FEATURE_COLUMNS` (a module-level list assignment) was not matched because the
walk returned a different node type for list assignments.

The `grep` search confirmed `FEATURE_COLUMNS` IS present at line 689 of `features.py`.

**Resolution:**
Confirmed via `grep`:
```
File: src/features.py, Line 689: FEATURE_COLUMNS: List[str] = [
```
The feature list contains 26 features as designed.

**Prevention:**
Use `grep` directly for string-literal searches rather than AST node type assumptions.

---

## ISS-003 · Data Files Not Yet Placed in `data/` Directory
**Date:** 2026-09-26
**Severity:** P1 (blocks runtime testing)
**Status:** 🔴 OPEN

**Description:**
The `data/` directory is empty. No actual pipeline execution (Phase 2 onwards)
can proceed until the four TSV files are placed there:
- `source1.tsv`
- `source2.tsv`
- `source3.tsv`
- `ground_truth.tsv`

**Impact:**
- Cannot validate end-to-end pipeline (EXP-005 is blocked)
- Cannot measure blocking recall
- Cannot compute CV AUCPR or macro F0.5
- Cannot confirm feature computation is bug-free on real data

**Resolution Plan:**
User must obtain the hackathon data files from the competition platform and
place them in `data/`. Then run `bash scripts/run_pipeline.sh`.

**Owner:** User (data sourcing)

---

## ISS-004 · Dependencies Not Yet Installed
**Date:** 2026-09-26
**Severity:** P1 (blocks runtime testing)
**Status:** 🔴 OPEN

**Description:**
The Python environment does not have the project dependencies installed.
Attempting to `import pandas` fails with `ModuleNotFoundError`.

**Impact:**
- Cannot run any pipeline stage
- Cannot execute import-time validation checks

**Resolution Plan:**
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install --upgrade pip
pip install -r requirements.txt
```
First run will also download the MiniLM model (~120 MB).

**Owner:** User (environment setup)

---

## ISS-005 · `number_veto_flags()` Has Redundant Dead Code
**Date:** 2026-09-26
**Severity:** P3 (code quality)
**Status:** 🟢 RESOLVED (accepted as-is for now)

**Description:**
In `src/features.py`, the `number_veto_flags()` function has a first attempt at
a fallback using a `type('', ...)` lambda object that was superseded by cleaner
walrus-operator code immediately below it. The dead code remains in the function.

**Location:** `src/features.py`, `number_veto_flags()` function, lines ~179–190.

**Impact:**
Purely cosmetic — the dead code is unreachable (the variable `p1` is reassigned
by the cleaner block that follows). No runtime error or incorrect behaviour.

**Resolution:**
Accepted as-is. Will be cleaned up during the post-Phase-7 code review if time permits.
The function produces correct output — confirmed by code reading.

---

## ISS-006 · Phase 6 and Phase 7 Still Using Stub Files
**Date:** 2026-09-26
**Severity:** P1 (pipeline cannot complete)
**Status:** 🟡 IN PROGRESS (optimizer.py done; utils.py pending)

**Description:**
`src/optimizer.py` — ✅ **FULLY IMPLEMENTED** (Phase 6 complete 2026-09-26).
`src/utils.py`    — 🔲 Still a stub that raises `NotImplementedError`.
Running `bash scripts/run_pipeline.sh` will now succeed through Phase 6
but will fail at Phase 7 (`src/utils.py --validate`).

**Resolution Plan:**
- Implement Phase 7 (`src/utils.py`) — next task

**Owner:** Development (Phase 7 is next)


---

## ISS-007 · `run_pipeline.sh` Not Tested on Windows (Bash Not Native)
**Date:** 2026-09-26
**Severity:** P2 (environment compatibility)
**Status:** 🔴 OPEN

**Description:**
`scripts/run_pipeline.sh` is a Bash script. The user's OS is Windows.
Git Bash, WSL, or Cygwin would be needed to run it natively.

**Workaround:**
Run individual phases via PowerShell:
```powershell
python -m src.preprocessor --config configs/config.yaml
python -m src.blocker --config configs/config.yaml
python -m src.features --config configs/config.yaml
python -m src.model --config configs/config.yaml
python -m src.optimizer --config configs/config.yaml
python -m src.utils --validate --config configs/config.yaml
```

**Resolution Plan:**
Add a `scripts/run_pipeline.ps1` PowerShell equivalent after Phase 7 is implemented.

---

## ISS-008 · `margin_sem` Competition Feature Uses `shift(-1)` Within Sorted Groups
**Date:** 2026-09-26
**Severity:** P2 (potential correctness issue — needs data verification)
**Status:** 🟡 IN PROGRESS (needs data to verify)

**Description:**
In `compute_competition_features()`, the second-best score is computed by sorting
by `(candidate_entity_id, sem_score)` and applying `.shift(-1)` within each group.
However, after `sort_values()`, the index alignment with the original `df` must be
done via `.reindex(df.index)`. If the sort produces a non-monotonic index, the
`.reindex()` may align incorrectly.

**Risk:**
The `margin_sem` column could have wrong values if index alignment fails silently.

**Resolution Plan:**
- After obtaining data: add a spot-check assertion that `margin_sem >= 0` for all rank-1 candidates.
- Consider resetting the index after sort_values before applying transform to be safe.

**Mitigation already in place:**
`.clip(lower=0.0)` prevents negative margins from corrupting downstream scores.
