# =============================================================
# run_pipeline.ps1
# End-to-end PowerShell pipeline runner for Windows.
# Usage:  .\scripts\run_pipeline.ps1 [-Config path\to\config.yaml]
#         .\scripts\run_pipeline.ps1 -SmokeTest
# =============================================================

param(
    [string]$Config     = "configs\config.yaml",
    [switch]$SmokeTest  = $false,
    [switch]$SkipPhase2 = $false,   # skip if preprocessed cache already exists
    [switch]$SkipPhase3 = $false,   # skip if candidate_pairs.tsv already exists
    [switch]$NoCache    = $false,   # force re-encode embeddings
    [switch]$InferenceOnly = $false # skip model training, load saved model
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

# ── Colour helpers ─────────────────────────────────────────────
function Write-Step  { param($msg) Write-Host "`n[$([datetime]::Now.ToString('HH:mm:ss'))] >>> $msg" -ForegroundColor Cyan }
function Write-Ok    { param($msg) Write-Host "    $msg" -ForegroundColor Green }
function Write-Warn  { param($msg) Write-Host "    WARNING: $msg" -ForegroundColor Yellow }
function Write-Err   { param($msg) Write-Host "    ERROR:   $msg" -ForegroundColor Red }

# ── Banner ─────────────────────────────────────────────────────
Write-Host ""
Write-Host "=============================================================" -ForegroundColor Magenta
Write-Host " Amazon ML Hackathon 2026 -- Business Entity Resolution      " -ForegroundColor Magenta
Write-Host " Graph-Aware Hybrid Dual-Encoder Pipeline                    " -ForegroundColor Magenta
Write-Host " Config : $Config                                            " -ForegroundColor Magenta
Write-Host " Root   : $ProjectRoot                                       " -ForegroundColor Magenta
Write-Host "=============================================================" -ForegroundColor Magenta
Write-Host ""

# ── Ensure output directories exist ────────────────────────────
New-Item -ItemType Directory -Force -Path "output\cache"  | Out-Null
New-Item -ItemType Directory -Force -Path "output\models" | Out-Null

# ── Smoke test mode (no data required) ─────────────────────────
if ($SmokeTest) {
    Write-Step "Smoke Test Mode"
    python -m src.utils --smoke-test --config $Config
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Smoke test FAILED."
        exit 1
    }
    Write-Ok "Smoke test PASSED."
    exit 0
}

# ── Helper: timed phase runner ──────────────────────────────────
function Invoke-Phase {
    param(
        [string]$Name,
        [string[]]$PythonArgs
    )
    Write-Step $Name
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    python @PythonArgs
    $sw.Stop()
    if ($LASTEXITCODE -ne 0) {
        Write-Err "$Name FAILED (exit code $LASTEXITCODE)"
        exit $LASTEXITCODE
    }
    Write-Ok "$Name completed in $([math]::Round($sw.Elapsed.TotalSeconds, 1))s"
}

# ── Phase 2: Preprocessing ──────────────────────────────────────
if (-not $SkipPhase2) {
    Invoke-Phase "Phase 2 . Preprocessing" @(
        "-m", "src.preprocessor", "--config", $Config
    )
} else {
    Write-Warn "Phase 2 skipped (--SkipPhase2 flag set)"
}

# ── Phase 3: Blocking ───────────────────────────────────────────
if (-not $SkipPhase3) {
    $blockerArgs = @("-m", "src.blocker", "--config", $Config)
    if ($NoCache) { $blockerArgs += "--no-cache" }
    Invoke-Phase "Phase 3 . Blocking" $blockerArgs
} else {
    Write-Warn "Phase 3 skipped (--SkipPhase3 flag set)"
}

# ── Phase 4: Feature Engineering ───────────────────────────────
Invoke-Phase "Phase 4 . Feature Engineering" @(
    "-m", "src.features", "--config", $Config
)

# ── Phase 5: Modelling ─────────────────────────────────────────
$modelArgs = @("-m", "src.model", "--config", $Config)
if ($InferenceOnly) { $modelArgs += "--inference-only" }
Invoke-Phase "Phase 5 . Modelling and Calibration" $modelArgs

# ── Phase 6: Post-Processing ────────────────────────────────────
Invoke-Phase "Phase 6 . Post-Processing and F0.5 Optimisation" @(
    "-m", "src.optimizer", "--config", $Config
)

# ── Phase 7: Validation ─────────────────────────────────────────
Invoke-Phase "Phase 7 . Output Validation and Summary" @(
    "-m", "src.utils", "--config", $Config, "--validate", "--summary"
)

# ── Final banner ────────────────────────────────────────────────
Write-Host ""
Write-Host "=============================================================" -ForegroundColor Green
Write-Host " Pipeline COMPLETE                                           " -ForegroundColor Green
Write-Host " candidate_pairs.tsv  --> output\candidate_pairs.tsv        " -ForegroundColor Green
Write-Host " matching_results.tsv --> output\matching_results.tsv        " -ForegroundColor Green
Write-Host "=============================================================" -ForegroundColor Green
Write-Host ""
