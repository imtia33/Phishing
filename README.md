# TRAC-Phish — Revision 6

**Beyond Prediction Confidence: Reliability-Calibrated Explanations for Phishing URL Detection**
Revision 6 — implementation of `Master_ImplementationV2.txt` (phases 0–10 with hard gates).

## What this repository contains

| Path | Description |
|---|---|
| `Master_ImplementationV2.txt` | The master implementation plan (Phases 0–10, blocker solutions 1–7, expected results). |
| `notebooks/trac-phish-revision5_runned.ipynb` | The executed Revision 5 notebook (the surgical baseline — **not** modified). |
| `notebooks/trac-phish-revision6.ipynb` | The Revision 6 notebook: Rev 5 preserved verbatim + 22 clearly-marked `REVISION 6 PHASE n` cells implementing every phase of the master plan. |
| `notebooks/trac-phish-revision6_executed.ipynb` | The **executed** Revision 6 notebook with all gate results and outputs (pushed after the run completes). |
| `build/` | The surgical build scripts that construct the Rev 6 notebook from the Rev 5 notebook (fully traceable: every edit cites the plan section it implements). |
| `results/` | Key gate JSONs + result tables exported from the executed run. |
| `worklog.md` | Agent work log (task IDs, decisions, bugs found and fixed). |

## Revision 6 phases (each with a pre-registered hard gate)

| Phase | What it implements | Gate |
|---|---|---|
| 0 | Dataset layer preserved verbatim; fingerprints re-verified (`48b79a07…`, `9f5f7424…`, `beea8d35…`, `8bef3edb…`, `41764824…`) | Rev 5 dataset/compatibility assertions pass |
| 1 | **F68-R** = F60-R + 8 new domain-invariant features (69 columns); strict Phase-1 shift gate over all 15 new features (≤0.15 pass, ≤0.25 binarise, >0.25 replace) | all new features ≤ 0.15 Wasserstein |
| 2 | Zero-shot ceiling: `M0 F68-R`, `M0 prune-v2` (prune iff domain-importance > 0.05 AND (source-SHAP < 0.01 **OR prevalence shift > 0.30**)), `M0 +rank` (source-train empirical CDF), `M0 +pseudo` (calibrated-confidence self-training on target TRAIN scores only) | best strict-external AUC ≥ 0.80 |
| 3 | Semi-supervised multi-source: `M2r6`, `M2r6-hybrid` (+CharSVD), `M4r6-hybrid` (+char fusion), **target-validation-tuned** hyper-parameters | external accuracy ≥ 0.95 both directions |
| 4 | Robustness repair: identity-preserving (P1,P2) + stress (P5,P6,P7) augmentation of SOURCE training, weight 0.30 | P3 and P5/P6/P7 external flip ≤ 15% |
| 5 | ERS target repair: cross-family `y_rel = 0.5·S_challenge + 0.5·S_calib_P3` (P1/P2 excluded) | cross-family ρ(E0, y_rel) > 0.10 on ≥ 3/4 runs |
| 6 | DTS repair: `StratumDTS` (route by C at 0.90), prior-corrected DTS (Saerens-Latinne-Decaestecker EM weights), evaluated on **external AURC** | ≥ 1 variant beats confidence-only externally |
| 7 | Reversal diagnosis: class confounding / class-asymmetric accuracy / F,M-per-class decomposition + within-class reversal test | explained, or reported as genuine negative |
| 8 | Test B restated as shift-conditioned: LRT per stratum (protocol / path-structure / lexicon), Holm within family | reporting gate (no universal claim) |
| 9 | Full XAI/ERS/DTS pipeline on F68-R (runs through `FULL_PIPELINE_FSETS`) + gate-health statement | conditional on Gates 1–8 |
| 10 | Gate table, blocker status (resolved / partial / open), negative-finding register, manifest update | — |

## Non-negotiables enforced

No PhishTank. No LegitPhish in the primary pipeline. No HTML. URL-only features.
No PhreshPhish TEST labels for tuning, calibration, thresholds, ERS or DTS fitting.
No target-driven post-hoc pruning. Domain-disjoint (eTLD+1) Gram splits preserved.
Exact + canonical overlap controls preserved. Structural path missingness stays semantically repaired.
Every number comes from an executed computation; negative results are reported as findings.

## Execution note (scale caveat)

Revision 6 was executed in the notebook's built-in deterministic **reduced** mode
(`TRAC_RUN_MODE=reduced`, `TRAC_MAX_ROWS=120,000` per dataset) because the available execution
environment provides a fraction of the memory/compute of the revision-5 full run (2 CPUs / 4 GB RAM).
Every stage and every gate still executes; every number is a real measurement on the stated reduced
corpus; the scale caveat is carried into the manifest, the phase-gate table and the final summary.
No gate was relaxed because of the scale, and no result is reported without it.

## How the Rev 6 notebook was built

`trac-phish-revision6.ipynb` is **not** a rewrite: it is `trac-phish-revision5_runned.ipynb` with
(1) small, audited edits to the config / extraction / feature-set / manifest cells, and (2) 22 new
cells that implement the master plan's phases. The transformation is fully scripted in `build/`
(`build_rev6.py` + parts 2–5 → `build_rev6_full.py`), so the provenance of every change is a cited
plan section. The Rev 5 cells and their executed outputs are preserved.

## Reproduce

```bash
pip install numpy pandas scipy scikit-learn xgboost lightgbm shap tldextract pyarrow nbformat nbclient joblib matplotlib seaborn statsmodels
# datasets: grambeddings_dataset_main.rar + phreshphish_url_only_2026.zip in run/input/
export TRAC_RUN_MODE=reduced          # deterministic reduced scale (see caveat)
export TRAC_INPUT_ROOT=$PWD/run/input
export TRAC_WORK_ROOT=$PWD/run/work
jupyter nbconvert --to notebook --execute notebooks/trac-phish-revision6.ipynb \
       --output trac-phish-revision6_executed.ipynb
```
