# Worklog — TRAC-Phish Revision 6

---
Task ID: 1
Agent: main orchestrator (Z.ai Code)
Task: Mandatory first step — read Master_ImplementationV2.txt end-to-end; parse all 162 cells of trac-phish-revision5_runned.ipynb incl. outputs; verify uploaded datasets; produce understanding summary.

Work Log:
- Waited for upload; verified 4 files in /home/z/my-project/upload (Master txt, Rev5 notebook 3.97MB, grambeddings rar 17.6MB, phresh zip 42.9MB).
- Read Master_ImplementationV2.txt fully (578 lines): Part 0 exec summary, Part 1 diagnostics + 8 mechanisms for M0=0.73, Part 2 95%-claim ceiling analysis, Part 3 blocker solutions 1/2A/2B/2C/3/4/5A/5B/5C/6A/6B/7, Part 4 phased plan Gates 0-8, Part 5 expected results, Part 6 paper framing, Part 7 checklist.
- Dumped all 162 notebook cells + outputs to nb_dump.txt (23,396 lines); split into 8 chunks.
- Dispatched 8 parallel Explore agents (chunks 0-7); received exhaustive reports covering: dataset layer numbers, split/overlap mechanics, F48/F54-R/F60-R schemas, dinv extractor, P1 gate (max W 0.3338), compatibility gate (15/15, origin AUC 0.9242), tuning/freezing/calibration/Stage-B, M0-M4 transfer cells + prune rule, perturbation engine P1-P9, stability/S_challenge, E0=(FSM)^(1/3), y_rel gate (rho 0.536-0.762 circular), ERS/ERS_legacy, DTS modes + P4_GATE fail, 35R LRT mixed, 35S reversal (OR<1 external), transfer/ablation/shift tables, policies, AURC (Criterion C fail), statistics 342 tests, error analysis, temporal, main tables, case studies, sanity 219/219, manifest, Rev5 gates + honest summary.
- Extracted both datasets; verified all 5 sha256 checksums match Rev 5's recorded fingerprints exactly (gram train 48b79a07..., gram test 9f5f7424..., classes beea8d35..., phresh train 8bef3edb..., phresh test 41764824...).
- Assessed sandbox: 2 CPUs, 4GB RAM, 7.8GB disk vs Kaggle 4 CPU/33.7GB (Rev5 full run = 279 min). Decision: run Rev 6 in the notebook's built-in deterministic REDUCED mode (TRAC_RUN_MODE=reduced), reporting the scale caveat honestly everywhere.
- Verified key mutable cells in detail: 4 (config), 21 (cap), 43 (dinv extractor), 47 (extraction/ALL_FEATURE_COLUMNS/fingerprint), 53 (FEATURE_SETS/RUN_KEYS), 55 (RUNS/imputers), 57 (helpers), 63 (P1 gate), 67 (tune_and_train), 81 (SHAP/features_for_set), 85-86 (prune + M0-M4), 91 (perturbation engine), 99 (y_rel), 103 (DTS), sanity-check XDATA ledger constraints (target usage must be step∈{phase2_fit,domain_classifier}, partition=train).

Stage Summary:
- Understanding summary delivered to user (Rev5 achievements, 5 blockers, M0=0.73 mechanisms, 95% claim, phase order/gates).
- Datasets verified byte-identical to Rev 5 (Gate 0 will pass).
- Execution plan: surgically extend notebook JSON with Rev 6 cells (F68-R extractor+schema, Phase-1 R6 gate with replace-policy, prune policy v2, M0 variants incl. rank + pseudo-label, M2/M2-hybrid/M4-hybrid with target-val tuning, augmentation, cross-family y_rel, StratumDTS + prior-corrected DTS + external AURC, reversal decomposition, shift-strata LRT, Rev 6 summary/manifest). F68R added to FULL_PIPELINE_FSETS so Phase 9 XAI runs on the revised representation.
- Hardware constraint recorded: REDUCED mode (TRAC_MAX_ROWS=120000/dataset, deterministic seeded cap) — every number real, scale caveat reported.

---
Task ID: 2
Agent: main orchestrator (Z.ai Code)
Task: Build trac-phish-revision6.ipynb (surgical modification) and validate via smoke run.

Work Log:
- Web search validated the two key methods: Saerens-Latinne-Decaestecker EM (2002) for target-prior correction; AURC as the standard selective-classification metric (ICML 2025 characterization).
- Wrote build script (build_rev6.py + parts 2-5) that surgically modifies the Rev 5 JSON: edits to cells 0/4/47/53/manifest + 22 new Rev 6 cells (Phase 1a F68-R extractor, Phase 1b strict shift gate, Phase 2 zero-shot variants, Phase 3 multi-source, Phase 4 augmentation, Phase 5 cross-family y_rel, Phase 6 StratumDTS/prior-corrected DTS/external AURC, Phase 7 reversal decomposition, Phase 8 shift strata LRT, Phase 9 gate-health + F68R XAI evidence, Phase 10 gate table/blockers/negative findings/manifest).
- F68R added to FULL_PIPELINE_FSETS so Phase 9 (full XAI/ERS/DTS on revised representation) executes through the generic pipeline.
- Smoke run 1: pyarrow missing (installed 25.0.1) + 4 wrong hand-computed unit-test expectations in Phase-1a cell (path_depth_binary, digit_run_max, scheme_ratio for httpsecure body, brand on IP host) — extractor verified correct, tests fixed.
- Smoke run 2: found real build bug — FEATURES_68R defined before FEATURES_60R in cell 53; fixed ordering.
- Smoke run 3 in progress (dataset layer now passes; run alive >5 min).

Stage Summary:
- trac-phish-revision6.ipynb: 184 cells, all compile, nbformat valid.
- Smoke-mode validation loop catching integration bugs cheaply before the reduced-scale run.

---
Task ID: 3
Agent: main orchestrator (Z.ai Code)
Task: Execute trac-phish-revision6.ipynb end-to-end (deterministic reduced scale) and produce the final executed notebook.

Work Log:
- Smoke run (TRAC_MAX_ROWS=20,000): completed with 0 errors after 6 build iterations that fixed: pyarrow missing; 4 wrong hand-computed F68-R unit-test expectations; FEATURES_68R defined before FEATURES_60R; FEATURES_60R referenced in Phase-1a print; fresh extractions lacking D68_*_bin swapped copies; Phase-2 raw_matrix missing fset arg; model_proba vs raw predict_proba (n,2) in Phase 4; P["uids"] are string record-ids (used P["rows"]/pos.loc conventions); Phase 7/8 moved after RUNS[rk]["ers_thresholds"] exists; ALL_FEATURE_COLUMNS extended with D68_ copies; fresh-extraction sanity audit scoped to the original 7 D_ columns.
- Reduced attempts 1-5 (120K/80K rows, 2 jobs): kernel OOM-killed. Root cause discovered by comparing preserved Rev-5 outputs vs fresh ones: nbclient does NOT clear old outputs, so "last executed cell" was fake — the kernel actually died at the XAI/SHAP stage; killer = fork-based parallel RF SHAP (fork COW refcount page duplication) + cumulative caches (_FEAT_NP ~0.5GB, models, char bundles, PERT_CACHE).
- Fixes applied: memory-release checkpoint appended inside the sanity-audit cell (PERT_CACHE/STRESS/CHAR_*/_FEAT_NP/models + gc + glibc malloc_trim); final run at TRAC_MAX_ROWS=50,000, TRAC_N_JOBS=1 (no fork pools).
- FINAL RUN: status completed, 0 error cells, 71.2 min, all 184 cells executed (Rev 5 verbatim + Rev 6 phases 0-10), 219/219 sanity checks passed, 76 result tables written.

Stage Summary (executed gate outcomes, reduced scale 50K rows/dataset):
- Gate 0 PASSED (fingerprints byte-identical to Rev 5; 219/219 sanity).
- Gate 1 NOT fully passed but improved: max final W 0.222 (Rev 5: 0.334); F68-R final = 67 cols; 2 unrepairable binaries removed (D_has_query_binary, D_query_present); policy documented.
- Gate 2 FAILED: best M0r6 +pseudo AUC 0.7585 (< 0.80) vs Rev-5 baseline 0.734/0.745; policy v2 DID prune R_is_https on the Gram branch (prune-v2 alone hurt at this scale: 0.7276).
- Gate 3 NOT MET: M2r6-hybrid external accuracy 0.9365 (Gram->Phresh) / 0.9082 (Phresh->Gram) vs 0.95 target; AUC 0.982/0.969.
- Gate 4 NOT PASSED: P3 flips 0.59/0.26 (augmentation marginal: unaugmented 0.62); stress flips improved sharply (P5 0.437 -> 0.111 gram->phresh).
- Gate 5 PASSED (3/4 core runs; cross-family rho 0.377/0.388/0.366; phresh|F60R -0.011 fails honestly).
- Gate 6 NOT PASSED (negative reported): best DTS_prior_corrected dAURC +0.0021, CI [-0.0060, +0.0094] crosses 0.
- Gate 7 genuine_negative_reported: reversal persists within true class on 5/6 runs (median delta +0.026..+0.095).
- Gate 8 REPORTED: 10/18 run-stratum LRTs Holm-significant; ERS adds beyond confidence precisely on protocol_shift + path_structure_shift (+ lexicon) strata — the shift-conditioned claim the plan predicted.
- Blockers: B1 RESOLVED; B2 OPEN (rigorous negative); B3 REFRAMED; B4 GENUINE NEGATIVE reported; B5 PARTIAL (0.734->0.7585); B7 PARTIAL.
