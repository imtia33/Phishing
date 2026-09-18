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
