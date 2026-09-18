
# ============================================================================
# NEW CELLS C/D — REVISION 6 PHASES 2 & 3 (inserted after the Rev-5 M0-M4 cell)
# ============================================================================
CELL_C_MD = md("""## REVISION 6 PHASE 2 — Zero-shot transfer ceiling on F68-R (Master V2 Phase 2, Gate 2)

Four **source-only** variants are evaluated on the strict-external population (the target TEST partition
and its labels are never touched):

| variant | training data | mechanism |
|---|---|---|
| `M0r6 F68-R` | source TRAIN only | frozen Stage-A primary of the F68-R run |
| `M0r6 F68-R prune-v2` | source TRAIN only | refit on columns kept by the **prune policy v2**: prune iff domain importance > 0.05 AND (source \\|SHAP\\| < 0.01 OR prevalence shift > 0.30) — the fix that removes the `R_is_https` protocol shortcut (Blocker 7 / Solution 5B.3) |
| `M0r6 F68-R +rank` | source TRAIN only | each column replaced by its source-TRAIN empirical-CDF rank (Solution 5A.2; rank features are robust to monotone shift) |
| `M0r6 F68-R +pseudo` | source TRAIN + pseudo-labelled target TRAIN | self-training from **calibrated** confidence ≥ 0.90 only (Solution 5B.1); true target labels are used *nowhere* in the fit |

**Gate 2:** at least one variant reaches strict-external ROC-AUC ≥ 0.80 on Gram→Phresh. If it fails, the
representation (not the tuning) is diagnosed as the bottleneck and the failure is reported.""")

CELL_C = code(r'''# ---------------------------------------------------------------------------
# REVISION 6 PHASE 2 — zero-shot transfer ceiling on F68-R (Master V2 Phase 2)
# ---------------------------------------------------------------------------
class _RankTransform:
    """Replace each column by its source-TRAIN empirical-CDF value in [0, 1] (Solution 5A.2)."""

    def __init__(self, Xref: np.ndarray):
        self.sorted_cols = [np.sort(Xref[:, j][np.isfinite(Xref[:, j])]) for j in range(Xref.shape[1])]

    def transform(self, X: np.ndarray) -> np.ndarray:
        out = np.empty_like(X, dtype=np.float32)
        for j, srt in enumerate(self.sorted_cols):
            if srt.size == 0:
                out[:, j] = X[:, j]
            else:
                out[:, j] = (np.searchsorted(srt, X[:, j], side="right") / srt.size).astype(np.float32)
        return out


R6_ZERO: Dict[str, Dict[str, Any]] = {}
R6_P2_ROWS: List[Dict[str, Any]] = []
PRUNE68_ROWS: List[Dict[str, Any]] = []
_t0r6p2 = time.time()
for src in ("gram", "phresh"):
    rk = f"{src}|F68R"
    run = RUNS[rk]; tgt = run["target"]; kind = run["primary"]; bp = run["best_params"][kind]
    tr_s, va_s = partition_index(src, "train"), partition_index(src, "val")
    y_tr_s, y_va_s = CLEAN[src]["y"].values[tr_s], CLEAN[src]["y"].values[va_s]
    ext_strict = np.flatnonzero(EXTERNAL_MASKS[(src, tgt)]["strict_domain_unseen"])
    y_ext = CLEAN[tgt]["y"].values[ext_strict]
    full_cols = list(FEATURE_SETS["F68R"])

    # ---- pruning policy v2 (Master V2 Blocker 7 / Solution 5B.3) -----------------------
    clf68, dom_imp68, dom_auc68, n_dom68 = fit_domain_classifier(src, tgt, "F68R", derived_seed("domclf68", rk))
    shap_imp68 = source_shap_importance(rk, "F68R")
    _sh68 = feature_shift(raw_matrix(src, _sub_rows(tr_s, 60_000, f"ps68_{src}"), "F68R"),
                          raw_matrix(tgt, _sub_rows(partition_index(tgt, "train"), 60_000, f"pt68_{tgt}"), "F68R"),
                          full_cols)
    prev68 = dict(zip(_sh68["feature"], pd.Series(_sh68["prevalence_diff"]).fillna(0.0).abs()))
    thr_imp = CFG.revision6["prune_domain_importance"]
    thr_shap = CFG.revision6["prune_source_shap"]
    thr_prev = CFG.revision6["prune_prevalence_shift"]
    prune68 = np.array([(d > thr_imp) and (s < thr_shap or float(prev68.get(n, 0.0)) > thr_prev)
                        for n, d, s in zip(full_cols, dom_imp68, shap_imp68)])
    kept68 = [n for n, p_ in zip(full_cols, prune68) if not p_]
    LEDGER.record("domain_classifier", rk, f"{src}+{tgt}", "train", "diagnostic_only", n_dom68,
                  "revision-6 policy v2: F68-R features only; no labels of either dataset")
    for n_, d_, s_, p_ in zip(full_cols, dom_imp68, shap_imp68, prune68):
        PRUNE68_ROWS.append({"run": rk, "feature": n_, "domain_importance": float(d_),
                             "source_mean_abs_shap": float(s_),
                             "prevalence_shift": float(prev68.get(n_, 0.0)), "pruned_v2": bool(p_)})
    _pruned_named = [n for n, p_ in zip(full_cols, prune68) if p_]
    print(f"{rk}: domain-classifier ROC-AUC {dom_auc68:.4f} on {n_dom68:,} feature rows; policy v2 removed "
          f"{len(_pruned_named)}/{len(full_cols)} -> {_pruned_named}")
    if src == "gram":
        _ish = full_cols.index("R_is_https")
        print(f"  -> R_is_https pruned on the Gram-sourced branch: {bool('R_is_https' in _pruned_named)} "
              f"(domain importance {float(dom_imp68[_ish]):.4f}, source |SHAP| {float(shap_imp68[_ish]):.4f}, "
              f"prevalence shift {float(prev68.get('R_is_https', 0.0)):.4f})")

    def _X68(cols, ds, idx, ranker=None):
        m = RUNS[rk]["imputer"].transform(FEATS[ds][cols].to_numpy(dtype=np.float32)[idx])
        return m if ranker is None else ranker.transform(m)

    models68: Dict[str, Dict[str, Any]] = {}
    models68["M0r6 F68-R"] = {"cols": full_cols, "ranker": None, "calibrator": None, "model": None,
                              "frozen": True, "target_info": "none", "n_features": len(full_cols)}
    _m_prune = make_model(kind, bp["params"], rk, n_estimators=bp["n_estimators"])
    _m_prune.fit(_X68(kept68, src, tr_s), y_tr_s)
    LEDGER.record("r6_phase2_fit", rk, src, "train", "fit_model", len(tr_s), "M0r6 prune-v2 (source only)")
    models68["M0r6 F68-R prune-v2"] = {"cols": kept68, "ranker": None, "calibrator": None, "model": _m_prune,
                                       "frozen": False, "target_info": "none", "n_features": len(kept68)}
    _ranker = _RankTransform(_X68(kept68, src, tr_s))
    _m_rank = make_model(kind, bp["params"], rk, n_estimators=bp["n_estimators"])
    _m_rank.fit(_ranker.transform(_X68(kept68, src, tr_s)), y_tr_s)
    LEDGER.record("r6_phase2_fit", rk, src, "train", "fit_model", len(tr_s), "M0r6 +rank (source only)")
    models68["M0r6 F68-R +rank"] = {"cols": kept68, "ranker": _ranker, "calibrator": None, "model": _m_rank,
                                    "frozen": False, "target_info": "none", "n_features": len(kept68)}
    # pseudo-label self-training on calibrated confidence only (Solution 5B.1)
    _p_va = model_proba(kind, _m_prune, _X68(kept68, src, va_s))
    _cal68, _ = select_calibrator(_p_va, y_va_s, CLEAN[src]["registered_domain"].values[va_s],
                                  f"{rk}|pseudo", CFG.calibration["cv_folds"])
    tr_t = partition_index(tgt, "train")
    y_tr_t_diag = CLEAN[tgt]["y"].values[tr_t]          # diagnostic agreement only, never fitted on
    Xtr_p, ytr_p = _X68(kept68, src, tr_s), y_tr_s
    cur_m, cur_cal = _m_prune, _cal68
    pseudo_log = []
    for it in range(CFG.revision6["pseudo_max_iter"]):
        p_t_raw = model_proba(kind, cur_m, _X68(kept68, tgt, tr_t))
        p_t = cur_cal.predict(p_t_raw)
        conf = np.maximum(p_t, 1.0 - p_t)
        idx_keep = np.flatnonzero(conf >= CFG.revision6["pseudo_threshold"])
        if len(idx_keep) > CFG.revision6["pseudo_row_cap"]:
            sel = np.argsort(-conf[idx_keep], kind="stable")[: CFG.revision6["pseudo_row_cap"]]
            idx_keep = np.sort(idx_keep[sel])
        if len(idx_keep) < 500:
            pseudo_log.append({"iter": it, "n_pseudo": int(len(idx_keep)), "note": "too few pseudo rows; stop"})
            break
        yps = (p_t[idx_keep] >= 0.5).astype(int)
        agree = float((yps == y_tr_t_diag[idx_keep]).mean())
        Xtr_p = np.vstack([Xtr_p, _X68(kept68, tgt, tr_t[idx_keep])])
        ytr_p = np.concatenate([ytr_p, yps])
        cur_m = make_model(kind, bp["params"], rk, n_estimators=bp["n_estimators"])
        cur_m.fit(Xtr_p, ytr_p)
        _p_va_it = model_proba(kind, cur_m, _X68(kept68, src, va_s))
        cur_cal, _ = select_calibrator(_p_va_it, y_va_s, CLEAN[src]["registered_domain"].values[va_s],
                                       f"{rk}|pseudo{it}", CFG.calibration["cv_folds"])
        LEDGER.record("phase2_fit", rk, tgt, "train", "fit_model", int(len(idx_keep)),
                      f"pseudo-label self-training iter {it}: calibrated confidence >= "
                      f"{CFG.revision6['pseudo_threshold']}; NO true target label used")
        pseudo_log.append({"iter": it, "n_pseudo": int(len(idx_keep)), "agreement_diagnostic": agree})
        print(f"  {rk} pseudo iter {it}: +{len(idx_keep):,} pseudo rows (agreement with true target-train "
              f"labels {agree:.3f}; diagnostic only)")
    models68["M0r6 F68-R +pseudo"] = {"cols": kept68, "ranker": None, "calibrator": cur_cal, "model": cur_m,
                                      "frozen": False, "target_info": "unlabelled target TRAIN scores only",
                                      "n_features": len(kept68)}
    R6_ZERO[rk] = {"kept68": kept68, "ranker": _ranker, "models": models68, "pseudo_log": pseudo_log,
                   "pruned": _pruned_named, "domain_auc": float(dom_auc68)}

    # ---- strict-external evaluation of all four variants --------------------------------
    for mname, spec in models68.items():
        if spec["frozen"]:
            p_ext = stage_a_p_cal(rk, tgt, ext_strict)
            p_va = stage_a_p_cal(rk, src, va_s)
        else:
            def _pred(ds, idx, _s=spec):
                X = _X68(_s["cols"], ds, idx, _s["ranker"])
                p = model_proba(kind, _s["model"], X)
                return _s["calibrator"].predict(p) if _s["calibrator"] is not None else p
            p_ext, p_va = _pred(tgt, ext_strict), _pred(src, va_s)
        thr = select_threshold(y_va_s, p_va, CFG.threshold_metric)
        met = classification_metrics(y_ext, p_ext, thr)
        R6_P2_ROWS.append({"direction": f"{CFG.datasets[src]['display']} -> {CFG.datasets[tgt]['display']}",
                           "model": mname, "n_features": spec["n_features"],
                           "target_information_used": spec["target_info"], "n_eval": len(y_ext),
                           "roc_auc": float(fast_auc(y_ext, p_ext)), "tau_source_val_mcc": float(thr),
                           "accuracy": met["accuracy"], "mcc": met["mcc"], "brier": met["brier"]})
    LOG.info("Revision-6 Phase 2 variants trained+evaluated for %s (%.0fs elapsed)", rk, time.time() - _t0r6p2)

R6_P2_TABLE = pd.DataFrame(R6_P2_ROWS)
PRUNE68_TABLE = pd.DataFrame(PRUNE68_ROWS)
display(R6_P2_TABLE.round(4))
save_table(R6_P2_TABLE, "table0E2_revision6_phase2_zero_shot_variants")
save_table(PRUNE68_TABLE, "table0D2_revision6_prune_policy_v2")

_g2dir = R6_P2_TABLE[R6_P2_TABLE["direction"].str.startswith(CFG.datasets["gram"]["display"])]
_best2 = _g2dir.loc[_g2dir["roc_auc"].idxmax()]
P2_GATE_R6 = {
    "threshold": CFG.revision6["gate2_min_external_roc_auc"],
    "best_model_gram_to_phresh": _best2["model"],
    "best_external_roc_auc_gram_to_phresh": float(_best2["roc_auc"]),
    "all_models_gram_to_phresh": _g2dir.set_index("model")["roc_auc"].round(4).to_dict(),
    "all_models_phresh_to_gram": R6_P2_TABLE[R6_P2_TABLE["direction"].str.startswith(CFG.datasets["phresh"]["display"])]
        .set_index("model")["roc_auc"].round(4).to_dict(),
    "r_is_https_pruned_on_gram_branch": bool("R_is_https" in R6_ZERO["gram|F68R"]["pruned"]),
    "passed": bool(_best2["roc_auc"] >= CFG.revision6["gate2_min_external_roc_auc"]),
}
save_json(P2_GATE_R6, DIRS["metadata"] / "phase2_revision6_gate.json")
print(json.dumps(P2_GATE_R6, indent=2, default=str))
if P2_GATE_R6["passed"]:
    print(f"\nPHASE 2 (REVISION 6) GATE PASSED: best source-only strict-external ROC-AUC = "
          f"{P2_GATE_R6['best_external_roc_auc_gram_to_phresh']:.4f} >= {P2_GATE_R6['threshold']}.")
else:
    print(f"\nPHASE 2 (REVISION 6) GATE FAILED: best source-only strict-external ROC-AUC = "
          f"{P2_GATE_R6['best_external_roc_auc_gram_to_phresh']:.4f} < {P2_GATE_R6['threshold']}. Per the plan the "
          f"representation is diagnosed as the bottleneck (see the shift/prune tables above); the failure is "
          f"reported and the pipeline continues so the multi-source and reliability phases can still be measured. "
          f"No gate is relaxed and nothing is re-tuned on the external set.")''')

CELL_D_MD = md("""## REVISION 6 PHASE 3 — Semi-supervised multi-source on F68-R: the honest 95% claim (Master V2 Phase 3, Gate 3)

The Master plan is explicit: **95% is a semi-supervised multi-source number, not a zero-shot number.** This
phase therefore uses target TRAIN labels (declared) and tunes hyper-parameters on the **target inner tune
split** (Solution 5C.2) — never on the target TEST partition:

| model | training data | notes |
|---|---|---|
| `M2r6 multi-source F68-R+ds` | source TRAIN + target TRAIN | dataset-indicator feature; target-val-tuned hyper-parameters |
| `M2r6-hybrid F68-R+ds` | as above | + CharSVD-64 (basis fitted on SOURCE TRAIN only) |
| `M4r6-hybrid F68-R+ds+char` | as above | + char-TFIDF fusion, α tuned on TARGET VALIDATION |

Decision thresholds are selected on **target validation** (MCC-optimal). **Gate 3:** best multi-source
strict-external accuracy ≥ 0.95 on **both** directions. If it fails, the plan prescribes expanding target-train
usage and reporting — not forcing the number.""")

CELL_D = code(r'''# ---------------------------------------------------------------------------
# REVISION 6 PHASE 3 — semi-supervised multi-source on F68-R (Master V2 Phase 3)
# ---------------------------------------------------------------------------
def _design_r6(ds, idx, cols, src_, with_svd):
    """Multi-source design: imputed F68-R columns + dataset indicator (+ optional source-fitted CharSVD)."""
    X = RUNS[f"{src_}|F68R"]["imputer"].transform(FEATS[ds][cols].to_numpy(dtype=np.float32)[idx])
    parts = [X, np.full(len(idx), 0.0 if ds == src_ else 1.0, dtype=np.float32)]
    if with_svd:
        parts.append(svd_matrix(src_, ds, idx))
    return np.column_stack(parts)


R6_MULTI: Dict[Tuple[str, str], Dict[str, Any]] = {}
R6_P3_ROWS: List[Dict[str, Any]] = []
R6_TUNE_ROWS: List[Dict[str, Any]] = []
_t0r6p3 = time.time()
for src in ("gram", "phresh"):
    rk = f"{src}|F68R"
    run = RUNS[rk]; tgt = run["target"]; kind = run["primary"]
    tr_s = partition_index(src, "train"); y_tr_s = CLEAN[src]["y"].values[tr_s]
    tr_t = partition_index(tgt, "train"); y_tr_t = CLEAN[tgt]["y"].values[tr_t]
    va_t = partition_index(tgt, "val"); y_va_t = CLEAN[tgt]["y"].values[va_t]
    ext_strict = np.flatnonzero(EXTERNAL_MASKS[(src, tgt)]["strict_domain_unseen"])
    y_ext = CLEAN[tgt]["y"].values[ext_strict]
    kept68 = R6_ZERO[rk]["kept68"]
    ym = np.hstack([y_tr_s, y_tr_t])

    if kind in ("lgbm", "xgb"):
        _space = CFG.models[kind]["space"]; _n_iter = CFG.models[kind]["n_iter"]
    elif kind == "rf":
        _space, _n_iter = CFG.models["rf"]["grid"], len(CFG.models["rf"]["grid"])
    else:
        _space, _n_iter = [{"C": c} for c in CFG.models["lr"]["C_grid"]], len(CFG.models["lr"]["C_grid"])

    for with_svd, label in ((False, "M2r6 multi-source F68-R+ds"), (True, "M2r6-hybrid F68-R+ds")):
        # ---- hyper-parameter search scored on the TARGET inner tune split (Solution 5C.2) ----
        fit_t = partition_index(tgt, "train", "fit"); tune_t = partition_index(tgt, "train", "tune")
        src_sub = _sub_rows(tr_s, CFG.tune_max_rows, f"r6tune_src_{src}")
        fit_t_sub = _sub_rows(fit_t, CFG.tune_max_rows, f"r6tune_tgt_{tgt}")
        tune_t_sub = _sub_rows(tune_t, CFG.tune_eval_max_rows, f"r6tuneval_{tgt}")
        Xf = np.vstack([_design_r6(src, src_sub, kept68, src, with_svd),
                        _design_r6(tgt, fit_t_sub, kept68, src, with_svd)])
        yf = np.hstack([CLEAN[src]["y"].values[src_sub], CLEAN[tgt]["y"].values[fit_t_sub]])
        Xt = _design_r6(tgt, tune_t_sub, kept68, src, with_svd)
        yt = CLEAN[tgt]["y"].values[tune_t_sub]
        best = None
        for params in sample_space(_space, _n_iter, derived_seed("r6tune", rk, label)):
            n_est = None
            if kind == "lgbm":
                m = make_model(kind, params, rk, n_estimators=CFG.models[kind]["max_estimators"])
                m.fit(Xf, yf, eval_set=[(Xt, yt)],
                      callbacks=[lgb.early_stopping(CFG.models[kind]["early_stopping_rounds"], verbose=False)])
                n_est = int(m.best_iteration_ or CFG.models[kind]["max_estimators"])
            elif kind == "xgb":
                m = make_model(kind, params, rk, n_estimators=CFG.models[kind]["max_estimators"],
                               early_stopping=CFG.models[kind]["early_stopping_rounds"])
                m.fit(Xf, yf, eval_set=[(Xt, yt)], verbose=False)
                n_est = int(m.best_iteration) + 1
            else:
                m = make_model(kind, params, rk)
                m.fit(Xf, yf)
            auc = fast_auc(yt, model_proba(kind, m, Xt))
            R6_TUNE_ROWS.append({"source": src, "model": label, "kind": kind, "params": json.dumps(params),
                                 "n_estimators": n_est, "target_tune_roc_auc": auc})
            if best is None or auc > best[0] + 1e-12:
                best = (auc, params, n_est)
        _, params6, n_est6 = best
        del Xf, Xt
        LEDGER.record("phase2_fit", rk, tgt, "train", "select_hyperparameter", len(yt),
                      f"{label}: hyper-parameters selected on the TARGET inner tune split (TRAIN labels only)")
        # ---- final multi-source fit on FULL source TRAIN + target TRAIN --------------------
        Xm = np.vstack([_design_r6(src, tr_s, kept68, src, with_svd),
                        _design_r6(tgt, tr_t, kept68, src, with_svd)])
        m6 = make_model(kind, params6, rk, n_estimators=n_est6)
        m6.fit(Xm, ym)
        del Xm
        gc.collect()
        LEDGER.record("phase2_fit", rk, f"{src}+{tgt}", "train", "fit_model", len(ym),
                      f"{label} (target-val-tuned hyper-parameters)")
        R6_MULTI[(src, label)] = {"model": m6, "kind": kind, "params": params6, "cols": kept68,
                                  "with_svd": with_svd, "n_train": int(len(ym)),
                                  "target_tune_auc": float(best[0])}
        LOG.info("Revision-6 Phase 3: %s fitted (target-tune AUC %.4f, %.0fs elapsed)",
                 label, best[0], time.time() - _t0r6p3)

    # ---- M4r6-hybrid: char-TFIDF fusion with alpha tuned on TARGET VALIDATION --------------
    m2h = R6_MULTI[(src, "M2r6-hybrid F68-R+ds")]
    p_va_struct = model_proba(m2h["kind"], m2h["model"], _design_r6(tgt, va_t, m2h["cols"], src, True))
    p_va_char = char_proba(CHAR_MODELS[src]["bundle_B"], CLEAN[tgt]["url_raw"].values[va_t])
    best_alpha, best_auc = 1.0, -1.0
    for a_ in CFG.char_model["fusion_alpha_grid"]:
        auc_ = fast_auc(y_va_t, a_ * p_va_struct + (1.0 - a_) * p_va_char)
        if auc_ > best_auc + 1e-12:
            best_alpha, best_auc = float(a_), float(auc_)
    R6_MULTI[(src, "M4r6-hybrid F68-R+ds+char")] = {"struct": m2h, "alpha": best_alpha, "kind": kind,
                                                    "target_val_fusion_auc": best_auc}
    print(f"{rk}: M4r6-hybrid fusion alpha = {best_alpha} (target-val fused AUC {best_auc:.4f})")

    # ---- strict-external evaluation at TARGET-VALIDATION MCC thresholds --------------------
    p_struct_va = model_proba(m2h["kind"], m2h["model"], _design_r6(tgt, va_t, m2h["cols"], src, True))
    p_struct_ext = model_proba(m2h["kind"], m2h["model"], _design_r6(tgt, ext_strict, m2h["cols"], src, True))
    p_m2_va = model_proba(R6_MULTI[(src, "M2r6 multi-source F68-R+ds")]["kind"],
                          R6_MULTI[(src, "M2r6 multi-source F68-R+ds")]["model"],
                          _design_r6(tgt, va_t, kept68, src, False))
    p_m2_ext = model_proba(R6_MULTI[(src, "M2r6 multi-source F68-R+ds")]["kind"],
                           R6_MULTI[(src, "M2r6 multi-source F68-R+ds")]["model"],
                           _design_r6(tgt, ext_strict, kept68, src, False))
    p_char_ext = char_proba(CHAR_MODELS[src]["bundle_B"], CLEAN[tgt]["url_raw"].values[ext_strict])
    _preds = {"M2r6 multi-source F68-R+ds": (p_m2_va, p_m2_ext),
              "M2r6-hybrid F68-R+ds": (p_struct_va, p_struct_ext),
              "M4r6-hybrid F68-R+ds+char": (best_alpha * p_struct_va + (1.0 - best_alpha) * p_va_char,
                                            best_alpha * p_struct_ext + (1.0 - best_alpha) * p_char_ext)}
    for label, (p_va, p_ext) in _preds.items():
        thr = select_threshold(y_va_t, p_va, CFG.threshold_metric)      # TARGET VALIDATION threshold
        met = classification_metrics(y_ext, p_ext, thr)
        R6_P3_ROWS.append({"direction": f"{CFG.datasets[src]['display']} -> {CFG.datasets[tgt]['display']}",
                           "model": label, "n_train": R6_MULTI[(src, label)].get("n_train",
                           R6_MULTI[(src, "M2r6-hybrid F68-R+ds")]["n_train"]),
                           "target_information_used": "target TRAIN labels + target VAL tuning (declared)",
                           "n_eval": len(y_ext), "tau_target_val_mcc": float(thr),
                           "roc_auc": float(fast_auc(y_ext, p_ext)), "accuracy": met["accuracy"],
                           "precision": met["precision"], "recall": met["recall"], "f1": met["f1"],
                           "mcc": met["mcc"], "brier": met["brier"]})
    LOG.info("Revision-6 Phase 3 evaluated for %s (%.0fs elapsed)", rk, time.time() - _t0r6p3)

R6_P3_TABLE = pd.DataFrame(R6_P3_ROWS)
display(R6_P3_TABLE.round(4))
save_table(R6_P3_TABLE, "table0E3_revision6_phase3_multisource")
save_table(pd.DataFrame(R6_TUNE_ROWS), "table0E4_revision6_target_val_tuning")

_g3 = {}
for d_ in sorted(R6_P3_TABLE["direction"].unique()):
    sub = R6_P3_TABLE[R6_P3_TABLE["direction"] == d_]
    b = sub.loc[sub["accuracy"].idxmax()]
    _g3[d_] = {"best_model": b["model"], "best_external_accuracy": float(b["accuracy"]),
               "roc_auc": float(b["roc_auc"]), "tau_target_val_mcc": float(b["tau_target_val_mcc"])}
P3_GATE_R6 = {
    "threshold": CFG.revision6["gate3_min_external_accuracy"],
    "per_direction": _g3,
    "passed_both_directions": bool(all(v["best_external_accuracy"] >= CFG.revision6["gate3_min_external_accuracy"]
                                       for v in _g3.values())),
    "claim_framing": "semi-supervised multi-source (target TRAIN labels + target-VAL tuning); NOT zero-shot",
}
save_json(P3_GATE_R6, DIRS["metadata"] / "phase3_revision6_gate.json")
print(json.dumps(P3_GATE_R6, indent=2, default=str))
if P3_GATE_R6["passed_both_directions"]:
    print(f"\nPHASE 3 (REVISION 6) GATE PASSED: multi-source external accuracy >= "
          f"{P3_GATE_R6['threshold']} on BOTH directions. The 95% claim is stated as a SEMI-SUPERVISED "
          f"multi-source result, exactly as the plan prescribes.")
else:
    print(f"\nPHASE 3 (REVISION 6) GATE NOT MET: best external accuracy per direction shown above. Per the plan, "
          f"target-train usage is already maximal (full TRAIN partitions) and hyper-parameters are target-val "
          f"tuned; the shortfall is REPORTED (reduced-scale execution and/or dataset-pair difficulty), not forced.")''')

print("cells C/D ready")
