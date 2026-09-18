
# ============================================================================
# NEW CELL G — REVISION 6 PHASE 6 (inserted after the statistical-utilities cell)
# ============================================================================
CELL_G_MD = md("""## REVISION 6 PHASE 6 — DTS repair: stratum routing, prior correction and the external AURC gate (Master V2 Blocker 2, Solutions 2A/2B/2C, Gate 6)

Revision 5's decision layer collapsed to confidence because on source validation confidence is already
near-sufficient. Three repairs are implemented exactly as the plan prescribes:

- **2A StratumDTS** — route by confidence (`C < 0.9` vs `C ≥ 0.9`) and fit a separate
  `P(correct | C, ERS)` logistic per stratum, capturing the stratum-conditional effect both earlier
  revisions hinted at;
- **2B prior-corrected DTS** — reweight the source-validation fit by the **Saerens-Latinne-Decaestecker EM**
  target-prevalence estimate computed on the *unlabelled* strict-external confidence distribution (no target
  labels);
- **2C external evaluation** — the decision layer is judged where it matters: **risk-coverage AURC on the
  strict-external population** (paired bootstrap vs confidence), not only validation OOF Brier.

**Gate 6:** at least one DTS variant achieves lower external AURC than confidence-only. If not, the negative
result is reported.""")

CELL_G = code(r'''# ---------------------------------------------------------------------------
# REVISION 6 PHASE 6 — StratumDTS + prior-corrected DTS + external AURC gate
# ---------------------------------------------------------------------------
class StratumDTS:
    """Route by confidence; fit a separate logistic P(correct | C, ERS) per stratum (plan Solution 2A)."""

    def __init__(self, split: float = None):
        self.split = float(split or CFG.revision6["dts_stratum_C_split"])
        self.models_: Dict[str, Optional[DecisionTrustModel]] = {}

    def fit(self, C, E, y, groups, key, k=5):
        C, E, y = np.asarray(C, float), np.asarray(E, float), np.asarray(y).astype(int)
        groups = np.asarray(groups)
        self.models_ = {}
        for lo, hi, name in ((0.0, self.split, "low_C"), (self.split, 1.01, "high_C")):
            m = (C >= lo) & (C < hi)
            if m.sum() >= CFG.revision6["dts_stratum_min_rows"] and 5 <= (1 - y[m]).sum() <= m.sum() - 5:
                self.models_[name] = DecisionTrustModel("logistic_2f").fit(
                    C[m], E[m], y[m], groups[m], f"{key}|{name}", k)
            else:
                self.models_[name] = None
                LOG.warning("StratumDTS[%s]: stratum %s not fittable (n=%d, errors=%d) -> routed to confidence",
                            key, name, int(m.sum()), int((1 - y[m]).sum()))
        return self

    def predict(self, C, E):
        C, E = np.asarray(C, float), np.asarray(E, float)
        out = np.clip(C, 0, 1).copy()
        for lo, hi, name in ((0.0, self.split, "low_C"), (self.split, 1.01, "high_C")):
            m = (C >= lo) & (C < hi)
            if self.models_.get(name) is not None and m.any():
                out[m] = self.models_[name].predict(C[m], E[m])
        return out


def fit_prior_corrected_dts(C, E, y, groups, conf_tgt_unlabelled, prior_src, key, k=5):
    """Solution 2B: source-validation rows reweighted by the EM target prior (no target labels used)."""
    prior_tgt = estimate_target_prior(np.asarray(conf_tgt_unlabelled, dtype=np.float64), prior_source=prior_src)
    w = np.where(np.asarray(C, dtype=np.float64) >= 0.5,
                 prior_tgt / prior_src, (1 - prior_tgt) / (1 - prior_src))
    w = w / w.mean()
    X = _dts_design(np.asarray(C, float), np.asarray(E, float), "logistic_2f")
    y = np.asarray(y).astype(int)
    folds = domain_folds(np.asarray(groups), k, key)
    best, bestC = np.inf, CFG.dts_l2_Cs[0]
    for c_ in CFG.dts_l2_Cs:
        oof = np.zeros(len(y))
        for f in range(k):
            trm, tem = folds != f, folds == f
            if tem.sum() == 0 or len(np.unique(y[trm])) < 2:
                oof[tem] = y[trm].mean() if trm.any() else 0.5
                continue
            oof[tem] = LogisticRegression(C=c_, penalty="l2", max_iter=5000).fit(
                X[trm], y[trm], sample_weight=w[trm]).predict_proba(X[tem])[:, 1]
        b_ = brier_score_loss(y, np.clip(oof, 0, 1), sample_weight=w)
        if b_ < best:
            best, bestC = b_, c_
    lr = LogisticRegression(C=bestC, penalty="l2", max_iter=5000).fit(X, y, sample_weight=w)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
        lr.predict_proba(X)[:, 1], y, sample_weight=w)

    def _predict(C_, E_):
        p = lr.predict_proba(_dts_design(np.asarray(C_, float), np.asarray(E_, float), "logistic_2f"))[:, 1]
        return np.clip(iso.predict(p), 0, 1)

    return _predict, float(prior_tgt), float(bestC)


R6_DTS_ROWS: List[Dict[str, Any]] = []
R6_DTS_MODELS: Dict[Tuple[str, str], Any] = {}
for rk in RUN_KEYS:
    run = RUNS[rk]
    V, E_pop = POPS[(rk, "val")], POPS[(rk, "ext")]
    okv = V["ers_defined"] & np.isfinite(V["ERS"])
    oke = E_pop["ers_defined"] & np.isfinite(E_pop["ERS"])
    Cv, Ev = V["C"][okv], V["ERS"][okv]
    yv = (V["yhat"][okv] == V["y"][okv]).astype(int)
    gv = CLEAN[run["source"]]["registered_domain"].values[np.asarray(V["rows"]).astype(np.int64)[okv]]
    Ce, Ee = E_pop["C"][oke], E_pop["ERS"][oke]
    ye = (E_pop["yhat"][oke] == E_pop["y"][oke]).astype(int)
    prior_src = float(CLEAN[run["source"]]["y"].values[partition_index(run["source"], "train")].mean())
    # unlabelled target scores for the EM prior: strict-external confidence distribution (no labels)
    ext_all = np.flatnonzero(EXTERNAL_MASKS[(run["source"], run["target"])]["strict_domain_unseen"])
    p_ext_all = stage_a_p_cal(rk, run["target"], ext_all)
    conf_ext_all = np.maximum(p_ext_all, 1.0 - p_ext_all)

    stratum = StratumDTS().fit(Cv, Ev, yv, gv, f"{rk}|stratum", CFG.dts["cv_folds"])
    dts_prior_fn, prior_tgt_hat, prior_C = fit_prior_corrected_dts(
        Cv, Ev, yv, gv, conf_ext_all, prior_src, f"{rk}|prior", CFG.dts["cv_folds"])
    R6_DTS_MODELS[(rk, "stratum")] = stratum
    R6_DTS_MODELS[(rk, "prior")] = {"predict": dts_prior_fn, "prior_tgt_hat": prior_tgt_hat, "l2_C": prior_C}

    scores = {"C (confidence)": Ce,
              "DTS_2f (revision-5)": E_pop["DTS"][oke],
              "DTS_stratum (revision-6)": stratum.predict(Ce, Ee),
              "DTS_prior_corrected (revision-6)": dts_prior_fn(Ce, Ee)}
    a_c = aurc(Ce, ye)
    for name, s_ in scores.items():
        point, ci, p_ = paired_bootstrap_diff(aurc, s_, Ce, ye, CFG.stats["bootstrap_B"],
                                              derived_seed("r6dts", rk, name))
        R6_DTS_ROWS.append({"run": rk, "score": name, "AURC": float(aurc(s_, ye)),
                            "dAURC_vs_C": float(point), "ci_low": float(ci[0]), "ci_high": float(ci[1]),
                            "p_boot": float(p_), "n_ext": int(oke.sum()),
                            "estimated_target_prior": prior_tgt_hat if "prior" in name else np.nan})
    print(f"{rk}: external AURC C={a_c:.4f} | stratum "
          f"{aurc(scores['DTS_stratum (revision-6)'], ye):.4f} | prior-corrected "
          f"{aurc(scores['DTS_prior_corrected (revision-6)'], ye):.4f} (EM prior hat {prior_tgt_hat:.3f})")

R6_DTS_TABLE = pd.DataFrame(R6_DTS_ROWS)
display(R6_DTS_TABLE.round(5))
save_table(R6_DTS_TABLE, "table0G2_revision6_phase6_dts_external_aurc")

_variants6 = ["DTS_stratum (revision-6)", "DTS_prior_corrected (revision-6)"]
_sub6 = R6_DTS_TABLE[R6_DTS_TABLE["score"].isin(_variants6)]
_best6 = _sub6.loc[_sub6["dAURC_vs_C"].idxmin()]
P6_GATE_R6 = {
    "criterion": "at least one DTS variant achieves lower EXTERNAL AURC than confidence-only (Master V2 Gate 6)",
    "best_variant": _best6["score"], "best_run": _best6["run"],
    "best_dAURC": float(_best6["dAURC_vs_C"]),
    "best_ci": [float(_best6["ci_low"]), float(_best6["ci_high"])],
    "passed": bool((_sub6["dAURC_vs_C"] < 0).any()),
}
save_json(P6_GATE_R6, DIRS["metadata"] / "phase6_revision6_gate.json")
print(json.dumps(P6_GATE_R6, indent=2, default=str))
if P6_GATE_R6["passed"]:
    print("\nPHASE 6 (REVISION 6) GATE PASSED: a DTS variant achieves lower external AURC than "
          "confidence-only (details above; the CI tells whether the improvement is decisive).")
else:
    print("\nPHASE 6 (REVISION 6) GATE NOT PASSED: no DTS variant achieves lower external AURC than "
          "confidence-only. This is REPORTED as a negative result: on this dataset pair the decision layer "
          "cannot extract decision-relevant information from ERS beyond confidence, even with stratum "
          "routing and prior correction.")''')

# ============================================================================
# NEW CELL H — REVISION 6 PHASE 7 (reversal diagnosis)
# ============================================================================
CELL_H_MD = md("""## REVISION 6 PHASE 7 — Reversal diagnosis (Master V2 Blocker 4, Gate 7)

On external data Revision 5 found high-ERS predictions carrying *more* errors than low-ERS predictions — the
sign-reversed pattern that would invalidate the paper if left unexplained. The plan's three candidate
mechanisms are decomposed per run on the strict-external population:

- **A — class confounding:** does mean ERS differ by (true / predicted) class by > 0.05?
- **B — class-asymmetric accuracy:** FNR vs FPR (target accuracy asymmetric by > 2×)?
- **C — faithfulness/consensus artifact:** do F or M differ by class by > 0.05?

plus the decisive test: **does the reversal survive conditioning on class?** If the high-ERS-higher-error
pattern disappears within true/predicted class, the reversal is *explained by class confounding*. If it
persists within class, it is reported as a **genuine negative**: explanation stability does not imply
decision reliability on external data.""")

CELL_H = code(r'''# ---------------------------------------------------------------------------
# REVISION 6 PHASE 7 — reversal decomposition (Master V2 Solution 4)
# ---------------------------------------------------------------------------
def diagnose_reversal_r6(P: Dict[str, Any], thr: Dict[str, float]) -> Dict[str, Any]:
    ok = P["ers_defined"] & np.isfinite(P["ERS"])
    y, yh = P["y"][ok], P["yhat"][ok]
    C, E = P["C"][ok], P["ERS"][ok]
    F_, M_ = P["F"][ok], P["M"][ok]
    out: Dict[str, Any] = {"n": int(ok.sum())}
    out["delta_ERS_by_true_class"] = float(E[y == 1].mean() - E[y == 0].mean()) if (y == 1).any() and (y == 0).any() else np.nan
    out["delta_ERS_by_pred_class"] = float(E[yh == 1].mean() - E[yh == 0].mean()) if (yh == 1).any() and (yh == 0).any() else np.nan
    out["FNR"] = float(((yh == 0) & (y == 1)).sum() / max((y == 1).sum(), 1))
    out["FPR"] = float(((yh == 1) & (y == 0)).sum() / max((y == 0).sum(), 1))
    out["FNR_FPR_asymmetry_gt_2x"] = bool(max(out["FNR"], out["FPR"]) > 2 * max(min(out["FNR"], out["FPR"]), 1e-9))
    out["delta_F_by_class"] = float(F_[y == 1].mean() - F_[y == 0].mean()) if (y == 1).any() and (y == 0).any() else np.nan
    out["delta_M_by_class"] = float(M_[y == 1].mean() - M_[y == 0].mean()) if (y == 1).any() and (y == 0).any() else np.nan
    hi, lo = thr["ERS_high"], thr["ERS_low"]
    rows = []
    for cls, cls_name in ((0, "benign"), (1, "phishing")):
        for scope_name, scope in (("true_class", y == cls), ("pred_class", yh == cls)):
            m = scope
            m_hi = m & (E >= hi)
            m_lo = m & (E < lo)
            if m_hi.sum() >= 10 and m_lo.sum() >= 10:
                err_hi = float((yh[m_hi] != y[m_hi]).mean())
                err_lo = float((yh[m_lo] != y[m_lo]).mean())
                rows.append({"scope": scope_name, "class": cls_name, "n_high_ERS": int(m_hi.sum()),
                             "err_high_ERS": err_hi, "n_low_ERS": int(m_lo.sum()), "err_low_ERS": err_lo,
                             "delta_err_high_minus_low": err_hi - err_lo})
    out["within_class"] = rows
    deltas = [r_["delta_err_high_minus_low"] for r_ in rows if r_["scope"] == "true_class"]
    out["median_within_true_class_delta"] = float(np.median(deltas)) if deltas else np.nan
    return out


R6_REV_ROWS: List[Dict[str, Any]] = []
R6_REV_DETAIL: Dict[str, Any] = {}
for rk in RUN_KEYS:
    P = POPS[(rk, "ext")]
    d = diagnose_reversal_r6(P, RUNS[rk]["ers_thresholds"])
    R6_REV_DETAIL[rk] = d
    R6_REV_ROWS.append({"run": rk, "population": "external STRICT", **{k_: v_ for k_, v_ in d.items()
                                                                        if k_ != "within_class"}})
    for r_ in d["within_class"]:
        R6_REV_ROWS.append({"run": rk, "population": f"external | within {r_['scope']}={r_['class']}",
                            "n_high_ERS": r_["n_high_ERS"], "err_high_ERS": r_["err_high_ERS"],
                            "n_low_ERS": r_["n_low_ERS"], "err_low_ERS": r_["err_low_ERS"],
                            "delta_err_high_minus_low": r_["delta_err_high_minus_low"]})
R6_REV_TABLE = pd.DataFrame(R6_REV_ROWS)
display(R6_REV_TABLE.round(4))
save_table(R6_REV_TABLE, "table0H2_revision6_phase7_reversal_decomposition")

# verdict: does the external reversal (high ERS -> more errors) survive conditioning on class?
_persist, _flip = [], []
for rk in RUN_KEYS:
    d_ = R6_REV_DETAIL[rk]["median_within_true_class_delta"]
    if np.isfinite(d_):
        (_persist if d_ > 0.02 else _flip).append((rk, float(d_)))
if len(_persist) >= max(1, len(RUN_KEYS) // 2):
    P7_STATUS = "genuine_negative_reported"
    P7_TEXT = ("The external reversal PERSISTS within true class: high-ERS predictions carry higher error than "
               "low-ERS predictions even after conditioning on class. Per the plan this is reported as a genuine "
               "negative: explanation stability does not imply decision reliability on external data, and "
               "high-C/low-ERS is NOT presented as a risk signal on external populations.")
elif len(_flip) >= max(1, len(RUN_KEYS) // 2):
    P7_STATUS = "explained_by_class_confounding"
    P7_TEXT = ("The external reversal DISAPPEARS (or inverts) after conditioning on class: ERS was acting as a "
               "proxy for the predicted class, not as an independent correctness signal. Per the plan the "
               "correct claim is that on external data ERS does not independently add correctness information "
               "beyond what the class already explains.")
else:
    P7_STATUS = "mixed_reported"
    P7_TEXT = ("The within-class decomposition is mixed across runs: the reversal persists within class for some "
               "runs and disappears for others. Both patterns are reported per run above; no universal claim is "
               "made either way.")
P7_GATE_R6 = {"status": P7_STATUS,
              "runs_persisting_within_class": [{"run": k_, "median_delta": v_} for k_, v_ in _persist],
              "runs_explained_within_class": [{"run": k_, "median_delta": v_} for k_, v_ in _flip],
              "class_confounding_candidates": {rk: {"delta_ERS_by_true_class": R6_REV_DETAIL[rk]["delta_ERS_by_true_class"],
                                                     "delta_ERS_by_pred_class": R6_REV_DETAIL[rk]["delta_ERS_by_pred_class"],
                                                     "delta_F_by_class": R6_REV_DETAIL[rk]["delta_F_by_class"],
                                                     "delta_M_by_class": R6_REV_DETAIL[rk]["delta_M_by_class"],
                                                     "FNR": R6_REV_DETAIL[rk]["FNR"], "FPR": R6_REV_DETAIL[rk]["FPR"]}
                                                for rk in RUN_KEYS},
              "verdict_text": P7_TEXT}
save_json(P7_GATE_R6, DIRS["metadata"] / "phase7_revision6_gate.json")
print(json.dumps({k_: v_ for k_, v_ in P7_GATE_R6.items() if k_ != "class_confounding_candidates"},
                 indent=2, default=str))
print("\n" + P7_TEXT)''')

# ============================================================================
# NEW CELL I — REVISION 6 PHASE 8 (shift strata)
# ============================================================================
CELL_I_MD = md("""## REVISION 6 PHASE 8 — Test B restated as a shift-conditioned test (Master V2 Blocker 3, Gate 8)

Instead of testing “ERS adds beyond confidence everywhere”, the LRT
`Error ~ logit C` vs `Error ~ logit C + logit ERS` is run **per pre-registered shift stratum** of the
strict-external population (Solution 3), Holm-corrected within each run's 3-stratum family:

1. **protocol-shift** — URLs whose `R_is_https` value is the *source-minority* value (the HTTPS-shortcut
   stress stratum; on Gram→Phresh this is HTTP URLs under a 94%-HTTPS target);
2. **path-structure-shift** — URLs whose `R_path_ratio` lies outside the source-TRAIN [q10, q90] support;
3. **lexicon-shift** — URLs whose semantic token counts (`R_auth_token_count` / `R_suspicious_token_count`)
   lie outside the source-TRAIN [q10, q90] support.

**Gate 8 is a reporting gate:** the deliverable is the honest map of *where* ERS adds beyond confidence and
where it does not. No universal effect is claimed.""")

CELL_I = code(r'''# ---------------------------------------------------------------------------
# REVISION 6 PHASE 8 — LRT per shift stratum, Holm-corrected within family
# ---------------------------------------------------------------------------
_R68_REF_ROWS = {s_: _sub_rows(partition_index(s_, "train"), 60_000, f"r6strata_{s_}")
                 for s_ in ("gram", "phresh")}
_R68_STRATA_FEATURES = {"protocol_shift": ["R_is_https"], "path_structure_shift": ["R_path_ratio"],
                        "lexicon_shift": ["R_auth_token_count", "R_suspicious_token_count"]}


def _feat_vals(ds, uids, col):
    return FEATS[ds][col].to_numpy(dtype=np.float64)[np.asarray(uids).astype(np.int64)]


R6_STRATA_ROWS: List[Dict[str, Any]] = []
for rk in RUN_KEYS:
    run = RUNS[rk]; src, tgt = run["source"], run["target"]
    P = POPS[(rk, "ext")]
    ok = P["ers_defined"] & np.isfinite(P["ERS"])
    y, yh = P["y"][ok], P["yhat"][ok]
    C, E = P["C"][ok], P["ERS"][ok]
    err = (yh != y).astype(int)
    ref = _R68_REF_ROWS[src]
    strata_masks: Dict[str, np.ndarray] = {}
    v_src = _feat_vals(src, ref, "R_is_https")
    p_src = float(np.nanmean(v_src))
    v_tgt = _feat_vals(tgt, P["rows"], "R_is_https")[ok]
    strata_masks["protocol_shift"] = ((v_tgt == 1) & (p_src < 0.5)) | ((v_tgt == 0) & (p_src >= 0.5))
    v_src = _feat_vals(src, ref, "R_path_ratio")
    lo_, hi_ = np.nanquantile(v_src, [0.10, 0.90])
    v_tgt = _feat_vals(tgt, P["rows"], "R_path_ratio")[ok]
    strata_masks["path_structure_shift"] = (v_tgt < lo_) | (v_tgt > hi_)
    m_lex = np.zeros(int(ok.sum()), dtype=bool)
    for col in _R68_STRATA_FEATURES["lexicon_shift"]:
        v_src = _feat_vals(src, ref, col)
        lo_, hi_ = np.nanquantile(v_src, [0.10, 0.90])
        v_tgt = _feat_vals(tgt, P["rows"], col)[ok]
        m_lex |= (v_tgt < lo_) | (v_tgt > hi_)
    strata_masks["lexicon_shift"] = m_lex

    pvals, rows = [], []
    for sname, mask in strata_masks.items():
        n_s = int(mask.sum())
        if n_s >= 50 and err[mask].sum() >= 5 and (1 - err[mask]).sum() >= 5:
            lrt = logistic_lrt(err[mask], base=[_logit(C[mask])], add=[_logit(E[mask])])
        else:
            lrt = {"lr_stat": np.nan, "p": np.nan, "coef_added": np.nan, "note": "stratum too small or degenerate"}
        pvals.append(lrt["p"])
        rows.append({"run": rk, "stratum": sname, "n": n_s,
                     "share_pct": 100.0 * n_s / max(int(ok.sum()), 1),
                     "error_rate": float(err[mask].mean()) if n_s else np.nan,
                     "lr_stat": lrt["lr_stat"], "p_unadj": lrt["p"], "coef_ERS": lrt["coef_added"],
                     "note": lrt.get("note", "")})
    ph = holm(np.array(pvals, dtype=float))
    for r_, p_ in zip(rows, ph):
        r_["p_holm"] = float(p_) if np.isfinite(p_) else np.nan
        r_["significant_holm_0.05"] = bool(np.isfinite(p_) and p_ < CFG.stats["alpha"])
    R6_STRATA_ROWS.extend(rows)

R6_STRATA_TABLE = pd.DataFrame(R6_STRATA_ROWS)
display(R6_STRATA_TABLE.round(5))
save_table(R6_STRATA_TABLE, "table0I2_revision6_phase8_shift_strata_lrt")

_sig = R6_STRATA_TABLE[R6_STRATA_TABLE["significant_holm_0.05"] == True]  # noqa: E712
P8_REPORT_R6 = {
    "gate_type": "reporting gate (Master V2 Gate 8): report where ERS adds; no universal claim",
    "n_run_stratum_tests": int(len(R6_STRATA_TABLE)),
    "n_significant_holm": int(len(_sig)),
    "significant_cells": _sig[["run", "stratum", "n", "error_rate", "lr_stat", "p_holm", "coef_ERS"]]
        .to_dict(orient="records"),
    "where_ers_adds": sorted(set(_sig["stratum"].tolist())) if len(_sig) else [],
    "where_ers_does_not_add": sorted(set(R6_STRATA_TABLE["stratum"]) - set(_sig["stratum"].tolist()))
        if len(_sig) < len(R6_STRATA_TABLE) else [],
}
save_json(P8_REPORT_R6, DIRS["metadata"] / "phase8_revision6_report.json")
print(json.dumps(P8_REPORT_R6, indent=2, default=str))
print("\nPHASE 8 (REVISION 6): the stratum-level map above states exactly where ERS carries information "
      "beyond confidence under shift and where it does not. No universal ERS-beyond-confidence claim is "
      "made; this is the shift-conditioned restatement of Test B the plan prescribes.")''')

print("cells G/H/I ready")
