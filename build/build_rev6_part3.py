
# ============================================================================
# NEW CELL E — REVISION 6 PHASE 4 (inserted before Section 31 markdown)
# ============================================================================
CELL_E_MD = md("""## REVISION 6 PHASE 4 — Robustness repair: perturbation augmentation (Master V2 Blocker 6, Solutions 6A/6B)

Revision 5 measured flip rates of 8–57% on identity-preserving families — a genuine model-robustness
failure that also contaminates ERS (it is evaluated on a prediction-unstable model). The repair is
**training-time augmentation of the SOURCE-ONLY branch** (never the target):

- **6A identity-preserving:** P1 (case normalisation) + P2 (percent-encoding of unreserved characters)
  perturbations of source TRAIN URLs, labels unchanged, sample weight 0.30;
- **6B stress:** P5 (subdomain insertion) + P6 (path padding) + P7 (query padding) — registered-domain-
  preserving, labels unchanged, sample weight 0.30.

The augmented model is then re-scored on the strict-external XAI sample under P3, P5, P6 and P7.
**Gate 4:** P3 dot-segment flip rate ≤ 15% **and** P5/P6/P7 flip rates ≤ 15% (augmented model, external).""")

CELL_E = code(r'''# ---------------------------------------------------------------------------
# REVISION 6 PHASE 4 — robustness repair via source-side augmentation (Solutions 6A/6B)
# ---------------------------------------------------------------------------
R6_AUG: Dict[str, Dict[str, Any]] = {}
R6_P4_ROWS: List[Dict[str, Any]] = []
AUG_ID_FAMILIES = ("P1_case_normalization", "P2_pct_encode_unreserved")
AUG_STRESS_FAMILIES = ("P5_subdomain_insertion", "P6_path_padding", "P7_query_padding")
GATE4_FAMILIES = ("P3_dot_segment",) + AUG_STRESS_FAMILIES
_t0r6p4 = time.time()
for src in ("gram", "phresh"):
    rk = f"{src}|F68R"
    run = RUNS[rk]; tgt = run["target"]; kind = run["primary"]; bp = run["best_params"][kind]
    tr_s = partition_index(src, "train"); y_tr_s = CLEAN[src]["y"].values[tr_s]
    kept68 = R6_ZERO[rk]["kept68"]
    base_spec = R6_ZERO[rk]["models"]["M0r6 F68-R prune-v2"]
    f68_final = list(FEATURE_SETS["F68R"])
    _kept_pos = [f68_final.index(c) for c in kept68]

    sub = _sub_rows(tr_s, CFG.revision6["aug_max_urls"], f"aug_{src}")
    urls_sub = CLEAN[src]["url_raw"].values[sub]

    def _aug_xy(recs, fam_tag):
        v = recs[recs["valid"]]
        if len(v) > CFG.revision6["aug_max_rows"]:
            v = v.iloc[np.sort(np.random.default_rng(derived_seed("augcap", src, fam_tag))
                               .choice(len(v), CFG.revision6["aug_max_rows"], replace=False))]
        Xa = RUNS[rk]["imputer"].transform(
            extract_features_frame(v["generated_url"].values, 1, CFG.feature_extraction_chunk, "f54r")[kept68]
            .to_numpy(dtype=np.float32))
        ya = CLEAN[src]["y"].values[v["source_uid"].values]
        return Xa, ya, int(len(v))

    rec_id = generate_perturbations(sub, urls_sub,
                                    {f: CFG.perturbation["identity_families"][f] for f in AUG_ID_FAMILIES},
                                    derived_seed("aug_id", src))
    Xid, yid, n_id = _aug_xy(rec_id, "id")
    rec_st = generate_perturbations(sub, urls_sub,
                                    {f: CFG.perturbation["stress_families"][f] for f in AUG_STRESS_FAMILIES},
                                    derived_seed("aug_st", src))
    Xst, yst, n_st = _aug_xy(rec_st, "st")

    X_aug = np.vstack([RUNS[rk]["imputer"].transform(FEATS[src][kept68].to_numpy(dtype=np.float32)[tr_s]),
                       Xid, Xst])
    y_aug = np.concatenate([y_tr_s, yid, yst])
    w_aug = np.concatenate([np.ones(len(tr_s), dtype=np.float32),
                            np.full(len(Xid), CFG.revision6["augment_weight"], dtype=np.float32),
                            np.full(len(Xst), CFG.revision6["augment_weight"], dtype=np.float32)])
    m_aug = make_model(kind, bp["params"], rk, n_estimators=bp["n_estimators"])
    m_aug.fit(X_aug, y_aug, sample_weight=w_aug)
    LEDGER.record("r6_phase4_fit", rk, src, "train", "fit_model", int(len(y_aug)),
                  f"source TRAIN + {n_id} identity-augmented + {n_st} stress-augmented rows "
                  f"(weight {CFG.revision6['augment_weight']}); perturbations applied to SOURCE URLs only")
    del X_aug
    gc.collect()

    # ---- flip rates on the strict-external XAI sample (paired with the unaugmented model) ----
    P_ext = POPS[(rk, "ext")]
    X_ext_orig = P_ext["X"][:, _kept_pos]
    _ext_uids = np.asarray(P_ext["uids"], dtype=object)   # string record-ids, as in the stability stage
    yhat_aug = (model_proba(kind, m_aug, X_ext_orig) >= 0.5)
    yhat_base = (model_proba(kind, base_spec["model"], X_ext_orig) >= 0.5)
    pos_map = pd.Series(np.arange(len(_ext_uids)), index=_ext_uids)
    for fam in GATE4_FAMILIES:
        fam_cfg = (CFG.perturbation["identity_families"].get(fam)
                   or CFG.perturbation["stress_families"][fam])
        rec = generate_perturbations(_ext_uids, P_ext["urls"], {fam: fam_cfg},
                                     derived_seed("r6flip", src, fam))
        v = rec[rec["valid"]]
        pos = pos_map.reindex(v["source_uid"].values).to_numpy(dtype=np.int64)
        Xp = RUNS[rk]["imputer"].transform(
            extract_features_frame(v["generated_url"].values, 1, CFG.feature_extraction_chunk, "f54r")[kept68]
            .to_numpy(dtype=np.float32))
        yp_aug = (model_proba(kind, m_aug, Xp) >= 0.5)
        yp_base = (model_proba(kind, base_spec["model"], Xp) >= 0.5)
        for tag, yp, yh0 in (("augmented", yp_aug, yhat_aug), ("unaugmented", yp_base, yhat_base)):
            fl = pd.DataFrame({"pos": pos, "flip": (yp != yh0[pos]).astype(bool)})
            agg = fl.groupby("pos")["flip"].any()
            parents_with = np.zeros(len(P_ext["uids"]), dtype=bool)
            parents_with[agg.index.values] = True
            parents_flip = np.zeros(len(P_ext["uids"]), dtype=bool)
            parents_flip[agg.index.values] = agg.values
            rate = float(parents_flip[parents_with].mean()) if parents_with.any() else float("nan")
            R6_P4_ROWS.append({"direction": f"{CFG.datasets[src]['display']} -> {CFG.datasets[tgt]['display']}",
                               "family": fam, "model": tag, "n_parents_with_valid": int(parents_with.sum()),
                               "flip_rate": rate})
    LOG.info("Revision-6 Phase 4 done for %s (identity %d rows, stress %d rows, %.0fs elapsed)",
             rk, n_id, n_st, time.time() - _t0r6p4)

R6_P4_TABLE = pd.DataFrame(R6_P4_ROWS)
display(R6_P4_TABLE.round(4))
save_table(R6_P4_TABLE, "table0E5_revision6_phase4_flip_rates")

_g4 = {}
for d_ in sorted(R6_P4_TABLE["direction"].unique()):
    sub4 = R6_P4_TABLE[(R6_P4_TABLE["direction"] == d_) & (R6_P4_TABLE["model"] == "augmented")]
    _g4[d_] = {r_["family"]: (None if not np.isfinite(r_["flip_rate"]) else float(r_["flip_rate"]))
               for _, r_ in sub4.iterrows()}
_pass4 = all(v is not None and v <= CFG.revision6["gate4_max_flip_rate"]
             for d_ in _g4.values() for v in d_.values())
P4_GATE_R6 = {
    "threshold": CFG.revision6["gate4_max_flip_rate"],
    "flip_rates_augmented_external": _g4,
    "flip_rates_unaugmented_external": {
        d_: {r_["family"]: (None if not np.isfinite(r_["flip_rate"]) else float(r_["flip_rate"]))
             for _, r_ in (R6_P4_TABLE[(R6_P4_TABLE["direction"] == d_) & (R6_P4_TABLE["model"] == "unaugmented")]).iterrows()}
        for d_ in sorted(R6_P4_TABLE["direction"].unique())},
    "passed": bool(_pass4),
}
save_json(P4_GATE_R6, DIRS["metadata"] / "phase4_revision6_gate.json")
print(json.dumps(P4_GATE_R6, indent=2, default=str))
if P4_GATE_R6["passed"]:
    print(f"\nPHASE 4 (REVISION 6) GATE PASSED: all augmented-model external flip rates <= "
          f"{P4_GATE_R6['threshold']}.")
else:
    print(f"\nPHASE 4 (REVISION 6) GATE NOT PASSED: at least one augmented-model external flip rate exceeds "
          f"{P4_GATE_R6['threshold']}. The comparison against the unaugmented model above shows whether the "
          f"augmentation helped; the residual fragility is REPORTED as a robustness finding.")''')

# ============================================================================
# NEW CELL F — REVISION 6 PHASE 5 (inserted before Section 33B markdown)
# ============================================================================
CELL_F_MD = md("""## REVISION 6 PHASE 5 — ERS target repair: the cross-family reliability target (Master V2 Blocker 1, Gate 5)

Revision 5's `y_rel = 0.5·S_challenge + 0.3·S_P2 + 0.2·S_P1` shares P1/P2 with the `S` term inside
`E0 = (F·S·M)^(1/3)`, so `rho(E0, y_rel)` was partly self-fulfilling. Revision 6 replaces the target with the
**cross-family** form (Solution 1):

$$y_{rel}^{(6)} = 0.5\\,S_{challenge} + 0.5\\,S_{calib\\text{-}P3}$$

where `S_challenge` aggregates the held-out challenge families (P4 ∪ P4B) and `S_calib-P3` is the P3
dot-segment stability — **neither P1 nor P2 enters the target any more**. Three correlations are reported
side by side, with the cross-family one as the PRIMARY gate:

- `rho(E0, y_rel_6)` — cross-family, PRIMARY
- `rho(E0, S_challenge)` — legacy held-out population
- `rho(E0, S_P3)` — legacy revision-4 target

**Gate 5:** cross-family `rho(E0, y_rel_6) > 0.10` on at least 3 of the 4 pre-existing runs (the two F68-R
runs are reported alongside). A run that fails is labelled *"ERS target not identified"* and no ERS claim is
made for it.""")

CELL_F = code(r'''# ---------------------------------------------------------------------------
# REVISION 6 PHASE 5 — cross-family reliability target (Master V2 Solution 1)
# ---------------------------------------------------------------------------
ERS_W_R6 = dict(CFG.revision6["ers_target_weights_6"])
FAMILY_OF_COMPONENT_R6 = {"S_challenge": None, "S_calib_P3": "P3_dot_segment"}


def build_reliability_target_r6(P: Dict[str, Any], weights: Dict[str, float] = None):
    """Cross-family y_rel: 0.5 * S_challenge + 0.5 * S_calib_P3, renormalised by the weight present."""
    weights = weights or ERS_W_R6
    n = len(P["C"])
    comps = {"S_challenge": P["S_challenge"], "S_calib_P3": _family_mean(P, FAMILY_OF_COMPONENT_R6["S_calib_P3"])}
    num, den = np.zeros(n), np.zeros(n)
    for name, w in weights.items():
        v = comps[name]
        m = np.isfinite(v)
        num[m] += w * v[m]
        den[m] += w
    out = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return out, comps


for _P in POPS.values():
    _P["y_rel_6"], _P["y_rel_6_components"] = build_reliability_target_r6(_P)

P5_ROWS = []
for rk in RUN_KEYS:
    V = POPS[(rk, "val")]
    ok = V["ers_defined"]
    row = {"run": rk, "n_val": int(ok.sum()),
           "coverage_y_rel6_pct": 100 * float(np.isfinite(V["y_rel_6"][ok]).mean()),
           "coverage_S_challenge_pct": 100 * float(np.isfinite(V["S_challenge"][ok]).mean()),
           "coverage_S_P3_pct": 100 * float(np.isfinite(V["S_calib"][ok]).mean())}
    for label, a, b in [("rho_E0_vs_y_rel_6  [CROSS-FAMILY, PRIMARY]", V["E0"], V["y_rel_6"]),
                        ("rho_E0_vs_S_challenge  [legacy disjoint]", V["E0"], V["S_challenge"]),
                        ("rho_E0_vs_S_P3  [legacy rev-4 target]", V["E0"], V["S_calib"]),
                        ("rho_C_vs_y_rel_6  [confidence baseline]", V["C"], V["y_rel_6"])]:
        r, p, n_ = _sp(a[ok], b[ok])
        row[label] = r
        row[label + " p"] = p
    P5_ROWS.append(row)
P5_TARGET_TABLE_R6 = pd.DataFrame(P5_ROWS)
display(P5_TARGET_TABLE_R6.round(4))
save_table(P5_TARGET_TABLE_R6, "table0F2_revision6_phase5_cross_family_target")

_gate5_col = "rho_E0_vs_y_rel_6  [CROSS-FAMILY, PRIMARY]"
_core4 = ["gram|F48", "gram|F60R", "phresh|F48", "phresh|F60R"]
_core_pass = [bool(P5_TARGET_TABLE_R6.loc[P5_TARGET_TABLE_R6["run"] == rk_, _gate5_col].iloc[0]
                   > CFG.revision6["gate5_min_spearman"]) for rk_ in _core4]
P5_GATE_R6 = {
    "threshold": CFG.revision6["gate5_min_spearman"],
    "rho_cross_family_by_run": P5_TARGET_TABLE_R6.set_index("run")[_gate5_col].round(4).to_dict(),
    "rho_legacy_S_challenge_by_run": P5_TARGET_TABLE_R6.set_index("run")[
        "rho_E0_vs_S_challenge  [legacy disjoint]"].round(4).to_dict(),
    "rho_legacy_S_P3_by_run": P5_TARGET_TABLE_R6.set_index("run")[
        "rho_E0_vs_S_P3  [legacy rev-4 target]"].round(4).to_dict(),
    "core4_runs": _core4, "core4_n_pass": int(sum(_core_pass)),
    "runs_failing": [rk_ for rk_ in _core4 + [r for r in RUN_KEYS if r.endswith("F68R")]
                     if P5_TARGET_TABLE_R6.loc[P5_TARGET_TABLE_R6["run"] == rk_, _gate5_col].iloc[0]
                     <= CFG.revision6["gate5_min_spearman"]],
    "circularity_note": "the revision-6 target shares NO component with E0's S term (P1/P2 excluded); "
                        "S_challenge is held-out by construction, S_calib_P3 is the calibration family",
    "passed": bool(sum(_core_pass) >= CFG.revision6["gate5_min_runs"]),
}
save_json(P5_GATE_R6, DIRS["metadata"] / "phase5_revision6_gate.json")
print(json.dumps(P5_GATE_R6, indent=2, default=str))
if P5_GATE_R6["passed"]:
    print(f"\nPHASE 5 (REVISION 6) GATE PASSED: cross-family rho(E0, y_rel_6) > {P5_GATE_R6['threshold']} on "
          f"{P5_GATE_R6['core4_n_pass']}/4 core runs.")
else:
    print(f"\nPHASE 5 (REVISION 6) GATE NOT PASSED: cross-family rho(E0, y_rel_6) > {P5_GATE_R6['threshold']} on "
          f"only {P5_GATE_R6['core4_n_pass']}/4 core runs. Runs below the threshold are reported as 'ERS target "
          f"not identified' and no ERS claim is made for them; this is the honest cross-family number the plan "
          f"asks for.")''')

print("cells E/F ready")
