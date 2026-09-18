#!/usr/bin/env python3
"""Build trac-phish-revision6.ipynb from trac-phish-revision5_runned.ipynb (surgical modification).

Every modification is traceable to Master_ImplementationV2.txt:
  PHASE 0  preserve dataset layer (no edits to Sections 1-12; fingerprints re-verified)
  PHASE 1  cell A (8 new dinv features) + cell 47/53 edits + cell B (Phase-1 R6 gate, replace policy)
  PHASE 2  cell C (M0 F68-R / prune-v2 / +rank / +pseudo)  -> Gate 2
  PHASE 3  cell D (M2/M2-hybrid/M4-hybrid, target-val tuned) -> Gate 3
  PHASE 4  cell E (identity + stress augmentation, flip rates) -> Gate 4
  PHASE 5  cell F (cross-family y_rel)                      -> Gate 5
  PHASE 6  cell G (StratumDTS + prior-corrected DTS, external AURC) -> Gate 6
  PHASE 7  cell H (reversal decomposition)                  -> Gate 7
  PHASE 8  cell I (shift strata LRT + Holm)                 -> Gate 8
  PHASE 9  cell J (full XAI/ERS/DTS on F68-R via FULL_PIPELINE_FSETS + gate-health statement)
  PHASE 10 cell K (final statistics, manifest, blocker table, negative findings)
"""
import json, copy, sys
import nbformat

SRC = "trac-phish-revision5_runned.ipynb"
DST = "trac-phish-revision6.ipynb"

nb = nbformat.read(SRC, as_version=4)
cells = nb.cells
print(f"loaded {SRC}: {len(cells)} cells")


def find_cell(anchor, start=0):
    for i in range(start, len(cells)):
        if anchor in cells[i].source:
            return i
    raise KeyError(f"anchor not found: {anchor!r}")


def replace_in_cell(anchor, old, new, count=1):
    i = find_cell(anchor)
    src = cells[i].source
    assert old in src, f"edit target not found in cell {i}: {old[:80]!r}"
    cells[i].source = src.replace(old, new, count)
    print(f"  edited cell {i}")


def md(text):
    return nbformat.v4.new_markdown_cell(text)


def code(text):
    return nbformat.v4.new_code_cell(text)


def insert_after(anchor, new_cells):
    i = find_cell(anchor)
    for j, c in enumerate(new_cells):
        cells.insert(i + 1 + j, c)
    print(f"  inserted {len(new_cells)} cell(s) after cell {i}")


# ============================================================================
# EDIT 0 — title
# ============================================================================
replace_in_cell("# Beyond Prediction Confidence",
                "# Beyond Prediction Confidence: Reliability-Calibrated Explanations for Phishing URL Detection",
                "# Beyond Prediction Confidence: Reliability-Calibrated Explanations for Phishing URL Detection\n"
                "\n"
                "**Revision 6** (Master_ImplementationV2). Revision 5 is preserved verbatim; Revision 6 adds the phased\n"
                "plan as clearly-marked `REVISION 6 PHASE n` sections: F68-R representation repair, prune policy v2,\n"
                "zero-shot ceiling variants (rank / pseudo-label), semi-supervised multi-source path to 95%,\n"
                "perturbation augmentation, cross-family ERS target, stratum & prior-corrected DTS on external AURC,\n"
                "reversal diagnosis, shift-strata LRT, and the revision-6 gate table with blocker status.\n"
                "Every phase has a pre-registered hard gate; failures are reported as findings, never repaired silently.")

# ============================================================================
# EDIT 1 — config (cell 4)
# ============================================================================
replace_in_cell("notebook_revision: int = 5",
                "notebook_revision: int = 5",
                "notebook_revision: int = 6")
replace_in_cell("dts_l2_Cs: List[float] = field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0, 100.0])",
                "dts_l2_Cs: List[float] = field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0, 100.0])",
                """dts_l2_Cs: List[float] = field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0, 100.0])
    f68_schema_version: str = "trac-phish-f68r-v1.0"   # revision 6: F60-R + 8 new domain-invariant features (69 columns)
    # ---------------------------------------------------------------------------------------
    # REVISION 6 CONFIGURATION (Master_ImplementationV2, pre-registered before execution)
    # ---------------------------------------------------------------------------------------
    revision6: Dict[str, Any] = field(default_factory=lambda: {
        "ers_target_weights_6": {"S_challenge": 0.5, "S_calib_P3": 0.5},  # cross-family target (Blocker 1)
        "prune_domain_importance": 0.05, "prune_source_shap": 0.01,
        "prune_prevalence_shift": 0.30,                                   # Blocker 7 / Solution 5B.3 policy fix
        "pseudo_threshold": 0.90, "pseudo_max_iter": 2, "pseudo_row_cap": 60_000,   # Solution 5B.1
        "augment_weight": 0.30, "aug_max_urls": 20_000, "aug_max_rows": 60_000,     # Solutions 6A / 6B
        "dts_stratum_C_split": 0.90, "dts_stratum_min_rows": 200,         # Solution 2A
        "gate2_min_external_roc_auc": 0.80,                               # Phase 2 gate (strict-external AUC)
        "gate3_min_external_accuracy": 0.95,                              # Phase 3 gate (both directions)
        "gate4_max_flip_rate": 0.15,                                      # Phase 4 gate (P3 and P5/P6/P7)
        "gate5_min_spearman": 0.10, "gate5_min_runs": 3,                  # Phase 5 gate (cross-family rho)
    })""")

# ============================================================================
# EDIT 2 — cell 47: extraction of the 8 new columns
# ============================================================================
replace_in_cell("def _extract_chunk_dinv(urls: List[str]) -> np.ndarray:",
                """def _extract_chunk_dinv(urls: List[str]) -> np.ndarray:
    return np.asarray([extract_url_features_dinv(u) for u in urls], dtype=np.float64)""",
                """def _extract_chunk_dinv(urls: List[str]) -> np.ndarray:
    return np.asarray([extract_url_features_dinv(u) for u in urls], dtype=np.float64)


def _extract_chunk_dinv68(urls: List[str]) -> np.ndarray:
    return np.asarray([extract_url_features_dinv68(u) for u in urls], dtype=np.float64)""")

replace_in_cell("schema: str = \"both\") -> pd.DataFrame:",
                """    schema: 'f48' | 'f54r' | 'dinv' | 'f60r' | 'both'. 'both' returns one frame with the 48 baseline
    columns, then the 54 robust (R_) columns, then the 7 domain-invariant (D_) columns; 'f60r' returns
    the 54 robust plus the 7 domain-invariant columns. Duplicated names are disambiguated by prefix, so
    the frame is indexed by feature-set membership later.""",
                """    schema: 'f48' | 'f54r' | 'dinv' | 'dinv68' | 'f60r' | 'f68r' | 'both'. 'both' returns one frame with
    the 48 baseline columns, then the 54 robust (R_) columns, then the 7 revision-5 domain-invariant (D_)
    columns, then the 8 revision-6 domain-invariant columns. 'f54r' returns ALL R_/D_ columns (robust +
    both domain-invariant generations) so that any R_/D_-only feature set (F54-R, F60-R, F68-R) can be
    selected from one extraction. Duplicated names are disambiguated by prefix, so the frame is indexed
    by feature-set membership later. Revision 6: the returned frame always passes through
    apply_f68_postprocess (host-entropy standardisation + revision-6 binarisations once registered).""")

replace_in_cell("""    if schema in ("dinv", "f60r", "both"):
        dv = pd.DataFrame(_run(_extract_chunk_dinv, 7), columns=[DINV_PREFIX + c for c in FEATURES_DINV])
        frames.append(dv)
    out = pd.concat(frames, axis=1) if len(frames) > 1 else frames[0]
    return apply_dinv_replacements(out)""",
                """    if schema in ("dinv", "f60r", "both"):
        dv = pd.DataFrame(_run(_extract_chunk_dinv, 7), columns=[DINV_PREFIX + c for c in FEATURES_DINV])
        frames.append(dv)
    out = pd.concat(frames, axis=1) if len(frames) > 1 else frames[0]
    return apply_dinv_replacements(out)""",
                """    if schema in ("dinv", "f54r", "f60r", "both"):
        dv = pd.DataFrame(_run(_extract_chunk_dinv, 7), columns=[DINV_PREFIX + c for c in FEATURES_DINV])
        frames.append(dv)
    if schema in ("dinv68", "f54r", "f68r", "both"):
        d68 = pd.DataFrame(_run(_extract_chunk_dinv68, len(FEATURES_DINV68)),
                           columns=[DINV_PREFIX + c for c in FEATURES_DINV68])
        frames.append(d68)
    out = pd.concat(frames, axis=1) if len(frames) > 1 else frames[0]
    return apply_f68_postprocess(apply_dinv_replacements(out))""")

replace_in_cell('f"{FEATURE_SCHEMA_VERSION}|{ROBUST_SCHEMA_VERSION}|{DINV_SCHEMA_VERSION}|{VOCAB_VERSION}|{len(CLEAN[ds])}|{CFG.run_mode}".encode()',
                'f"{FEATURE_SCHEMA_VERSION}|{ROBUST_SCHEMA_VERSION}|{DINV_SCHEMA_VERSION}|{VOCAB_VERSION}|{len(CLEAN[ds])}|{CFG.run_mode}".encode()',
                'f"{FEATURE_SCHEMA_VERSION}|{ROBUST_SCHEMA_VERSION}|{DINV_SCHEMA_VERSION}|{F68_SCHEMA_VERSION}|{VOCAB_VERSION}|{len(CLEAN[ds])}|{CFG.run_mode}".encode()')

replace_in_cell("""ALL_FEATURE_COLUMNS = (FEATURES_48 + [ROBUST_PREFIX + c for c in FEATURES_54R]
                       + [DINV_PREFIX + c for c in FEATURES_DINV])""",
                """ALL_FEATURE_COLUMNS = (FEATURES_48 + [ROBUST_PREFIX + c for c in FEATURES_54R]
                       + [DINV_PREFIX + c for c in FEATURES_DINV])""",
                """ALL_FEATURE_COLUMNS = (FEATURES_48 + [ROBUST_PREFIX + c for c in FEATURES_54R]
                       + [DINV_PREFIX + c for c in FEATURES_DINV]
                       + [DINV_PREFIX + c for c in FEATURES_DINV68])""")

replace_in_cell("_nan_ok = set(NAN_ALLOWED) | {ROBUST_PREFIX + c for c in NAN_ALLOWED_R} | {DINV_PREFIX + c for c in NAN_ALLOWED_D}",
                "_nan_ok = set(NAN_ALLOWED) | {ROBUST_PREFIX + c for c in NAN_ALLOWED_R} | {DINV_PREFIX + c for c in NAN_ALLOWED_D}",
                "_nan_ok = (set(NAN_ALLOWED) | {ROBUST_PREFIX + c for c in NAN_ALLOWED_R} | {DINV_PREFIX + c for c in NAN_ALLOWED_D}\n"
                "           | {DINV_PREFIX + c for c in NAN_ALLOWED_68})")

replace_in_cell("    assert (np.nan_to_num(arr, nan=0) >= 0).all()",
                "    assert (np.nan_to_num(arr, nan=0) >= 0).all()",
                "    _nn_cols = [c for c in ALL_FEATURE_COLUMNS if c not in SIGNED_FEATURE_COLUMNS_68]\n"
                "    assert (np.nan_to_num(FEATS[ds][_nn_cols].to_numpy(), nan=0) >= 0).all()")

# ============================================================================
# EDIT 3 — cell 53: F68-R feature set + full-pipeline runs
# ============================================================================
replace_in_cell("FEATURES_DINV_ALL = [D + c for c in FEATURES_DINV]",
                "FEATURES_DINV_ALL = [D + c for c in FEATURES_DINV]",
                """FEATURES_DINV_ALL = [D + c for c in FEATURES_DINV]
# Revision 6 (Master V2 Blocker 7): F68-R = F60-R + 8 new domain-invariant features.
# The plan's label is F68-R; its own arithmetic fixes the length at 61 + 8 = 69 columns
# (the same name-vs-count offset the plan already documents for F60-R = 61).
FEATURES_DINV68_ALL = [D + c for c in FEATURES_DINV68]""")

replace_in_cell("assert len(set(FEATURES_60R)) == len(FEATURES_60R), \"duplicate column name in F60-R\"",
                "assert len(set(FEATURES_60R)) == len(FEATURES_60R), \"duplicate column name in F60-R\"",
                """assert len(set(FEATURES_60R)) == len(FEATURES_60R), "duplicate column name in F60-R"
FEATURES_68R = FEATURES_60R + FEATURES_DINV68_ALL
assert len(FEATURES_68R) == 69 and len(set(FEATURES_68R)) == 69, \\
    f"F68-R must have 61 + 8 = 69 columns, found {len(FEATURES_68R)}\"""")

replace_in_cell('    "F60R":  FEATURES_60R,',
                '    "F60R":  FEATURES_60R,',
                '    "F60R":  FEATURES_60R,\n    "F68R":  FEATURES_68R,')

replace_in_cell("FULL_PIPELINE_FSETS = [BASELINE_FSET, PRIMARY_FSET]  # runs that receive the complete XAI/ERS/DTS pipeline",
                "FULL_PIPELINE_FSETS = [BASELINE_FSET, PRIMARY_FSET]  # runs that receive the complete XAI/ERS/DTS pipeline",
                "FULL_PIPELINE_FSETS = [BASELINE_FSET, PRIMARY_FSET, \"F68R\"]  # revision 6: F68-R receives the complete XAI/ERS/DTS pipeline")

replace_in_cell('''    return {"F44": "F44 (plan: F42)", "F54R": "F54-R", "F48B": "F48-B (body only)",
            "F60R": "F60-R (61 cols)"}.get(name, name)''',
                '''    return {"F44": "F44 (plan: F42)", "F54R": "F54-R", "F48B": "F48-B (body only)",
            "F60R": "F60-R (61 cols)"}.get(name, name)''',
                '''    return {"F44": "F44 (plan: F42)", "F54R": "F54-R", "F48B": "F48-B (body only)",
            "F60R": "F60-R (61 cols)", "F68R": "F68-R (69 cols)"}.get(name, name)''')

replace_in_cell('''                                "F60R": "PRIMARY (revision 5): F54-R + 7 domain-invariant features (61 columns)",''',
                '''                                "F60R": "PRIMARY (revision 5): F54-R + 7 domain-invariant features (61 columns)",''',
                '''                                "F60R": "PRIMARY (revision 5): F54-R + 7 domain-invariant features (61 columns)",
                                "F68R": "REVISION 6 PRIMARY: F60-R + 8 new domain-invariant features (69 columns)",''')

# ============================================================================
# EDIT 4 — manifest revision string
# ============================================================================
replace_in_cell('"revision": "2 (targeted methodological revision of the first executed run)",',
                '"revision": "2 (targeted methodological revision of the first executed run)",',
                '"revision": "6 (Master_ImplementationV2 phased revision: F68-R representation, prune policy v2, '
                'zero-shot variants, multi-source 95% path, perturbation augmentation, cross-family ERS target, '
                'stratum/prior-corrected DTS on external AURC, reversal diagnosis, shift strata)",')

print("existing-cell edits done")

# Revision 6: the fresh-extraction audit must compare on the ORIGINAL 7 domain-invariant columns;
# fresh frames now legitimately carry additional derived D68_ binarised copies (revision-6 gate).
replace_in_cell("R5 the Phase-1.3 replacement registry is applied to freshly extracted features too",
                'extract_features_frame(CLEAN["gram"]["url_raw"].values[:256], 1, 256, "dinv").to_numpy(dtype=np.float32),',
                'extract_features_frame(CLEAN["gram"]["url_raw"].values[:256], 1, 256, "dinv")[FEATURES_DINV_ALL].to_numpy(dtype=np.float32),')

# ============================================================================
# NEW CELL A — REVISION 6 PHASE 1a: the 8 new domain-invariant features
# ============================================================================
CELL_A_MD = md("""## Section 11F — REVISION 6 PHASE 1a: the F68-R domain-invariant extension (Master V2, Blocker 7)

Revision 5's Phase-1 gate did **not** fully pass (max normalised Wasserstein 0.334 > 0.15) and the pruning
policy failed to remove `R_is_https` (prevalence shift 0.41) because its source SHAP (0.022) exceeded the
0.01 threshold. Revision 6 therefore:

1. adds the plan's **8 new domain-invariant features** on top of F60-R, producing **F68-R = 61 + 8 = 69 columns**
   (the plan's name-vs-count offset is the same one already documented for F60-R = 61);
2. re-runs the Phase-1 shift gate over **all 15 new features** (7 revision-5 + 8 revision-6) with the strict
   policy: ≤ 0.15 pass, 0.15–0.25 binarise, > 0.25 replace (and if an already-binary feature cannot be
   repaired by binarisation, it is **removed from F68-R** and reported — F60-R is never touched);
3. fixes the pruning policy (Phase 2 cell): *prune iff domain importance > 0.05 AND (source |SHAP| < 0.01 OR
   prevalence shift > 0.30)* — a source-only signal plus a label-independent target feature distribution.

`D_host_entropy_norm_z` is standardised with the SOURCE-TRAIN mean/std (the per-URL extractor cannot know
corpus statistics, so it emits raw `host_entropy` and a registry-driven post-process — the same mechanism as
the Phase-1.3 replacement registry — standardises every frame, cached or freshly extracted).""")

CELL_A = code(r'''# ---------------------------------------------------------------------------
# REVISION 6 PHASE 1a — 8 additional domain-invariant features (Master V2, Blocker 7)
# F68-R = F60-R (61) + 8 new domain-invariant features = 69 columns.
# ---------------------------------------------------------------------------
F68_SCHEMA_VERSION = "trac-phish-f68r-v1.0"
assert F68_SCHEMA_VERSION == CFG.f68_schema_version

_F68 = []


def _df68_(name, ftype, definition, rng, rationale):
    _F68.append(dict(feature=name, group="I: domain-invariant (new in F68-R)", type=ftype,
                     definition=definition, parser_source="body/host", expected_range=rng, notes=rationale))


_df68_("body_scheme_ratio", "ratio",
       "(1 if 'https' occurs in u_body else 0) / max(1, body_token_count)", "[0, 1]",
       "normalised HTTPS concept: ratio, not raw indicator (plan Blocker 7)")
_df68_("host_tld_class", "ordinal",
       "0 = ICANN common gTLD (.com/.org/.net), 1 = ccTLD (2-letter suffix), 2 = other gTLD, 3 = IP literal",
       "{0,1,2,3}", "distribution-stable TLD category (plan Blocker 7)")
_df68_("path_depth_binary", "binary", "1 if path_depth >= 2 else 0", "{0,1}",
       "binarised structural feature (plan Blocker 7)")
_df68_("has_query_binary", "binary", "1 if query_length > 0 else 0", "{0,1}",
       "binarised structural feature (plan Blocker 7)")
_df68_("host_entropy_norm_z", "continuous (standardised)",
       "(host_entropy - mean_source_TRAIN) / std_source_TRAIN; the extractor emits raw host_entropy and the "
       "revision-6 Phase-1 gate standardises the column via the registry below", "R (z-score)",
       "shift-robust by construction (plan Blocker 7)")
_df68_("path_token_density", "continuous", "body_token_count / max(path_length, 1)", "[0, inf)",
       "ratio feature (plan Blocker 7)")
_df68_("digit_run_max", "count", "length of the longest consecutive digit run in the raw URL", "[0, inf)",
       "structural, dataset-invariant (plan Blocker 7)")
_df68_("brand_tld_match", "binary",
       "1 if any brand token occurs in the host AND any brand token occurs in the body", "{0,1}",
       "semantic-invariance check (plan Blocker 7)")

F68_SCHEMA = pd.DataFrame(_F68)
F68_SCHEMA.insert(0, "index", range(1, len(F68_SCHEMA) + 1))
FEATURES_DINV68 = F68_SCHEMA["feature"].tolist()
assert len(FEATURES_DINV68) == len(set(FEATURES_DINV68)) == 8, len(FEATURES_DINV68)
NAN_ALLOWED_68: set = {"host_entropy_norm_z"}      # NaN exactly when host_entropy is NaN (empty host)
SIGNED_FEATURE_COLUMNS_68 = {DINV_PREFIX + "host_entropy_norm_z"}   # z-scores may be negative

# Registries (EMPTY until the revision-6 Phase-1 gate runs; from then on every freshly extracted frame
# passes through apply_f68_postprocess, so a recomputed feature can never disagree with the stored one).
F68_STANDARDIZATION: Dict[str, Dict[str, float]] = {}
F68_REPLACEMENTS: Dict[str, Dict[str, Any]] = {}


def apply_f68_postprocess(df: "pd.DataFrame") -> "pd.DataFrame":
    """Revision-6 Phase-1 post-processing: host-entropy standardisation + revision-6 binarisations."""
    spec = F68_STANDARDIZATION.get("host_entropy_norm_z")
    if spec is not None and DINV_PREFIX + "host_entropy_norm_z" in df.columns:
        df[DINV_PREFIX + "host_entropy_norm_z"] = (
            (df[DINV_PREFIX + "host_entropy_norm_z"].to_numpy(dtype=np.float64) - spec["mean"]) / spec["std"]
        ).astype(np.float32)
    for col, bspec in F68_REPLACEMENTS.items():
        if col.startswith("D68_"):
            # binarised COPY of a revision-5 feature: derive the source column and create the copy
            src_col = DINV_PREFIX + col[len("D68_"):-len("_bin")]
            if src_col in df.columns:
                df[col] = (df[src_col].to_numpy(dtype=np.float64) > bspec["threshold"]).astype(np.float32)
        elif col in df.columns:
            df[col] = (df[col].to_numpy(dtype=np.float64) > bspec["threshold"]).astype(np.float32)
    return df


_BRAND_RES = [re.compile(re.escape(t)) for t in SEMANTIC_VOCAB["brand"]]
_COMMON_GTLD = frozenset({"com", "org", "net"})


def _host_tld_class(host: str) -> int:
    if host_is_ip(host):
        return 3
    labels = [lab for lab in host.lower().split(".") if lab]
    if not labels:
        return 2
    last = labels[-1]
    if last in _COMMON_GTLD:
        return 0
    if len(last) == 2 and last.isalpha():
        return 1
    return 2


def extract_url_features_dinv68(url: str) -> list:
    """Compute the 8 revision-6 domain-invariant features for one raw URL (deterministic, offline).

    host_entropy_norm_z is emitted as RAW host_entropy; the standardisation to the source-TRAIN mean/std
    is a registry-driven post-process (apply_f68_postprocess), because a per-URL function cannot know the
    corpus statistics. path_depth_binary uses the same path_depth definition as F54-R (number of
    NON-empty '/'-separated path segments).
    """
    u = url.strip()
    p = split_url(u)
    host, path = p.host, p.path
    b = url_body(u)
    bl = b.lower()
    n_tok = len(_TOKEN_RE.findall(b))
    segs = [s for s in path.split("/") if s]
    h_ent = shannon_entropy(host) if len(host) > 1 else 0.0
    if not math.isfinite(h_ent):
        h_ent = 0.0
    best = cur = 0
    for ch in u:
        if ch in _DIGITS:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    host_hit = any(rx.search(host.lower()) for rx in _BRAND_RES)
    body_hit = any(rx.search(bl) for rx in _BRAND_RES)
    return [
        (1 if "https" in bl else 0) / max(1, n_tok),
        _host_tld_class(host),
        1 if len(segs) >= 2 else 0,
        1 if (p.query is not None and len(p.query) > 0) else 0,
        h_ent,                                   # standardised later via the registry
        n_tok / max(len(path), 1),
        best,
        1 if (host_hit and body_hit) else 0,
    ]


# ---- unit tests (run before the features are used anywhere) --------------------------------
_f68_cases = {
    # url:                                                  scheme_ratio, tld, depth_bin, query_bin, density, run_max, brand
    "https://example.com/":                                 [0.0, 0, 0, 0, None, 0, 0],
    "http://example.com":                                   [0.0, 0, 0, 0, 2.0, 0, 0],
    "http://a.b.example.com:8080/x/y?q=1":                  [0.0, 0, 1, 1, None, 4, 0],
    "http://example.com/httpsecure/login?a=1&b=2":          [0.125, 0, 1, 1, None, 1, 0],
    "http://195.88.22.34/paypal-verify/login44":            [0.0, 3, 1, 0, None, 3, 0],
    "https://safe-paypal.example.co.uk/a1b2c3/x":           [0.0, 1, 1, 0, None, 1, 1],
    "http://paypal.com.verify-login.example.com/secure/paypal/signin": [0.0, 0, 1, 0, None, 0, 1],
}
for _u, _exp in _f68_cases.items():
    _got = extract_url_features_dinv68(_u)
    _names = ["body_scheme_ratio", "host_tld_class", "path_depth_binary", "has_query_binary",
              "host_entropy_norm_z(raw)", "path_token_density", "digit_run_max", "brand_tld_match"]
    for _j, _e in enumerate(_exp):
        if _e is not None:
            _k = _j if _j < 4 else _j + 1     # skip the raw-entropy slot (position 4)
            assert abs(_got[_k] - _e) < 1e-9, f"{_u}: {_names[_k]} = {_got[_k]}, expected {_e}"
# 'https' inside the BODY (not the scheme) must count: body of http://example.com/httpsecure contains it
_r = extract_url_features_dinv68("http://example.com/httpsecure/login")
assert _r[0] > 0, "body_scheme_ratio must fire on 'httpsecure' inside the body"
# brand_tld_match needs BOTH host and body hits
assert extract_url_features_dinv68("http://paypal.com/login")[7] == 1
assert extract_url_features_dinv68("http://example.com/paypal-login")[7] == 0
# digit_run_max counts the longest run anywhere in the raw URL
assert extract_url_features_dinv68("http://a.com/x12345y9")[6] == 5
# path_depth_binary consistency with the F54-R path_depth definition
for _u in ["http://a.com/x/y", "http://a.com/x/y/z", "http://a.com/", "http://a.com"]:
    _seg = len([s for s in split_url(_u).path.split("/") if s])
    assert extract_url_features_dinv68(_u)[2] == (1 if _seg >= 2 else 0), _u
print("F68-R revision-6 extractor: unit tests passed (8 features).")

F68_SCHEMA_TABLE = F68_SCHEMA.copy()
F68_SCHEMA_TABLE["schema_version"] = F68_SCHEMA_VERSION
save_table(F68_SCHEMA_TABLE, "table02f_revision6_f68_feature_schema")
save_json(F68_SCHEMA_TABLE.to_dict(orient="records"), DIRS["metadata"] / "feature_schema_f68_v1.json")
display(F68_SCHEMA_TABLE)
print(f"F68-R = {len(FEATURES_54R) + len(FEATURES_DINV)} (F60-R) + {len(FEATURES_DINV68)} (revision-6 domain-invariant) = "
      f"{len(FEATURES_54R) + len(FEATURES_DINV) + len(FEATURES_DINV68)} columns. The plan names this set F68-R; the count 69 is "
      f"authoritative (the same name-vs-count offset the plan documents for F60-R = 61).")''')

# ============================================================================
# NEW CELL B — REVISION 6 PHASE 1b: shift gate over the 15 new features
# ============================================================================
CELL_B_MD = md("""### REVISION 6 PHASE 1b — the strict Phase-1 shift gate on all 15 new features (Gate 1)

Gate 1 (Master V2): **every one of the 15 new features ≤ 0.15 normalised Wasserstein**; features in
(0.15, 0.25] are binarised; features above 0.25 are replaced — and an already-binary feature that
binarisation cannot repair is **removed from F68-R** (F60-R is never modified, so every revision-5 result
remains exactly reproducible). `D_host_entropy_norm_z` is first standardised with the SOURCE-TRAIN
mean/std (a label-free, source-only operation) and the two F68-R imputers are refitted, exactly as the
revision-5 gate refits imputers after its own replacements.""")

CELL_B = code(r'''# ---------------------------------------------------------------------------
# REVISION 6 PHASE 1b — Phase-1 shift gate over ALL 15 new features (Gate 1)
# ---------------------------------------------------------------------------
# 1) standardise D_host_entropy_norm_z with SOURCE (gram) TRAIN statistics (no labels, no target data)
_he_col = DINV_PREFIX + "host_entropy_norm_z"
_tr_g68 = partition_index("gram", "train")
_he_raw = FEATS["gram"][_he_col].to_numpy(dtype=np.float64)[_tr_g68]
_he_mu, _he_sd = float(np.nanmean(_he_raw)), float(np.nanstd(_he_raw))
assert np.isfinite(_he_mu) and _he_sd > 0, "host_entropy standardisation statistics are degenerate"
F68_STANDARDIZATION["host_entropy_norm_z"] = {"mean": _he_mu, "std": _he_sd}
for _ds in FEATS:
    FEATS[_ds][_he_col] = ((FEATS[_ds][_he_col].to_numpy(dtype=np.float64) - _he_mu) / _he_sd).astype(np.float32)
LEDGER.record("phase1_revision6", "gram|F68R", "gram", "train", "fit_preprocessing", len(_tr_g68),
              "host_entropy_norm_z standardised with source-TRAIN mean/std (label-free)")
print(f"D_host_entropy_norm_z standardised with Gram TRAIN mean={_he_mu:.4f}, std={_he_sd:.4f}.")

_R68_NEW15 = ([DINV_PREFIX + c for c in FEATURES_DINV] + [DINV_PREFIX + c for c in FEATURES_DINV68])
_F68_LEGACY7 = set(DINV_PREFIX + c for c in FEATURES_DINV)


def _dinv68_shift(cols: List[str]) -> pd.DataFrame:
    sh = feature_shift(raw_matrix("gram", _rows_g, "F68R"), raw_matrix("phresh", _rows_p, "F68R"), cols)
    return sh[sh["feature"].isin(_R68_NEW15)].reset_index(drop=True)


FEATURE_SETS["F68R"] = list(FEATURES_68R)          # the gate may shrink this list below
DINV68_SHIFT = _dinv68_shift(FEATURE_SETS["F68R"])
display(DINV68_SHIFT[["feature", "type", "wasserstein_norm", "js_divergence", "mean_source", "mean_target"]].round(4))

# 2) the strict revision-6 policy. Binarisations of the 8 revision-6 features happen IN PLACE (the
#    columns exist only in F68-R). A revision-5 feature that the stricter policy would binarise is
#    swapped for a D68_ binarised COPY so that F60-R and every revision-5 result stay untouched.
_w = dict(zip(DINV68_SHIFT["feature"], DINV68_SHIFT["wasserstein_norm"].astype(float)))
_swapped_cols, _removed_cols, _binarised_cols = [], [], []
for _f, _wv in sorted(_w.items()):
    if _wv <= P1_MAX:
        continue
    _new68 = _f not in _F68_LEGACY7
    if _wv <= P1_REPLACE:                       # binarise (0.15, 0.25]
        if _new68:
            _thr = float(np.nanmedian(FEATS["gram"][_f].to_numpy(dtype=np.float64)[_tr_g68]))
            for _ds in FEATS:
                FEATS[_ds][_f] = (FEATS[_ds][_f].to_numpy(dtype=np.float64) > _thr).astype(np.float32)
            F68_REPLACEMENTS[_f] = {"rule": "binarised at the source-TRAIN median (revision-6 gate)",
                                    "threshold": _thr}
            _binarised_cols.append(_f)
            LOG.warning("Revision-6 Phase 1: %s binarised at source-TRAIN median %.6g (W=%.3f)", _f, _thr, _wv)
        else:
            _swap = "D68_" + _f[len(DINV_PREFIX):] + "_bin"
            _thr = float(np.nanmedian(FEATS["gram"][_f].to_numpy(dtype=np.float64)[_tr_g68]))
            for _ds in FEATS:
                FEATS[_ds][_swap] = (FEATS[_ds][_f].to_numpy(dtype=np.float64) > _thr).astype(np.float32)
            F68_REPLACEMENTS[_swap] = {"rule": "binarised COPY of " + _f +
                                       " at the source-TRAIN median (revision-6 gate; F60-R untouched)",
                                       "threshold": _thr}
            _swapped_cols.append((_f, _swap))
            LOG.warning("Revision-6 Phase 1: %s swapped for binarised copy %s (W=%.3f); F60-R untouched",
                        _f, _swap, _wv)
    else:                                        # > 0.25: replace; remove if binarisation cannot repair
        if _new68 and _f not in F68_REPLACEMENTS:
            _thr = float(np.nanmedian(FEATS["gram"][_f].to_numpy(dtype=np.float64)[_tr_g68]))
            _before = FEATS["gram"][_f].to_numpy(dtype=np.float64)[_tr_g68].copy()
            _is_binary = set(np.unique(_before[~np.isnan(_before)])) <= {0.0, 1.0}
            if not _is_binary:
                for _ds in FEATS:
                    FEATS[_ds][_f] = (FEATS[_ds][_f].to_numpy(dtype=np.float64) > _thr).astype(np.float32)
                F68_REPLACEMENTS[_f] = {"rule": "binarised at the source-TRAIN median (revision-6 gate, W > 0.25)",
                                        "threshold": _thr}
                _binarised_cols.append(_f)
        elif not _new68:
            _swap = "D68_" + _f[len(DINV_PREFIX):] + "_bin"
            if _swap not in FEATS["gram"].columns:
                _thr = float(np.nanmedian(FEATS["gram"][_f].to_numpy(dtype=np.float64)[_tr_g68]))
                for _ds in FEATS:
                    FEATS[_ds][_swap] = (FEATS[_ds][_f].to_numpy(dtype=np.float64) > _thr).astype(np.float32)
                F68_REPLACEMENTS[_swap] = {"rule": "binarised COPY of " + _f + " (revision-6 gate, W > 0.25)",
                                           "threshold": _thr}
                _swapped_cols.append((_f, _swap))

# re-measure the shift of the (possibly binarised) features and remove unrepairable ones
FEATS_68_POST = _dinv68_shift(FEATURE_SETS["F68R"])
_w_post = dict(zip(FEATS_68_POST["feature"], FEATS_68_POST["wasserstein_norm"].astype(float)))
for _f, _wv in sorted(_w_post.items()):
    if _wv > P1_REPLACE:
        _col = _f
        if _col in FEATURE_SETS["F68R"]:
            FEATURE_SETS["F68R"] = [c for c in FEATURE_SETS["F68R"] if c != _col]
            _removed_cols.append((_f, float(_wv)))
            LOG.warning("Revision-6 Phase 1: %s REMOVED from F68-R (W=%.3f still > %.2f after binarisation; "
                        "already-binary features cannot be repaired) — reported, not hidden.", _f, _wv, P1_REPLACE)
for _old, _new in _swapped_cols:
    if _old in FEATURE_SETS["F68R"]:
        FEATURE_SETS["F68R"] = [_new if c == _old else c for c in FEATURE_SETS["F68R"]]
    if _new not in ALL_FEATURE_COLUMNS:
        ALL_FEATURE_COLUMNS.append(_new)   # keep the schema-identity audit true (sanity check)
assert len(FEATURE_SETS["F68R"]) == len(set(FEATURE_SETS["F68R"])), "duplicate column in final F68-R"

# 3) refit the F68-R imputers on the repaired columns and clear the stale matrix cache
_FEAT_NP.clear()
for _rk68 in ("gram|F68R", "phresh|F68R"):
    _src68 = RUNS[_rk68]["source"]
    _tr68 = partition_index(_src68, "train")
    RUNS[_rk68]["imputer"] = TrainOnlyImputer().fit(raw_matrix(_src68, _tr68, "F68R"),
                                                    CLEAN[_src68]["record_id"].values[_tr68])
    RUNS[_rk68]["features"] = FEATURE_SETS["F68R"]
LOG.warning("Revision-6 Phase 1: %d binarisation(s), %d swap(s), %d removal(s) -> F68-R imputers refitted.",
            len(_binarised_cols), len(_swapped_cols), len(_removed_cols))

# 4) redundancy of the 8 revision-6 features against F54-R (same audit as the revision-5 gate)
_REDUND68 = []
for _f in [DINV_PREFIX + c for c in FEATURES_DINV68]:
    _best, _bestr = None, 0.0
    _a_all = FEATS["gram"][_f].to_numpy(dtype=np.float64)[_tr_g68]
    for _g in FEATURES_54R_ALL:
        _b_all = FEATS["gram"][_g].to_numpy(dtype=np.float64)[_tr_g68]
        _m = np.isfinite(_a_all) & np.isfinite(_b_all)
        if _m.sum() < 100 or _a_all[_m].std() == 0 or _b_all[_m].std() == 0:
            continue
        _r = abs(float(np.corrcoef(_a_all[_m], _b_all[_m])[0, 1]))
        if _r > _bestr:
            _best, _bestr = _g, _r
    _REDUND68.append({"new_feature": _f, "closest_F54R_feature": _best, "abs_pearson_r": _bestr,
                      "verdict": "EXACT DUPLICATE (no new information)" if _bestr > 0.9999 else
                                 "highly redundant" if _bestr > 0.95 else "adds information"})
DINV68_REDUNDANCY = pd.DataFrame(_REDUND68)
display(DINV68_REDUNDANCY.round(4))

# 5) final gate measurement on the FINAL F68-R composition
DINV68_SHIFT_FINAL = _dinv68_shift(FEATURE_SETS["F68R"])
display(DINV68_SHIFT_FINAL[["feature", "wasserstein_norm", "js_divergence", "mean_source", "mean_target"]]
        .rename(columns={"wasserstein_norm": "wasserstein_norm_final"}).round(4))
_final_w = DINV68_SHIFT_FINAL["wasserstein_norm"].astype(float)
P1_GATE_R6 = {
    "n_new_features_measured": int(len(_R68_NEW15)),
    "n_features_final_f68r": len(FEATURE_SETS["F68R"]),
    "max_wasserstein_final": float(_final_w.max()),
    "n_pass_le_0.15_initial": int(sum(1 for v in _w.values() if v <= P1_MAX)),
    "n_binarised": len(_binarised_cols), "binarised": _binarised_cols,
    "n_swapped_to_f68_copy": len(_swapped_cols),
    "swapped": [f"{a} -> {b}" for a, b in _swapped_cols],
    "n_removed_unrepairable": len(_removed_cols), "removed": [f"{a} (W={b:.3f})" for a, b in _removed_cols],
    "replacements": F68_REPLACEMENTS,
    "standardization": F68_STANDARDIZATION,
    "gate_passed": bool((_final_w <= P1_MAX).all()),
    "decision_signal": "source/target TRAIN feature distributions only; no labels, no test partition, no target metric",
}
save_json(P1_GATE_R6, DIRS["metadata"] / "phase1_revision6_gate.json")
save_table(DINV68_SHIFT_FINAL, "table0C2_revision6_phase1_gate")
print(json.dumps({k: v for k, v in P1_GATE_R6.items() if k != "replacements"}, indent=2, default=str))
if P1_GATE_R6["gate_passed"]:
    print(f"\nPHASE 1 (REVISION 6) GATE PASSED: every new feature in the final F68-R composition has "
          f"normalised Wasserstein <= {P1_MAX}.")
else:
    print(f"\nPHASE 1 (REVISION 6) GATE: NOT fully passed even after the prescribed remedies. Features above "
          f"{P1_MAX} in the final composition are listed above with their measured shift. Per the plan this is "
          f"REPORTED as a representation-limitation finding; nothing was silently repaired and no target label "
          f"or target metric was used at any point.")''')

print("cell A/B sources ready")
nbformat.write(nb, "/dev/null") if False else None
json.dump({"ok": 1}, open("/tmp/_build_ok.json", "w"))
print("part 1 complete — continuing with cells C-K in build_rev6_part2.py")
