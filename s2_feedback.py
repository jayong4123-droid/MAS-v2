"""
Statement 2 - Employee free-text feedback: topics, department themes, individual profiles.

Run:  python src/s2_feedback.py [--k 9] [--scan]
Outputs go to outputs/s2_*.csv / .png

Pipeline
  1. Clean      : normalise text, flag non-answers ("na", "-", "\\n"), keep an audit trail.
  2. Segment    : split multi-point answers into sentences / numbered points (one answer often mixes
                  praise + complaint + suggestion), so a topic is assigned per idea, not per paragraph.
  3. Topics     : TF-IDF (uni+bigrams, light lemmatisation) -> NMF. k chosen by NPMI coherence and
                  seed-stability (scan), then labelled by a human from top terms + exemplars.
  4. Tone       : transparent lexicon-based stance per segment (praise / concern / suggestion).
  5. Departments: topic share, lift vs company, permutation test (cells are small -> no chi2 asymptotics).
  6. Profiles   : per-employee topic mix + tone -> k-means personas (bonus), with explicit caveats.
"""
import argparse
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"; OUT.mkdir(exist_ok=True)
SEED = 42

# ------------------------------------------------------------------ 1. cleaning
PLACEHOLDER = re.compile(r"^\W*(n\W?a\W?|nil|none|no(thing| comments?)?|nope|-+|\.+|,+)\W*$", re.I)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns={"unique_identifier": "id", "employee_feedback": "text_raw"}).copy()
    t = df["text_raw"].fillna("").astype(str)
    t = t.str.replace(r"\\n", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    df["text"] = t
    n_alpha = t.str.count(r"[A-Za-z]+")
    df["is_placeholder"] = t.map(lambda s: bool(PLACEHOLDER.match(s))) | (n_alpha == 0)
    df["n_words"] = n_alpha
    # one-word answers ("Good", "WFH", "MD", "many") carry no analysable content on their own
    df["status"] = np.where(df["is_placeholder"], "non-response",
                    np.where(df["n_words"] < 2, "too short (1 word)", "valid"))
    return df


def segment(text: str):
    """Split one answer into idea-level segments (sentences / numbered points) and repair fragments."""
    t = re.sub(r"\b(i\.e\.|e\.g\.|etc\.)", lambda m: m.group(0).replace(".", "~"), text, flags=re.I)
    t = re.sub(r"(?:^|\s)\(?\d{1,2}[.)]\s+", " |", t)                        # numbered points
    t = re.sub(r"\s+-\s+(?=[A-Z])", " |", t)                                  # dash bullets
    parts = [p.replace("~", ".").strip(" .;-") for p in re.split(r"(?<=[.!?;])\s+|\|", t)]
    out = []
    for p in parts:
        if len(re.findall(r"[A-Za-z]+", p)) < 3 or p.endswith(":"): continue
        # consequence sentences ("This has helped me feel...") belong to the idea before them
        if out and re.match(r"(this|that|these)\b", p, re.I): out[-1] += ". " + p
        else: out.append(p)
    return out


# ------------------------------------------------------------------ 2. text normalisation
DOMAIN_STOP = {"company", "employee", "employees", "feel", "feels", "like", "would", "could", "also", "just",
               "really", "make", "makes", "help", "helped", "helps", "helpful", "get", "one", "way", "well",
               "within", "much", "many", "sometimes", "still", "even", "see", "much", "role", "overall",
               "appreciate", "generally", "great", "good", "need", "needs", "lot", "ever", "us", "etc",
               "work", "especially", "key", "likes", "greatly", "highly", "whether", "often", "always", "every"}
STOP = set(ENGLISH_STOP_WORDS) | DOMAIN_STOP


def lemma(w):
    for suf, rep in (("ies", "y"), ("ings", "ing"), ("sses", "ss")):
        if w.endswith(suf) and len(w) > 5: return w[:-len(suf)] + rep
    if w.endswith("s") and not w.endswith(("ss", "us", "is")) and len(w) > 4: return w[:-1]
    return w


def tokenize(s):
    return [lemma(w) for w in re.findall(r"[a-z]+", s.lower()) if w not in STOP and len(w) > 2]


# ------------------------------------------------------------------ 3. topics
def npmi_coherence(top_idx, Xbin, eps=1e-12):
    n = Xbin.shape[0]; df = np.asarray(Xbin.sum(0)).ravel() / n
    co = (Xbin.T @ Xbin).toarray() / n
    s = []
    for i, j in combinations(top_idx, 2):
        pij = co[i, j] + eps
        s.append(np.log(pij / (df[i] * df[j] + eps)) / -np.log(pij))
    return float(np.mean(s))


def fit_nmf(X, k, seed=SEED, init="nndsvda"):
    m = NMF(k, init=init, random_state=seed, max_iter=1500, l1_ratio=0.0, alpha_W=0.0)
    W = m.fit_transform(X); return m, W


def scan_k(X, Xbin, ks, n_top=10, n_seeds=5):
    rows = []
    for k in ks:
        m, _ = fit_nmf(X, k)
        top = np.argsort(-m.components_, axis=1)[:, :n_top]
        coh = np.mean([npmi_coherence(t, Xbin) for t in top])
        # stability: best-match cosine of topic vectors across random-init refits
        Hs = [fit_nmf(X, k, seed=s, init="random")[0].components_ for s in range(n_seeds)]
        unit = lambda H: H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)
        base = unit(m.components_)
        stab = np.mean([(base @ unit(H).T).max(1).mean() for H in Hs])
        rows.append({"k": k, "coherence_npmi": coh, "stability_cos": stab, "recon_err": m.reconstruction_err_})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 4. tone
POS = {"positive", "supportive", "excellent", "satisfied", "empower", "empowered", "trust", "valued", "proud",
       "motivated", "collaborative", "transparent", "special", "strong", "happy", "comfortable", "effective",
       "enjoyable", "seamless", "constructive", "respected", "heard", "accommodating", "fair", "competitive",
       "instrumental", "commitment", "open", "camaraderie", "energized", "efficient", "thorough", "robust",
       "celebrate", "inclusive", "available", "recognize", "recognized", "appreciated", "streamlined", "streamlining"}
NEG = {"lack", "lacking", "lacks", "difficult", "unclear", "outdated", "unhelpful", "frustrating", "frustration",
       "burnout", "stress", "stressed", "overwhelming", "unreasonable", "insufficient", "lackluster", "minimal",
       "disconnected", "limited", "bored", "unchallenged", "blame", "slow", "ignoring", "vague", "meaningless",
       "dominate", "toll", "misunderstanding", "miscommunication", "delay", "roadblock", "issue", "problem",
       "whim", "favourite", "unsafe", "decreasing", "out of touch", "heavy", "blurred", "short", "poor"}
SUGGEST = re.compile(r"\b(should|would like|could be|need(s|ed)? to|more (effort|emphasis|investment|opportunit|frequent)"
                     r"|improv|consider|suggest|wish|relook|recommend|must|would help|ought)", re.I)
CONTRAST = re.compile(r"\b(however|but|although|though|yet|while)\b", re.I)


def stance(seg: str) -> str:
    w = set(re.findall(r"[a-z]+", seg.lower()))
    pos, neg = len(w & POS), len(w & NEG)
    negated = bool(re.search(r"\b(don't|do not|doesn't|not|no|isn't|never)\b", seg.lower()))
    if SUGGEST.search(seg): return "suggestion"
    if neg > pos or (negated and pos <= neg + 1 and neg > 0): return "concern"
    if pos > neg: return "praise"
    if negated: return "concern"
    return "neutral"


# ------------------------------------------------------------------ 5. guided theme taxonomy
# Pure NMF on ~230 short segments gives muddy "catch-all" topics (see REPORT_S2.md). We therefore use NMF
# for discovery / validation and a transparent, multi-label keyword taxonomy for the final labels.
THEMES = {
    "Workload & work-life balance": r"workload|overtime|burnout|work-life|work life|balance|weekend|all hours|hours|flexib|wfh|work from home|schedule|time off|manageable|boundar|headcount|priorit",
    "Pay, benefits & recognition": r"\bpay\b|salary|salaries|compensat|benefit|bonus|incentive|reward|recogni|raise|benchmark|remunerat",
    "Career growth & development": r"career|growth|professional development|training|promotion|advancement|skills|mobility|opportunit(y|ies) (to|for) (learn|grow)|learn|development|job description|cross-train|job shadow",
    "Feedback & performance mgmt": r"feedback|performance review|check-in|appraisal|after-action",
    "Leadership, direction & transparency": r"leader|senior management|top management|upper management|executive|transparen|vision|direction|strateg|objective|decisive|out of touch|disconnected|purpose of|company direction",
    "Line-manager support": r"manager|\bboss|supervisor|managerial|lead by example|my team lead",
    "Teamwork & collaboration": r"\bteam|teammate|collaborat|colleague|camaraderie|rapport|cross-unit|silos|celebrate",
    "Communication & information flow": r"communicat|information|meeting|knowledge|agenda|documentation|updates|share[sd]? ",
    "Psychological safety & voice": r"psycholog|speak up|voice|opinion|ideas?\b|heard|receptive|mistake|blame|open-door|openness|open environment|candid|vulnerab|trust|concerns",
    "Diversity, inclusion & culture": r"divers|inclus|represent|equity|belonging|respect|culture|favourite|hate crime",
    "Well-being & mental health": r"mental health|wellness|well-being|wellbeing|stress|burnout|toll",
    "Resources, tools & processes": r"technolog|resource|tools?\b|process|procurement|infrastructure|video conferenc|polic(y|ies)|platform|innovation|efficien|streamlin",
    "Purpose, meaning & engagement": r"purpose|meaning|passion|motivat|bored|unchallenged|proud|mission|excitement|energi[sz]ed|satisf|empower",
    "Sustainability & social impact": r"sustainab|environment(?!al? )|social impact|community|customer",
}
THEME_RX = {k: re.compile(v, re.I) for k, v in THEMES.items()}
ALERT_RX = re.compile(r"hate crime|hitler|harass|discriminat|bully|abus|threat|unsafe|retaliat|victimi", re.I)


def assign_themes(text: str):
    return [t for t, rx in THEME_RX.items() if rx.search(text)]


def perm_test(df_resp, theme_cols, n_perm=5000, seed=SEED):
    """Is a department's mention rate of a theme different from the rest? Permutation test on department labels."""
    rng = np.random.default_rng(seed)
    depts = df_resp["department"].values; M = df_resp[theme_cols].values.astype(float)
    uniq = sorted(set(depts))
    def stat(lbl):
        return np.array([[abs(M[lbl == d, j].mean() - M[lbl != d, j].mean()) for j in range(M.shape[1])] for d in uniq])
    obs = stat(depts); cnt = np.zeros_like(obs)
    for _ in range(n_perm):
        cnt += stat(rng.permutation(depts)) >= obs - 1e-12
    return pd.DataFrame((cnt + 1) / (n_perm + 1), index=uniq, columns=theme_cols)


def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / (np.arange(n) + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def main(k=None, scan=False):
    raw = pd.read_excel(ROOT / "data" / "DS2-Simulated-Employee-Feedback.xlsx")
    df = clean(raw)
    df[["id", "department", "text_raw", "text", "status", "n_words"]].to_csv(OUT / "s2_cleaning_audit.csv", index=False)
    print("Responses:", len(df)); print(df["status"].value_counts().to_string())
    nr = df.groupby("department")["status"].apply(lambda s: (s != "valid").mean()).round(2)
    print("\nNon-/low-response rate by dept:\n", nr.to_string())
    # truncated answers (cut mid-word / no terminal punctuation after long text) - data-quality flag
    df["maybe_truncated"] = (df["status"] == "valid") & (df["n_words"] >= 8) & ~df["text"].str.contains(r"[.!?)\]\"]$")
    print("Possibly truncated answers:", int(df["maybe_truncated"].sum()))

    v = df[df["status"] == "valid"].copy()
    seg = (v.assign(segment=v["text"].map(segment)).explode("segment").dropna(subset=["segment"])
             .rename_axis("row").reset_index())
    seg["seg_id"] = np.arange(len(seg))
    print(f"\nValid responses {len(v)} -> {len(seg)} idea-level segments")

    # ---------------- (A) unsupervised discovery
    tf = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, ngram_range=(1, 2),
                         min_df=2, max_df=0.5, sublinear_tf=True)
    X = tf.fit_transform(seg["segment"]); terms = np.array(tf.get_feature_names_out())
    Xbin = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, ngram_range=(1, 2),
                           min_df=2, max_df=0.5, binary=True, vocabulary=tf.vocabulary_).fit_transform(seg["segment"])
    S = scan_k(X, Xbin, range(4, 15)); S.round(3).to_csv(OUT / "s2_k_scan.csv", index=False)
    print("\nNMF k-scan\n", S.round(3).to_string(index=False))
    if scan: return
    k = k or 10
    m, W = fit_nmf(X, k); H = m.components_
    seg["nmf_topic"] = W.argmax(1)
    nmf_tbl = pd.DataFrame({"nmf_topic": range(k), "n_segments": [int((seg.nmf_topic == t).sum()) for t in range(k)],
                            "top_terms": [", ".join(terms[np.argsort(-H[t])[:8]]) for t in range(k)]})
    nmf_tbl.to_csv(OUT / "s2_nmf_topics.csv", index=False)
    print(f"\nNMF discovery (k={k})\n", nmf_tbl.to_string(index=False))

    # ---------------- (B) guided taxonomy
    seg["themes"] = seg["segment"].map(assign_themes)
    seg["stance"] = seg["segment"].map(stance)
    seg["primary_theme"] = seg["themes"].map(lambda t: t[0] if t else "Unassigned")
    cover = (seg["themes"].map(len) > 0).mean()
    print(f"\nTaxonomy coverage of segments: {cover:.0%}")
    print("Unassigned examples:"); [print("  -", x[:110]) for x in seg.loc[seg.themes.map(len) == 0, "segment"].head(8)]
    # agreement between the two views (NMI): does the taxonomy recover structure NMF found?
    from sklearn.metrics import normalized_mutual_info_score as nmi
    ok = seg["primary_theme"] != "Unassigned"
    print("NMI(NMF topic, taxonomy primary theme):", round(nmi(seg.loc[ok, "nmf_topic"], seg.loc[ok, "primary_theme"]), 3))
    seg.drop(columns=["themes"]).assign(themes=seg["themes"].map("; ".join)).to_csv(OUT / "s2_segments_labelled.csv", index=False)

    # ---------------- response level
    theme_cols = list(THEMES)
    for t in theme_cols: v[t] = v["text"].map(lambda x, t=t: bool(THEME_RX[t].search(x))).astype(int)
    sg = seg.groupby("row")["stance"].value_counts().unstack(fill_value=0).reindex(columns=["praise", "concern", "suggestion", "neutral"], fill_value=0)
    v = v.join(sg).fillna({c: 0 for c in sg.columns})
    n_seg = v[["praise", "concern", "suggestion", "neutral"]].sum(1).replace(0, 1)
    for c in ["praise", "concern", "suggestion"]: v[f"{c}_share"] = v[c] / n_seg
    v["net_tone"] = v["praise_share"] - v["concern_share"] - 0.5 * v["suggestion_share"] * 0   # praise minus concern
    v["alert"] = v["text"].str.contains(ALERT_RX)

    # ---------------- (1) company-wide topics
    overall = v[theme_cols].mean().sort_values(ascending=False)
    stance_by_theme = (seg.explode("themes").dropna(subset=["themes"]).groupby("themes")["stance"]
                          .value_counts(normalize=True).unstack(fill_value=0).reindex(theme_cols))
    topic_summary = pd.DataFrame({"pct_of_valid_responses": (overall * 100).round(1),
                                  "n_segments": seg.explode("themes")["themes"].value_counts().reindex(overall.index).fillna(0).astype(int)}
                                 ).join((stance_by_theme[["praise", "concern", "suggestion"]] * 100).round(0))
    topic_summary.to_csv(OUT / "s2_company_topics.csv")
    print("\n=== Company-wide themes ===\n", topic_summary.to_string())

    # ---------------- (2) departments
    dep_n = v["department"].value_counts().sort_index()
    share = v.groupby("department")[theme_cols].mean() * 100
    lift = (v.groupby("department")[theme_cols].mean() / v[theme_cols].mean()).round(2)
    pv = perm_test(v, theme_cols)
    qv = pd.DataFrame(bh(pv.values.ravel()).reshape(pv.shape), index=pv.index, columns=pv.columns)
    share.round(1).to_csv(OUT / "s2_dept_theme_share.csv"); lift.to_csv(OUT / "s2_dept_theme_lift.csv")
    pv.round(4).to_csv(OUT / "s2_dept_theme_pvalues.csv"); qv.round(4).to_csv(OUT / "s2_dept_theme_qvalues_BH.csv")
    tone = v.groupby("department")[["praise_share", "concern_share", "suggestion_share"]].mean().round(2)
    tone["n_valid"] = dep_n; tone.to_csv(OUT / "s2_dept_tone.csv")
    print("\nValid responses per dept:", dep_n.to_dict()); print("\nDept tone\n", tone.to_string())
    print("\nSignificant dept x theme cells (BH q<0.10):")
    for d in qv.index:
        for t in qv.columns:
            if qv.loc[d, t] < 0.10:
                print(f"  {d}: {t:38s} share={share.loc[d, t]:.0f}% vs company {overall[t]*100:.0f}%  (lift {lift.loc[d, t]}, q={qv.loc[d, t]:.3f})")

    # ---------------- (3) individual profiles (bonus)
    feat = v[theme_cols + ["praise_share", "concern_share", "suggestion_share"]].copy()
    feat["log_len"] = np.log1p(v["n_words"]); feat = (feat - feat.mean()) / feat.std().replace(0, 1)
    best = None
    for kk in range(3, 7):
        km = KMeans(kk, n_init=20, random_state=SEED).fit(feat); sil = silhouette_score(feat, km.labels_)
        print(f"  k-means k={kk}: silhouette={sil:.3f}")
        if best is None or sil > best[0] + 0.01: best = (sil, kk, km)
    sil, kk, km = best; v["persona"] = km.labels_
    prof = v.groupby("persona")[theme_cols + ["praise_share", "concern_share", "suggestion_share", "n_words"]].mean()
    print(f"\nPersonas: k={kk}, silhouette={sil:.2f}")
    for pcl, g in v.groupby("persona"):
        dev = (prof.loc[pcl, theme_cols] - v[theme_cols].mean()).sort_values(ascending=False)
        print(f"\nPersona {pcl} (n={len(g)}; depts {g.department.value_counts().to_dict()}) praise={g.praise_share.mean():.2f} "
              f"concern={g.concern_share.mean():.2f} sugg={g.suggestion_share.mean():.2f} words={g.n_words.mean():.0f}")
        print("   over-indexes on:", ", ".join(f"{a} (+{b:.2f})" for a, b in dev.head(3).items()))
        for tx in g.sort_values("n_words", ascending=False)["text"].head(2): print("   e.g.", tx[:130])
    v[["id", "department", "text", "persona", "praise_share", "concern_share", "suggestion_share", "alert"] + theme_cols] \
        .to_csv(OUT / "s2_employee_profiles.csv", index=False)
    print("\nResponses flagged for HR/ethics review (not modelled):")
    for _, r in v[v.alert].iterrows(): print(f"  id={r['id']} {r.department}: {r.text[:100]}...")

    # ---------------- figures
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ann = share.round(0).astype(int).astype(str) + "%"
    ann = ann.where(qv >= 0.10, ann + "*")
    sns.heatmap(share[overall.index].T, annot=ann[overall.index].T, fmt="", cmap="Blues", cbar_kws={"label": "% of dept's valid responses"}, ax=ax)
    ax.set_xticklabels([f"{d}\n(n={dep_n[d]})" for d in share.index], rotation=0); ax.set_ylabel("")
    ax.set_title("Themes by department  (* = differs from rest of company, BH q<0.10)")
    plt.tight_layout(); plt.savefig(OUT / "s2_dept_theme_heatmap.png"); plt.close()

    fig, ax = plt.subplots(figsize=(9, 5))
    st = (stance_by_theme[["praise", "suggestion", "concern", "neutral"]] * 100).loc[overall.index][::-1]
    st.plot.barh(stacked=True, ax=ax, color=["#4daf4a", "#ffb000", "#e41a1c", "#cccccc"])
    ax.set_xlabel("% of segments"); ax.set_ylabel(""); ax.set_title("What is said about each theme: praise vs suggestion vs concern")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(.5, -.25), frameon=False)
    plt.tight_layout(); plt.savefig(OUT / "s2_theme_stance.png"); plt.close()

    from sklearn.decomposition import PCA
    xy = PCA(2, random_state=SEED).fit_transform(feat)
    fig, ax = plt.subplots(figsize=(7, 5))
    for pcl in sorted(v.persona.unique()):
        ax.scatter(*xy[(v.persona == pcl).values].T, label=f"Persona {pcl}", s=28, alpha=.8)
    ax.set_title("Employee personas (PCA of theme + tone features)"); ax.legend(frameon=False)
    plt.tight_layout(); plt.savefig(OUT / "s2_personas.png"); plt.close()
    for f in ("_W.npy", "_seg.pkl", "_df.pkl"): (OUT / f).unlink(missing_ok=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int); ap.add_argument("--scan", action="store_true")
    a = ap.parse_args(); main(a.k, a.scan)
