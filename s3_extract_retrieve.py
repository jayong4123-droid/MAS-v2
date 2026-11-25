"""
Statement 3 (GenAI) - step 1+2: extract page-tagged text, chunk, and build a retrieval index.

Run: python src/s3_extract_retrieve.py            -> outputs/s3_chunks.csv, outputs/s3_evidence_pack.md
     python src/s3_extract_retrieve.py "query"    -> top evidence chunks for an ad-hoc question

Why retrieval first? Hallucination is lowest when the LLM only sees short, page-tagged passages and is told
to answer from them. The evidence pack is what you paste into a free-tier chat model (see prompts/).
"""
import re, sys
from pathlib import Path
import pandas as pd, pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "outputs"; OUT.mkdir(exist_ok=True)
DOCS = {"SWISS": ROOT / "data/DS3-Swiss-Dormant-Assets.pdf", "UK": ROOT / "data/DS3-UK-Dormant-Assets.pdf"}


def norm(s: str) -> str:
    """Canonical form used for BOTH chunking and quote verification (de-hyphenate, collapse whitespace/quotes)."""
    s = s.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2013", "-").replace("\u2022", " ").replace("\uf03d", " ")
    return re.sub(r"\s+", " ", s).strip()


def pages(pdf_path):
    with pdfplumber.open(pdf_path) as p:
        return [(i, p_.extract_text() or "") for i, p_ in enumerate(p.pages, 1)]


def chunk_page(doc, page, text):
    if doc == "UK":      # statute-review paragraphs are numbered (1.1, 2.8 ...): natural, citable units
        parts = re.split(r"\n(?=\d\.\d+\s)", text)
    else:                # Swiss factsheet: bullets / short paragraphs
        parts = re.split(r"\n(?=\u2022|[A-Z][a-z]+ [a-z]+\n)|\n\n", text)
    out = []
    for p in parts:
        t = norm(p)
        if len(t) < 40: continue
        m = re.match(r"(\d\.\d+)\s", t) if doc == "UK" else None
        out.append({"doc": doc, "page": page, "para": m.group(1) if m else "", "text": t})
    return out


def build():
    rows = [c for d, path in DOCS.items() for pg, tx in pages(path) for c in chunk_page(d, pg, tx)]
    df = pd.DataFrame(rows); df["chunk_id"] = [f"{r.doc}-p{r.page}-{i}" for i, r in enumerate(df.itertuples())]
    return df


class Retriever:
    def __init__(self, df):
        self.df = df
        self.v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.X = self.v.fit_transform(df["text"])

    def search(self, q, doc=None, k=3):
        s = (self.X @ self.v.transform([q]).T).toarray().ravel()
        d = self.df.assign(score=s)
        if doc: d = d[d.doc == doc]
        return d.sort_values("score", ascending=False).head(k)


DIMENSIONS = {
    "Dormancy trigger & timeline": "when is an account classed as dormant years without contact or customer activity",
    "Who is covered / participation": "which banks building societies opt into scheme participants",
    "Where the money goes": "transfer of assets to federal government reclaim fund Big Lottery Fund",
    "Rights of the owner after transfer": "claims reclaim after transfer null and void",
    "Publication, tracing & searching": "publish dormant accounts website tracing service search ombudsman",
    "Treatment of assets while dormant, fees": "savings accounts interest invested costs fees charged",
    "Scale / evidence of outcomes": "amount transferred million reunited depositors",
}

if __name__ == "__main__":
    df = build(); df.to_csv(OUT / "s3_chunks.csv", index=False)
    R = Retriever(df)
    if len(sys.argv) > 1:
        for doc in DOCS:
            for r in R.search(" ".join(sys.argv[1:]), doc).itertuples():
                print(f"[{doc} p.{r.page} {r.para}] ({r.score:.2f}) {r.text[:200]}")
    else:
        lines = ["# Evidence pack (auto-retrieved, page-tagged)\n"]
        for dim, q in DIMENSIONS.items():
            lines.append(f"\n## {dim}")
            for doc in DOCS:
                for r in R.search(q, doc, k=3).itertuples():
                    ref = f"{doc} p.{r.page}" + (f" para {r.para}" if r.para else "")
                    lines.append(f"- **[{ref}]** {r.text}")
        (OUT / "s3_evidence_pack.md").write_text("\n".join(lines))
        print(f"{len(df)} chunks -> outputs/s3_chunks.csv ; evidence pack -> outputs/s3_evidence_pack.md")
