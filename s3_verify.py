"""
Statement 3 - step 3: anti-hallucination check.  Every claim must carry a verbatim quote and page;
this script confirms each quote really appears on that page of the source PDF (whitespace/quote-normalised).

Run: python src/s3_verify.py [answers/comparison.json]   -> outputs/s3_verification.csv, outputs/s3_comparison.md
Use it on a ChatGPT/other-LLM answer too: ask the model for the same JSON schema (see prompts/), save, verify.
Exit code 1 if any quote fails -> usable as a CI gate.
"""
import json, sys
from pathlib import Path
import pandas as pd
from s3_extract_retrieve import DOCS, pages, norm

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "outputs"


def main(path):
    items = json.load(open(path)); P = {d: {pg: norm(t) for pg, t in pages(p)} for d, p in DOCS.items()}
    rows, md = [], ["# Swiss vs UK dormant-assets regimes: comparison (every claim quote-verified)\n"]
    for it in items:
        md.append(f"## {it['dimension']}")
        for side, doc in (("swiss", "SWISS"), ("uk", "UK")):
            c = it[side]
            if c is None:
                md.append(f"- **{doc.title() if doc=='UK' else 'Swiss'}:** _not addressed in this document_"); continue
            q = norm(c["quote"]); page = c["page"]
            ok = q in P[doc][page]; anywhere = [pg for pg, t in P[doc].items() if q in t]
            rows.append({"dimension": it["dimension"], "doc": doc, "page": page, "verified_on_cited_page": ok,
                         "found_on_pages": anywhere, "quote": c["quote"]})
            md.append(f"- **{'Swiss' if doc=='SWISS' else 'UK'}:** {c['claim']} _[{doc} p.{page}]_ — \"{c['quote']}\" {'✔' if ok else '✘ NOT FOUND'}")
        md.append(f"- **Difference:** {it['difference']}\n")
    df = pd.DataFrame(rows); df.to_csv(OUT / "s3_verification.csv", index=False)
    (OUT / "s3_comparison.md").write_text("\n".join(md))
    n_ok = int(df.verified_on_cited_page.sum()); print(f"{n_ok}/{len(df)} quotes verified on the cited page")
    print(df[~df.verified_on_cited_page].to_string() if n_ok < len(df) else "All citations check out.")
    return 0 if n_ok == len(df) else 1

if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "answers/comparison.json"))
