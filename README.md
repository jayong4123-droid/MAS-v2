# Data Science Assessment: Statement 2 (major) + Statement 3 GenAI

Reproduce everything: `pip install -r requirements.txt && ./run_all.sh` (Python 3.10+, CPU only, ~1 min, no network).
Seeds are fixed (42). Outputs land in `outputs/`.

| Part | What | Entry point |
|---|---|---|
| Statement 2 (major) | Topics, department themes, individual profiles from free-text survey answers | `src/s2_feedback.py`, write-up in `REPORT_S2.md` |
| Statement 3 (GenAI) | Concise, source-cited, hallucination-controlled Swiss vs UK comparison | `src/s3_*.py`, `prompts/`, `answers/`, write-up below |

```
data/      inputs (DS2 xlsx, 2 PDFs)          src/      code
prompts/   LLM system prompt                  answers/  comparison.json (quote-cited answer)
outputs/   tables, figures, verification      REPORT_S2.md
```

---------------------------------------------------------------------------------------------------------------

## Statement 3: GenAI, how hallucination is controlled

**Method (4 steps).** (1) `s3_extract_retrieve.py` extracts text page by page and chunks it (UK by numbered paragraph,
e.g. 2.8; Swiss by bullet) into 59 page-tagged chunks. (2) TF-IDF retrieval pulls, per comparison dimension, the top
passages from *each* document into `outputs/s3_evidence_pack.md`. (3) An LLM answers **only from that pack** under
`prompts/s3_system_prompt.txt`: JSON output, a verbatim quote and page for every claim, `null` when a document is silent,
temperature 0. (4) `s3_verify.py` checks mechanically that every quote exists on the cited page of the PDF
(16/16 pass for the reference answer) and exits non-zero otherwise. Hallucinated or paraphrased "quotes" are
therefore caught automatically, for any model's output.

**Result** (full version with quotes: `outputs/s3_comparison.md`)

| | Switzerland (SBA factsheet, Dec 2015) | UK (HMT review of 2008 Act, Mar 2014) |
|---|---|---|
| Dormancy | 10 yrs "without contact" + 50 yrs "dormant" = 60 yrs | >15 yrs without customer-initiated activity |
| Participation | Described as obligations on Swiss banks; no opt-in mentioned | Voluntary opt-in; 10 institutions, >90% of personal deposits |
| Destination | Federal government (Federal Dept of Finance) | Reclaim Fund Ltd (non-profit); surplus to Big Lottery Fund |
| Owner's rights after transfer | Extinguished ("null and void") | Reclaimable at any time |
| Finding owners | Publication on dormantaccounts.ch at year 60 (>CHF 500 or unknown value; 1 yr, 5 yrs if last contact ≤1954); Ombudsman-searchable database; ≤CHF 500 transferred unpublished | Free "mylostaccount" tracing service, also covers accounts dormant <15 yrs |
| Evidence of outcomes | none given | ~£600m transferred, >£180m to Big Lottery Fund, ~£200m reunited with depositors |

**Headline:** the sharpest contrast is finality (Swiss transfer ends the claim; UK transfer does not) and timing (60 vs 15 years).
**Important caveat built into the answer:** the two PDFs are different kinds of document (industry factsheet of rules vs
government performance review), so some gaps (e.g. no UK fee rules, no Swiss outcome figures) reflect document scope, not law.

**Which model produced what (be explicit with reviewers).** The reference `answers/comparison.json` was drafted by
Claude following the protocol above and checked by the verifier, plus a manual read of each claim against the source. The
brief asks for free/open models: run `python src/s3_run_llm.py --dry-run`, paste the printed prompt into the ChatGPT free
plan (or run it locally with Ollama + Llama 3.1 8B via `s3_run_llm.py`), save the JSON reply as `answers/llm_<name>.json`,
and run `python src/s3_verify.py answers/llm_<name>.json`. *TODO before submitting: add the pass-rate of that run
here (the Ollama runner is untested in the authoring sandbox, which had no network).*

**Limitations met with the free/open route**
- Free-plan chat: no fixed model version, no seed/temperature control, context and message caps, outputs not reproducible run to run; local 8B models follow strict JSON/quote rules less reliably.
- Verification is quote-level (does the quote exist on the page?). It does not prove the *claim* is entailed by the quote; I covered that by manual review, but at scale you need an NLI/LLM-judge step.
- Source quirks: UK PDF has blank pages and footnote superscripts glued to text ("Act 20081"); Swiss p.5 is a flowchart image with no text layer (its content is duplicated in p.2/p.4 text and was cross-checked by eye); one UK "page" label differs from printed page numbers (PDF page = printed page + 4), so citations use PDF pages.
- TF-IDF retrieval is lexical; with only 59 chunks it works, at scale you'd want embeddings.
- Privacy: do not paste confidential policy documents into a consumer chat tool.

**If paid APIs were available** (design, not run): same pipeline with (a) embedding retrieval + reranker over a document
corpus; (b) a frontier model with native citation / structured-output features at temperature 0, forced to emit the JSON
schema; (c) a *second, independent* model pass acting as verifier (claim ⊨ quote? yes/no), rejecting or regenerating
unsupported claims; (d) the same string-match gate as `s3_verify.py`; (e) a small gold set of question→expected-fact
pairs scored for every model/prompt change (recall of facts, citation precision, hallucination rate); (f) logging of
model version, prompt hash and retrieved chunks for audit; (g) human sign-off before anything reaches a decision-maker.
Cost is small: a two-document comparison is ~10k input tokens per run.

---------------------------------------------------------------------------------------------------------------

## Statement 2: summary (details, figures and caveats in `REPORT_S2.md`)

Cleaning audit → idea-level segmentation (one answer often mixes praise + complaint + suggestion) → NMF discovery with k
chosen by coherence/stability → transparent 14-theme taxonomy for final labels (90% segment coverage; NMI 0.43 vs NMF) →
department comparison by permutation tests with Benjamini-Hochberg correction → k-means personas (bonus).
