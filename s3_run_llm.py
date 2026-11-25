"""
Statement 3 - step 2b: run an open-source LLM locally (Ollama) on the evidence pack, then verify.

  ollama pull llama3.1:8b          # free, open weights, runs on a laptop
  python src/s3_run_llm.py --dry-run          # just print the exact prompt (also paste into ChatGPT free plan)
  python src/s3_run_llm.py --model llama3.1:8b --out answers/llm_ollama.json
  python src/s3_verify.py answers/llm_ollama.json

NOTE: written against Ollama's documented /api/generate endpoint with the standard library only; it could not be
executed in the authoring sandbox (no network / no Ollama), so treat the HTTP call as untested.
"""
import argparse, json, re, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def build_prompt():
    sysmsg = (ROOT / "prompts/s3_system_prompt.txt").read_text()
    ev = (ROOT / "outputs/s3_evidence_pack.md").read_text()
    return f"{sysmsg}\n\nEVIDENCE:\n{ev}\n\nJSON:"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--out", default=str(ROOT / "answers/llm_ollama.json")); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(); prompt = build_prompt()
    if a.dry_run: print(prompt); raise SystemExit
    req = urllib.request.Request("http://localhost:11434/api/generate", method="POST",
        data=json.dumps({"model": a.model, "prompt": prompt, "stream": False, "format": "json",
                         "options": {"temperature": 0, "seed": 42}}).encode(),
        headers={"Content-Type": "application/json"})
    txt = json.loads(urllib.request.urlopen(req, timeout=600).read())["response"]
    data = json.loads(re.sub(r"^```json|```$", "", txt.strip()))
    data = data if isinstance(data, list) else next(v for v in data.values() if isinstance(v, list))
    Path(a.out).write_text(json.dumps(data, indent=1)); print("saved", a.out)
