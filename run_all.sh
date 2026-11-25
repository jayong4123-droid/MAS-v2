#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python src/s2_feedback.py                 # Statement 2 (major)
python src/s3_extract_retrieve.py         # Statement 3: extract + retrieve evidence pack
cd src && python s3_verify.py             # Statement 3: verify the reference answer's citations
