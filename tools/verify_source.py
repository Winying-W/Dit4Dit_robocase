"""Verify all sealed delivery files without requiring training dependencies."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
expected=json.loads((ROOT/'SHA256SUMS.json').read_text())
errors=[]
for name,digest in expected.items():
    path=ROOT/name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
        errors.append(name)
if errors:raise SystemExit('Missing or modified delivery files: '+', '.join(errors))
print(f'Verified {len(expected)} delivery files')
