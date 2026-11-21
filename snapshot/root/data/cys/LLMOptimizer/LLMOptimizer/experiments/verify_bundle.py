#!/usr/bin/env python3
"""Check that the experiment source bundle is complete and import-free."""
from pathlib import Path
import ast, hashlib, json
ROOT=Path(__file__).resolve().parent
required={
 "query_fsbo": ["batch_code/run.py", "batch_code/method.py", "batch_code/hybrid_selector.py", "batch_code/qwen_inference_server.py"],
 "peptide_fsbo": ["code/evaluate.py", "code/pipeline.py", "code/retrain_schedule.py", "code/target_gate.py", "tests/retrain_schedule.json"],
}
for group, files in required.items():
    for rel in files:
        p=ROOT/group/rel
        if not p.exists(): raise SystemExit(f"missing {p}")
for p in ROOT.glob("**/*.py"):
    source = p.read_text()
    ast.parse(source)
    compile(source, str(p), "exec")
manifest = json.loads((ROOT / "manifest.json").read_text())
indexed = {row["path"]: row for row in manifest["files"]}
actual = {str(p.relative_to(ROOT)) for p in ROOT.rglob("*")
          if p.is_file() and p.name != "manifest.json" and "__pycache__" not in p.parts}
if actual != set(indexed):
    raise SystemExit(f"manifest file set differs: missing={set(indexed)-actual}, unindexed={actual-set(indexed)}")
for name in sorted(actual):
    if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != indexed[name]["sha256"]:
        raise SystemExit(f"manifest hash differs: {name}")
print(json.dumps({"passed":True,"python_files":len(list(ROOT.glob("**/*.py"))),"hashed_files":len(actual),"root":str(ROOT)}))
