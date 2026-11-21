#!/usr/bin/env python3
"""Replay-parse raw LLM responses from existing ptllm_tpe__P5__H4 eval logs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from mindware.components.optimizers.pt4opt_llm_optimizer import parse_llm_json  # noqa: E402

BASE = REPO / "benchmark_data" / "real_hpo_v2"
VARIANT = "ptllm_tpe__P5__H4"


def _hp_subset(cfg: dict) -> dict:
  return {k: v for k, v in cfg.items() if k != "algorithm"}


def main() -> int:
  trial_files = sorted(BASE.glob(f"*/**/{VARIANT}/**/llm_conversations/trial_*.json"))
  if not trial_files:
    print(f"No trial logs found under {BASE} for {VARIANT}", file=sys.stderr)
    return 1

  ok = fail = skipped = 0
  failures: list[str] = []
  for path in trial_files:
    data = json.loads(path.read_text())
    attempts = data.get("llm_input", {}).get("attempts") or data.get("attempts") or []
    for attempt in attempts:
      raw = attempt.get("raw_response") or ""
      if not raw.strip():
        skipped += 1
        continue
      expected_payload = attempt.get("parsed_payload") or {}
      expected_cfg = expected_payload.get("config") or {}
      if not expected_cfg:
        skipped += 1
        continue
      try:
        cfg, rationale, _ = parse_llm_json(raw)
      except Exception as exc:
        fail += 1
        failures.append(f"{path.name}: {exc}")
        continue
      mism = {
        k: (expected_cfg.get(k), cfg.get(k))
        for k in expected_cfg
        if str(expected_cfg.get(k)) != str(cfg.get(k))
      }
      if mism:
        fail += 1
        failures.append(f"{path.name}: config mismatch {mism}")
      else:
        ok += 1
        if rationale and not attempt.get("parsed_payload", {}).get("rationale"):
          pass  # rationale recovered from prefix — fine

  print(f"Replay {VARIANT}: ok={ok} fail={fail} skipped={skipped} total={ok + fail + skipped}")
  if failures:
    print("\nFirst 10 failures:")
    for line in failures[:10]:
      print(f"  {line}")
    return 1
  print("All replayed responses parsed to the same config as the original run.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
