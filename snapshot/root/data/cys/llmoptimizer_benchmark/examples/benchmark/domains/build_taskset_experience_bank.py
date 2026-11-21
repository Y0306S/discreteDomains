#!/usr/bin/env python3
"""Build the TaskSet CASH experience bank from the train838 TPE BO corpus.

Writes <out>.jsonl (one card per train task, schema taskset_experience.v2),
<out>.profile_stats.json (within-family spec standardization) and
<out>.manifest.json (provenance). Test tasks never enter the bank.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mindware.domains.dlhpo.taskset_llm import experience as E, spec as S
from mindware.domains.dlhpo.taskset_space import space_from_taskset
from taskset_transfer_common import METRIC, PROTOCOL, SPLIT_SHA256, digest, load_corpus, read_split

CORPUS = Path("/root/data/logs/taskset_transfer_train838_20261002/corpus")
OUT = ROOT / "history_data/domain_taskset/experience_bank_train838.jsonl"


def build(corpus, out, check_sources):
    manifest, tasks = load_corpus(corpus, check_sources=check_sources)
    split = read_split()
    if set(tasks) & set(split["test"]):
        raise ValueError("Test tasks in the experience-bank corpus")
    specs, cs = S.load_task_specs(), space_from_taskset()
    cards = [E.build_card(t, r, specs[t], cs) for t, r in tasks.items()]
    stats = E.profile_stats({c["problem"]["task_id"]: c["problem"]["features"] for c in cards})
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as stream:
        for card in cards:
            stream.write(json.dumps(card, sort_keys=True) + "\n")
    stats_path = out.with_name(out.stem + ".profile_stats.json")
    stats_path.write_text(json.dumps(stats, sort_keys=True) + "\n")
    report = {"schema_version": E.BANK_SCHEMA, "n_cards": len(cards), "split_sha256": SPLIT_SHA256,
              "corpus": str(corpus), "corpus_manifest_sha256": digest(corpus / "manifest.json"),
              "protocol": PROTOCOL, "metric": METRIC,
              "code_sha256": {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                              for p in (__file__, E.__file__, S.__file__)},
              "files": {p.name: digest(p) for p in (out, stats_path)},
              "cards_with_regions": sum(bool(c["regions"]["family_level"] or c["regions"]["by_family"]) for c in cards),
              "family_region_counts": {f: sum(f in c["regions"]["by_family"] for c in cards)
                                       for f in sorted({f for c in cards for f in c["families"]})},
              "cards_with_family_level_line": sum(bool(c["regions"]["family_level"]) for c in cards)}
    out.with_name(out.stem + ".manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--skip-source-check", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.corpus, args.out, not args.skip_source_check), indent=2))
