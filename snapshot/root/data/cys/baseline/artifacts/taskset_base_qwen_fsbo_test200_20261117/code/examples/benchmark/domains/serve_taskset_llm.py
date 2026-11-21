#!/usr/bin/env python3
"""Explicitly start one TaskSet model service; never started by imports/configs."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from mindware.domains.dlhpo.taskset_llm.inference import make_server, model_identity, fingerprint
from mindware.domains.dlhpo.taskset_llm.llm_client import BASE_MODEL, QwenLoRAClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--adapter", type=Path)
    model.add_argument("--full-model", type=Path, help="Completed BOLT full SFT (preset generation only)")
    model.add_argument("--base-only", action="store_true", help="Serve the clean base checkpoint without an adapter")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--identity-out", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--disable-thinking", action="store_true",
                        help="Use Qwen's explicit empty thinking-block generation prefix")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-pending", type=int, default=16,
                        help="Bounded number of queued plus active generation requests")
    args = parser.parse_args()
    client = QwenLoRAClient(str(args.adapter) if args.adapter else None,
                            base_path=str(args.full_model or args.base_model),
                            temperature=args.temperature, max_new_tokens=args.max_new_tokens,
                            disable_thinking=args.disable_thinking)
    client._load()  # Health becomes ready only after exact adapter verification.
    identity = model_identity(client)
    server = make_server((args.host, args.port), client, identity, args.token_file.read_text().strip(),
                         max_pending=args.max_pending)
    args.identity_out.parent.mkdir(parents=True, exist_ok=True)
    args.identity_out.write_text(json.dumps({"identity": identity, "identity_sha256": fingerprint(identity)}, indent=2) + "\n")
    print(json.dumps({"ready": True, "address": server.server_address, "identity_sha256": fingerprint(identity)}), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
