#!/usr/bin/env python3
"""Build hash/LLM semantic profile embeddings + TUNE-only VAE encoder.

Does not call an embedding API unless --llm is set. Hash embeddings are the
offline default so this is reproducible without network.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_BENCH = Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from dataset_split import is_tune, load_split
from mindware.components.experience.profile import profile_vector
from mindware.components.experience.retrieve import ExperienceBank
from mindware.components.experience.semantic_embed import (
    EmbeddingCache,
    TinyVAE,
    default_cache_path,
    default_vae_path,
    embed_profile,
    pad_views,
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, default=ROOT / "history_data" / "experience_bank_v2.jsonl")
    p.add_argument("--llm", action="store_true", help="Try OpenAI embeddings (falls back to hash)")
    args = p.parse_args(argv)

    bank = ExperienceBank.load(args.bank)
    split = load_split()
    client = None
    if args.llm:
        try:
            from openai import OpenAI
            from mindware.modules.configs import AVAILABLE_LLMs, LLMCONFIGS

            cfg = AVAILABLE_LLMs.get(LLMCONFIGS.kd_chat_engine) or {}
            client = OpenAI(base_url=cfg.get("base_url"), api_key=cfg.get("api_key"))
        except Exception as exc:
            print(f"LLM client unavailable ({exc}); using hash embeddings")
            client = None

    caches = {
        "classification": EmbeddingCache(default_cache_path("cls", ROOT)),
        "regression": EmbeddingCache(default_cache_path("rgs", ROOT)),
    }
    vae_rows = {"classification": [], "regression": []}
    seen = set()
    for exp in bank.experiences:
        prob = exp.get("problem") or {}
        ds = str(prob.get("dataset_id") or "")
        task = str(prob.get("task_type") or "classification")
        key = (task, ds.lower())
        if not ds or key in seen:
            continue
        seen.add(key)
        cache = caches.get(task) or caches["classification"]
        vec = embed_profile(
            ds,
            (exp.get("retrieval") or {}).get("profile_vec") or (prob.get("profile") or {}),
            task,
            algorithm_id=prob.get("algorithm_id"),
            client=client,
        )
        cache.put(task, ds, vec)
        ret = exp.setdefault("retrieval", {})
        ret["semantic_vec"] = vec
        if is_tune(ds, split):
            view = pad_views(vec, list(profile_vector(ret.get("profile_vec") or {})))
            if view:
                vae_rows.setdefault(task, []).append(view)

    for task, cache in caches.items():
        cache.save()
        print(f"semantic cache {cache.path} n={len(cache.data)}")

    for task, rows in vae_rows.items():
        prefix = "cls" if task.startswith("class") else "rgs"
        vae = TinyVAE()
        vae.fit(rows)
        path = default_vae_path(prefix, ROOT)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(vae))
        print(f"VAE {path} trained on {len(rows)} TUNE rows pca={vae.use_pca}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Build hash/LLM semantic profile embeddings + TUNE-only VAE encoder.

Does not call an embedding API unless --llm is set. Hash embeddings are the
offline default so this is reproducible without network.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_BENCH = Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from dataset_split import is_tune, load_split
from mindware.components.experience.profile import profile_vector
from mindware.components.experience.retrieve import ExperienceBank
from mindware.components.experience.semantic_embed import (
    EmbeddingCache,
    TinyVAE,
    default_cache_path,
    default_vae_path,
    embed_profile,
    pad_views,
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, default=ROOT / "history_data" / "experience_bank_v2.jsonl")
    p.add_argument("--llm", action="store_true", help="Try OpenAI embeddings (falls back to hash)")
    args = p.parse_args(argv)

    bank = ExperienceBank.load(args.bank)
    split = load_split()
    client = None
    if args.llm:
        try:
            from openai import OpenAI
            from mindware.modules.configs import AVAILABLE_LLMs, LLMCONFIGS

            cfg = AVAILABLE_LLMs.get(LLMCONFIGS.kd_chat_engine) or {}
            client = OpenAI(base_url=cfg.get("base_url"), api_key=cfg.get("api_key"))
        except Exception as exc:
            print(f"LLM client unavailable ({exc}); using hash embeddings")
            client = None

    caches = {
        "classification": EmbeddingCache(default_cache_path("cls", ROOT)),
        "regression": EmbeddingCache(default_cache_path("rgs", ROOT)),
    }
    vae_rows = {"classification": [], "regression": []}
    seen = set()
    for exp in bank.experiences:
        prob = exp.get("problem") or {}
        ds = str(prob.get("dataset_id") or "")
        task = str(prob.get("task_type") or "classification")
        key = (task, ds.lower())
        if not ds or key in seen:
            continue
        seen.add(key)
        cache = caches.get(task) or caches["classification"]
        vec = embed_profile(
            ds,
            (exp.get("retrieval") or {}).get("profile_vec") or (prob.get("profile") or {}),
            task,
            algorithm_id=prob.get("algorithm_id"),
            client=client,
        )
        cache.put(task, ds, vec)
        ret = exp.setdefault("retrieval", {})
        ret["semantic_vec"] = vec
        if is_tune(ds, split):
            view = pad_views(vec, list(profile_vector(ret.get("profile_vec") or {})))
            if view:
                vae_rows.setdefault(task, []).append(view)

    for task, cache in caches.items():
        cache.save()
        print(f"semantic cache {cache.path} n={len(cache.data)}")

    for task, rows in vae_rows.items():
        prefix = "cls" if task.startswith("class") else "rgs"
        vae = TinyVAE()
        vae.fit(rows)
        path = default_vae_path(prefix, ROOT)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(vae))
        print(f"VAE {path} trained on {len(rows)} TUNE rows pca={vae.use_pca}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Build hash/LLM semantic profile embeddings + TUNE-only VAE encoder.

Does not call an embedding API unless --llm is set. Hash embeddings are the
offline default so this is reproducible without network.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_BENCH = Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from dataset_split import is_tune, load_split
from mindware.components.experience.profile import profile_vector
from mindware.components.experience.retrieve import ExperienceBank
from mindware.components.experience.semantic_embed import (
    EmbeddingCache,
    TinyVAE,
    default_cache_path,
    default_vae_path,
    embed_profile,
    pad_views,
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, default=ROOT / "history_data" / "experience_bank_v2.jsonl")
    p.add_argument("--llm", action="store_true", help="Try OpenAI embeddings (falls back to hash)")
    args = p.parse_args(argv)

    bank = ExperienceBank.load(args.bank)
    split = load_split()
    client = None
    if args.llm:
        try:
            from openai import OpenAI
            from mindware.modules.configs import AVAILABLE_LLMs, LLMCONFIGS

            cfg = AVAILABLE_LLMs.get(LLMCONFIGS.kd_chat_engine) or {}
            client = OpenAI(base_url=cfg.get("base_url"), api_key=cfg.get("api_key"))
        except Exception as exc:
            print(f"LLM client unavailable ({exc}); using hash embeddings")
            client = None

    caches = {
        "classification": EmbeddingCache(default_cache_path("cls", ROOT)),
        "regression": EmbeddingCache(default_cache_path("rgs", ROOT)),
    }
    vae_rows = {"classification": [], "regression": []}
    seen = set()
    for exp in bank.experiences:
        prob = exp.get("problem") or {}
        ds = str(prob.get("dataset_id") or "")
        task = str(prob.get("task_type") or "classification")
        key = (task, ds.lower())
        if not ds or key in seen:
            continue
        seen.add(key)
        cache = caches.get(task) or caches["classification"]
        vec = embed_profile(
            ds,
            (exp.get("retrieval") or {}).get("profile_vec") or (prob.get("profile") or {}),
            task,
            algorithm_id=prob.get("algorithm_id"),
            client=client,
        )
        cache.put(task, ds, vec)
        ret = exp.setdefault("retrieval", {})
        ret["semantic_vec"] = vec
        if is_tune(ds, split):
            view = pad_views(vec, list(profile_vector(ret.get("profile_vec") or {})))
            if view:
                vae_rows.setdefault(task, []).append(view)

    for task, cache in caches.items():
        cache.save()
        print(f"semantic cache {cache.path} n={len(cache.data)}")

    for task, rows in vae_rows.items():
        prefix = "cls" if task.startswith("class") else "rgs"
        vae = TinyVAE()
        vae.fit(rows)
        path = default_vae_path(prefix, ROOT)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(vae))
        print(f"VAE {path} trained on {len(rows)} TUNE rows pca={vae.use_pca}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Build hash/LLM semantic profile embeddings + TUNE-only VAE encoder.

Does not call an embedding API unless --llm is set. Hash embeddings are the
offline default so this is reproducible without network.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_BENCH = Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from dataset_split import is_tune, load_split
from mindware.components.experience.profile import profile_vector
from mindware.components.experience.retrieve import ExperienceBank
from mindware.components.experience.semantic_embed import (
    EmbeddingCache,
    TinyVAE,
    default_cache_path,
    default_vae_path,
    embed_profile,
    pad_views,
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, default=ROOT / "history_data" / "experience_bank_v2.jsonl")
    p.add_argument("--llm", action="store_true", help="Try OpenAI embeddings (falls back to hash)")
    args = p.parse_args(argv)

    bank = ExperienceBank.load(args.bank)
    split = load_split()
    client = None
    if args.llm:
        try:
            from openai import OpenAI
            from mindware.modules.configs import AVAILABLE_LLMs, LLMCONFIGS

            cfg = AVAILABLE_LLMs.get(LLMCONFIGS.kd_chat_engine) or {}
            client = OpenAI(base_url=cfg.get("base_url"), api_key=cfg.get("api_key"))
        except Exception as exc:
            print(f"LLM client unavailable ({exc}); using hash embeddings")
            client = None

    caches = {
        "classification": EmbeddingCache(default_cache_path("cls", ROOT)),
        "regression": EmbeddingCache(default_cache_path("rgs", ROOT)),
    }
    vae_rows = {"classification": [], "regression": []}
    seen = set()
    for exp in bank.experiences:
        prob = exp.get("problem") or {}
        ds = str(prob.get("dataset_id") or "")
        task = str(prob.get("task_type") or "classification")
        key = (task, ds.lower())
        if not ds or key in seen:
            continue
        seen.add(key)
        cache = caches.get(task) or caches["classification"]
        vec = embed_profile(
            ds,
            (exp.get("retrieval") or {}).get("profile_vec") or (prob.get("profile") or {}),
            task,
            algorithm_id=prob.get("algorithm_id"),
            client=client,
        )
        cache.put(task, ds, vec)
        ret = exp.setdefault("retrieval", {})
        ret["semantic_vec"] = vec
        if is_tune(ds, split):
            view = pad_views(vec, list(profile_vector(ret.get("profile_vec") or {})))
            if view:
                vae_rows.setdefault(task, []).append(view)

    for task, cache in caches.items():
        cache.save()
        print(f"semantic cache {cache.path} n={len(cache.data)}")

    for task, rows in vae_rows.items():
        prefix = "cls" if task.startswith("class") else "rgs"
        vae = TinyVAE()
        vae.fit(rows)
        path = default_vae_path(prefix, ROOT)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(vae))
        print(f"VAE {path} trained on {len(rows)} TUNE rows pca={vae.use_pca}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
