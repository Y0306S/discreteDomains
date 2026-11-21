# PT4Opt external baselines

Setup for the 10 external baselines listed in `pt4opt-baselines.md`, to be compared against
our method:

| Field | Value |
|---|---|
| Adapter | `llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2_b_prefix_grpo_rsmix_v4_pareto32_v12/ckpt32` |
| Prompt variant | `P5plusA5T50X50RS5f05v4` |
| Hybrid / planner | `HPUCB_arm` / `ptllm_tpe_qwen` |
| Optimizer label | `ptllm_tpe_qwen__P5plusA5T50X50RS5f05v4__HPUCB_arm` |
| Study output | `/root/data/cys/blockdata_archive/llmoptimizer_benchmark/benchmark_data/real_hpo_v2_mltest20_algo40_p5plus_grpo_rsmix_v12_p5_s1b_attr_t2_b_prefix_grpo_rsmix_v4_pareto32_v12_ckpt32/` |
| Protocol | 40 tasks (20 cls + 20 rgs) × 80 trials, seed 1, `tasks_40.json` |

## Layout

```
baseline/
├── README.md                # this file
├── MANIFEST.md              # baseline → repo → env → scenario table
├── repos/                   # upstream source, unmodified
├── envs/                    # one venv/conda env per baseline
├── logs/                    # clone + env build logs
└── build_env_*.sh           # reproducible env builds
```

**`repos/` is kept pristine** — no upstream file is edited. Version pins come from each
repo's own `environment.yaml` / `requirements.txt`; anything we must add to make our
scenarios work goes in adapter code outside `repos/`.

## Network notes (this host)

`github.com` and `pypi.org` are **not** routable from here. Working paths:

| Purpose | Endpoint |
|---|---|
| git clone | `https://gh-proxy.com/https://github.com/<org>/<repo>.git` |
| pip | `https://mirrors.aliyun.com/pypi/simple/` (+ `https://mirror.nju.edu.cn/pypi/web/simple/`) |
| conda | `https://mirrors.bfsu.edu.cn/anaconda/...` |
| HuggingFace | `https://hf-mirror.com` |
| Google Cloud Storage | `https://storage.googleapis.com` (reachable — OptFormer ckpts/datasets) |

Written to `/root/.config/pip/pip.conf` and `/root/.condarc`.
`ghproxy.net` and `ghfast.top` answer HTTP but stall on git clone — use `gh-proxy.com`.
