"""Run BOLT native LOLBO (GP + TuRBO trust region + E2E VAE) on benchmark peptides."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from mindware.domains.history import observation, write_openbox_history
from mindware.domains.paths import BOLT_PEPTIDES, domain_bench, protein_bo80_bolt_root
from mindware.domains.protein.bolt_constraints import (
    bolt_reference_sequence,
    template_similarity,
)
from mindware.domains.protein.bolt_seed0_init import (
    BOLT_FULL_ORACLE_BUDGET,
    BOLT_SEED0_HPARAMS,
    CHECKPOINT_ORACLE_CALLS,
    bolt_maximize_score_to_mic,
    load_seed0_pool,
    total_trial_budget,
)
from mindware.domains.protein.bolt_lolbo_checkpoint import (
    checkpoint_path,
    ensure_checkpoint,
    load_checkpoint,
    restore_vae_to_objective,
    save_checkpoint,
    vae_state_dict_from_objective,
)
from mindware.domains.protein.bolt_peptide_e2e import (
    reencode_lolbo_state_latents,
    refit_lolbo_surrogate_on_all,
)

BOLT_PEPTIDE_ROOT = Path(os.environ.get("BOLT_PEPTIDES", str(BOLT_PEPTIDES)))
LOLBO_SCRIPTS = BOLT_PEPTIDE_ROOT / "lolbo_scripts"
_VAE_DIR = BOLT_PEPTIDE_ROOT / "uniref_vae/saved_models"
_VAE_STEM = "dim128_k1_kl0001_eff256_dff256_pious-sea-2_model_state_epoch_118"
# BOLT's peptide VAE is APEXGo's checkpoint (Wan et al. 2024 / Torres et al. 2024);
# `<stem>_pretrained_authors.pkl` is byte-identical to
# APEXGo/generation/checkpoints/<stem>.pkl. The un-suffixed file on this box is a
# LOCAL retrain: same architecture, but 183/185 tensors differ and it only round-trips
# 874/1000 init sequences vs 1000/1000 for the authors' weights. Default to the
# authors' checkpoint; override with BOLT_VAE_CKPT to reproduce older runs.
VAE_CKPT = Path(
    os.environ.get("BOLT_VAE_CKPT", str(_VAE_DIR / f"{_VAE_STEM}_pretrained_authors.pkl"))
)

CONSISTENT10 = [
    "pep_009",
    "pep_791",
    "pep_231",
    "pep_291",
    "pep_277",
    "pep_726",
    "pep_012",
    "pep_813",
    "pep_474",
    "pep_124",
]


def _ensure_bolt_paths() -> None:
    for p in (str(BOLT_PEPTIDE_ROOT), str(LOLBO_SCRIPTS)):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    for key in list(sys.modules):
        if key == "lolbo" or key.startswith("lolbo."):
            del sys.modules[key]


def _protein_seqs(limit: int, offset: int) -> List[str]:
    seqs, _ = load_seed0_pool(limit=int(limit) + int(offset))
    return seqs[int(offset) : int(offset) + int(limit)]


def _parse_pep_idx(task_id: str) -> int:
    return int(str(task_id).strip().split("_")[-1])


def _z_config(z_vec: torch.Tensor, dim: int) -> Dict[str, float]:
    arr = z_vec.detach().cpu().float().reshape(-1).tolist()
    return {f"z_{i}": float(arr[i]) for i in range(min(dim, len(arr)))}


def _append_index(path: Path, row: dict) -> None:
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, default=str) + "\n"
    with path.open("a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _task_ids(n_tasks: int, offset: int, shard_index: int, n_shards: int) -> List[str]:
    all_ids = [f"pep_{i:03d}" for i in range(int(n_tasks))]
    if int(n_shards) <= 1:
        return all_ids[int(offset) :]
    return [tid for i, tid in enumerate(all_ids) if i % int(n_shards) == int(shard_index)]


class BenchmarkPeptideLOLBO:
    """Wrapper around BOLT APEXConstrainedDiverseOptimization with benchmark export."""

    def __init__(
        self,
        *,
        pep_task_id: str,
        start_seq: str,
        n_trials: int = 500,
        seed: Optional[int] = None,
        device_id: int = 0,
        bsz: int = 50,
        e2e_freq: int = 10,
        init_n_update_epochs: int = 20,
        num_initialization_points: int = 1000,
        max_string_length: int = 30,
        verbose: bool = True,
        oracle_mode: str = "bolt_native",
        min_similarity: float = 0.75,
        constraint_index: int = 0,
        pep_idx: Optional[int] = None,
        run_mode: str = "checkpoint",
        strict_oracle_budget: Optional[bool] = None,
        update_e2e: Optional[bool] = None,
        max_oracle_budget: Optional[int] = None,
        checkpoint_calls: int = CHECKPOINT_ORACLE_CALLS,
        init_data_path: Optional[str] = None,
        init_scores_path: Optional[str] = None,
        init_offset_helper: int = 0,
        require_feasible_init: bool = True,
        budget_mode: str = "acquisition",
    ) -> None:
        _ensure_bolt_paths()
        os.chdir(BOLT_PEPTIDE_ROOT)
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(int(device_id))

        from lolbo_scripts.info_transformer_vae_optimization import APEXConstrainedDiverseOptimization

        self.pep_task_id = pep_task_id
        self.start_seq = start_seq
        self.pep_idx = pep_idx
        self.n_trials = int(n_trials)
        self.num_init_points = int(num_initialization_points)
        self.oracle_mode = str(oracle_mode or "bolt_native").lower()
        self.min_similarity = float(min_similarity)
        # Each pep_* task uses REFERENCE_SEQUENCE[pep_idx] for both the similarity
        # constraint and the init pool. Upstream BOLT selects the init block with
        #     bacteria_num = constraint_types[0] - init_offset_helper
        # so init_offset_helper is a property of the *init file* (the index of the first
        # reference whose block it contains), NOT of the task. Setting it per-task to
        # pep_idx pins every task to block 0, which leaves 999/1000 references with zero
        # constraint-feasible init points. See docs/BOLT_PEPTIDE_BO_FIX_PLAN.md.
        self.constraint_index = int(pep_idx) if pep_idx is not None else int(constraint_index)
        self._init_offset_helper = int(init_offset_helper)
        self.require_feasible_init = bool(require_feasible_init)
        self._seconds = 0.0
        self._lolbo_state = None
        self._objective = None
        self._init_score: Optional[float] = None
        self._n_init_loaded: int = 0
        self._n_oracle_calls: int = 0
        self._truncated: Optional[str] = None
        self._n_init_feasible: Optional[int] = None
        self._init_max_similarity: Optional[float] = None
        self._n_acq_calls: int = 0
        self._n_e2e_calls: int = 0
        self._n_rounds: int = 0
        # Provenance for every row appended to train_x, as half-open spans.
        # train_x order is the evaluation order, but nothing upstream records WHICH rows
        # were proposed together, nor whether a row came from acquisition or from the E2E
        # recenter (which re-appends top_k every epoch and outnumbers acquisition). Without
        # this you cannot rebuild an acquisition batch, so no winbatch-style SFT target and
        # no round-level preference pairs. Each span:
        #   {start, stop, round, phase, batch_id, t, incumbent_before}
        self._spans: List[Dict[str, Any]] = []
        self.budget_mode = str(budget_mode or "acquisition").lower()
        if self.budget_mode not in {"acquisition", "total"}:
            raise ValueError(f"budget_mode must be 'acquisition' or 'total', got {budget_mode!r}")
        self.run_mode = str(run_mode or "checkpoint").lower()
        self.checkpoint_calls = int(checkpoint_calls)
        if self.run_mode == "continue_bolt":
            self.strict_oracle_budget = False if strict_oracle_budget is None else bool(strict_oracle_budget)
            self.max_oracle_budget = int(max_oracle_budget or BOLT_FULL_ORACLE_BUDGET)
            self.n_trials = self.max_oracle_budget
        else:
            self.strict_oracle_budget = True if strict_oracle_budget is None else bool(strict_oracle_budget)
            self.max_oracle_budget = int(max_oracle_budget or self.n_trials)
        # Encode-only E2E does not spend oracle, so keep it on unless explicitly disabled.
        self._use_e2e = True if update_e2e is None else bool(update_e2e)
        self._resume_ckpt: Optional[Dict[str, Any]] = None

        if self.oracle_mode == "bolt_native":
            hps = dict(BOLT_SEED0_HPARAMS)
            hps["constraint_thresholds"] = [self.min_similarity]
            hps["constraint_types"] = [self.constraint_index]
            hps["num_initialization_points"] = self.num_init_points
            hps["bsz"] = int(bsz)
            hps["e2e_freq"] = int(e2e_freq)
            hps["init_n_update_epochs"] = int(init_n_update_epochs)
            hps["max_string_length"] = int(max_string_length)
            hps["path_to_vae_statedict"] = str(VAE_CKPT)
            hps["init_data_path"] = str(init_data_path or BOLT_SEED0_HPARAMS["init_data_path"])
            hps["init_scores_path"] = str(init_scores_path or BOLT_SEED0_HPARAMS["init_scores_path"])
            hps["init_offset_helper"] = int(self._init_offset_helper)
            # Record what actually ran, not the module defaults (the old code exported
            # BOLT_SEED0_HPARAMS verbatim, so every row claimed bsz=50 regardless).
            self._hps_used = dict(hps)

            class _BoltNativeRunner(APEXConstrainedDiverseOptimization):
                def __init__(self_inner, outer: BenchmarkPeptideLOLBO, **kwargs):
                    self_inner._outer = outer
                    super().__init__(**kwargs)

                def initialize_objective(self_inner):
                    super().initialize_objective()
                    obj = self_inner.objective
                    ys = self_inner.init_train_y.detach().cpu().float().reshape(-1)
                    for x, y in zip(self_inner.init_train_x, ys):
                        obj.xs_to_scores_dict[str(x)] = float(y.item())
                    self_inner._outer._n_init_loaded = len(self_inner.init_train_x)
                    self_inner._outer._assert_feasible_init(self_inner.init_train_x)
                    return self_inner

                def load_train_z(self_inner, *args, **kwargs):
                    """Never reuse the global `<vae_statedict>-train-zs.csv` latent cache.

                    Upstream keys this cache on the VAE state dict alone, so it is shared by
                    every task, and it only validates `len(zs) >= num_initialization_points`
                    -- never that the latents correspond to the sequences. With per-reference
                    init pools that pairs reference 0's latents with every other reference's
                    sequences, which is silent and total: the GP is fit on latents that do not
                    encode the x/y it is trained against, and the trust region is centred on a
                    foreign latent.

                    Overriding `compute_train_zs` was not enough: `initialize_train_data`
                    calls `load_train_z` first, and when the cache hits, `init_train_z` is
                    non-None so `compute_train_zs` is never reached. Force it to None here so
                    the pool is always encoded with the VAE actually in use (~1s), and delete
                    the file so nothing downstream can pick it up.
                    """
                    # load_train_z runs before initialize_objective, so self.objective may
                    # not exist yet; the path lives on the optimizer in both phases.
                    _sd = getattr(
                        getattr(self_inner, "objective", None),
                        "path_to_vae_statedict",
                        None,
                    ) or self_inner.path_to_vae_statedict
                    stale = Path(str(Path(_sd).with_suffix("")) + "-train-zs.csv")
                    try:
                        if stale.is_file():
                            stale.unlink()
                    except OSError:
                        pass
                    self_inner.init_train_z = None
                    return self_inner

                def compute_train_zs(self_inner, *args, **kwargs):
                    """Encode the init pool, then remove the cache super() just wrote."""
                    zs = super().compute_train_zs(*args, **kwargs)
                    # load_train_z runs before initialize_objective, so self.objective may
                    # not exist yet; the path lives on the optimizer in both phases.
                    _sd = getattr(
                        getattr(self_inner, "objective", None),
                        "path_to_vae_statedict",
                        None,
                    ) or self_inner.path_to_vae_statedict
                    stale = Path(str(Path(_sd).with_suffix("")) + "-train-zs.csv")
                    try:
                        if stale.is_file():
                            stale.unlink()
                    except OSError:
                        pass
                    return zs

                def _remaining_calls(self_inner) -> int:
                    return max(
                        0,
                        int(self_inner.max_n_oracle_calls)
                        - int(self_inner.lolbo_state.objective.num_calls),
                    )

                def _target_acq_rounds(self_inner) -> int:
                    bsz = max(1, int(self_inner.lolbo_state.bsz))
                    return max(1, int(self_inner.max_n_oracle_calls) // bsz)

                def _scheduled_e2e_reencode(self_inner, n_rounds: int) -> int:
                    """Every e2e_freq acquisition rounds: finetune VAE, remap x->z, no oracle."""
                    if not self_inner.update_e2e:
                        return 0
                    freq = max(1, int(self_inner.e2e_freq))
                    if n_rounds <= 0 or (n_rounds % freq) != 0:
                        return 0
                    pre = int(self_inner.lolbo_state.objective.num_calls)
                    n_x = len(self_inner.lolbo_state.train_x)
                    if not self_inner.recenter_only:
                        self_inner.lolbo_state.update_models_e2e()
                    n_enc = reencode_lolbo_state_latents(self_inner.lolbo_state)
                    refit_lolbo_surrogate_on_all(self_inner.lolbo_state)
                    post = int(self_inner.lolbo_state.objective.num_calls)
                    if post != pre or len(self_inner.lolbo_state.train_x) != n_x:
                        print(
                            json.dumps(
                                {
                                    "task_id": self_inner._outer.pep_task_id,
                                    "e2e_oracle_leak": True,
                                    "n_oracle_delta": int(post - pre),
                                    "n_train_x_delta": int(len(self_inner.lolbo_state.train_x) - n_x),
                                }
                            ),
                            flush=True,
                        )
                    print(
                        json.dumps(
                            {
                                "task_id": self_inner._outer.pep_task_id,
                                "e2e_reencode": True,
                                "round": int(n_rounds),
                                "e2e_updates": int(self_inner.lolbo_state.tot_num_e2e_updates),
                                "n_reencoded": int(n_enc),
                                "n_train_x": len(self_inner.lolbo_state.train_x),
                                "n_oracle_calls": post,
                            }
                        ),
                        flush=True,
                    )
                    return int(n_enc)

                def _hist_path(self_inner) -> Path:
                    return (
                        protein_bo80_bolt_root()
                        / self_inner._outer.pep_task_id
                        / "bolt_native"
                        / "seed1"
                        / "history.json"
                    )

                def _save_runtime_checkpoint(self_inner, *, force: bool = False) -> None:
                    state = self_inner.lolbo_state
                    obj = self_inner.objective
                    n_calls = int(obj.num_calls)
                    last_saved = int(getattr(self_inner, "_last_ckpt_calls", 0))
                    min_interval = 100 if self_inner._outer.run_mode == "continue_bolt" else 5000
                    if not force and (n_calls - last_saved) < min_interval:
                        return
                    hist = self_inner._hist_path()
                    save_checkpoint(
                        checkpoint_path(hist),
                        train_x=list(state.train_x),
                        train_y=state.train_y.detach().cpu(),
                        train_z=state.train_z.detach().cpu(),
                        train_c=state.train_c.detach().cpu() if state.train_c is not None else None,
                        num_calls=n_calls,
                        xs_to_scores_dict=dict(obj.xs_to_scores_dict),
                        n_init_loaded=int(self_inner._outer._n_init_loaded or self_inner._outer.num_init_points),
                        tot_num_e2e_updates=int(state.tot_num_e2e_updates),
                        best_x_seen=state.best_x_seen,
                        best_score_seen=state.best_score_seen,
                        minimize=bool(state.minimize),
                        vae_state_dict=vae_state_dict_from_objective(obj),
                        n_acq_calls=int(self_inner._outer._n_acq_calls),
                        n_e2e_calls=int(self_inner._outer._n_e2e_calls),
                        n_rounds=int(self_inner._outer._n_rounds),
                        spans=list(self_inner._outer._spans),
                    )
                    self_inner._last_ckpt_calls = n_calls
                    ckpt = checkpoint_path(hist)
                    print(
                        json.dumps(
                            {
                                "task_id": self_inner._outer.pep_task_id,
                                "checkpoint_saved": True,
                                "n_oracle_calls": n_calls,
                                "e2e_updates": int(state.tot_num_e2e_updates),
                                "latest": str(ckpt),
                                "previous": str(ckpt.with_name("checkpoint.prev.pt")),
                            }
                        ),
                        flush=True,
                    )

                def _restore_checkpoint(self_inner, ckpt: Dict[str, Any]) -> None:
                    state = self_inner.lolbo_state
                    obj = self_inner.objective
                    state.train_x = list(ckpt["train_x"])
                    state.train_y = ckpt["train_y"].float()
                    state.train_z = ckpt["train_z"].float()
                    if ckpt.get("train_c") is not None:
                        state.train_c = ckpt["train_c"].float()
                    else:
                        state.train_c = obj.compute_constraints(state.train_x)
                    obj.num_calls = int(ckpt.get("num_calls") or 0)
                    obj.xs_to_scores_dict = dict(ckpt.get("xs_to_scores_dict") or {})
                    state.tot_num_e2e_updates = int(ckpt.get("tot_num_e2e_updates") or 0)
                    # Restore cumulative acquisition accounting (absent in pre-fix
                    # checkpoints -> treated as 0, which reproduces the old behaviour).
                    self_inner._outer._n_acq_calls = int(ckpt.get("n_acq_calls") or 0)
                    self_inner._outer._n_e2e_calls = int(ckpt.get("n_e2e_calls") or 0)
                    self_inner._outer._n_rounds = int(ckpt.get("n_rounds") or 0)
                    self_inner._outer._spans = list(ckpt.get("spans") or [])
                    if ckpt.get("best_x_seen") is not None:
                        state.best_x_seen = ckpt["best_x_seen"]
                    if ckpt.get("best_score_seen") is not None:
                        state.best_score_seen = torch.tensor(float(ckpt["best_score_seen"]))
                    state.initial_model_training_complete = False
                    state.initialize_top_k()
                    state.initialize_surrogate_model()
                    state.initialize_tr_state()
                    self_inner._outer._n_init_loaded = int(
                        ckpt.get("n_init_loaded") or self_inner._outer._n_init_loaded
                    )
                    vae_ok = restore_vae_to_objective(obj, ckpt.get("vae_state_dict"))
                    e2e_n = int(ckpt.get("tot_num_e2e_updates") or 0)
                    if e2e_n > 0 and not vae_ok:
                        print(
                            json.dumps(
                                {
                                    "task_id": self_inner._outer.pep_task_id,
                                    "vae_resume_warning": True,
                                    "tot_num_e2e_updates": e2e_n,
                                    "message": "checkpoint missing finetuned VAE; using base VAE until next E2E",
                                }
                            ),
                            flush=True,
                        )
                    elif vae_ok:
                        print(
                            json.dumps(
                                {
                                    "task_id": self_inner._outer.pep_task_id,
                                    "vae_resume": True,
                                    "tot_num_e2e_updates": e2e_n,
                                }
                            ),
                            flush=True,
                        )

                def _install_shutdown_handler(self_inner) -> None:
                    if getattr(self_inner, "_shutdown_handler_installed", False):
                        return

                    def _on_shutdown(signum: int, _frame: Any) -> None:
                        print(
                            json.dumps(
                                {
                                    "task_id": self_inner._outer.pep_task_id,
                                    "shutdown_signal": int(signum),
                                    "n_oracle_calls": int(self_inner.objective.num_calls),
                                }
                            ),
                            flush=True,
                        )
                        if self_inner._outer.run_mode == "continue_bolt":
                            self_inner._save_runtime_checkpoint(force=True)
                        raise SystemExit(0)

                    signal.signal(signal.SIGTERM, _on_shutdown)
                    signal.signal(signal.SIGINT, _on_shutdown)
                    self_inner._shutdown_handler_installed = True

                def run_lolbo_bolt_native(self_inner):
                    """BOLT loop: 100 rounds x bsz, E2E every e2e_freq rounds, encode-only remap."""
                    self_inner.create_wandb_tracker()
                    self_inner._last_ckpt_calls = int(self_inner.lolbo_state.objective.num_calls)
                    self_inner._install_shutdown_handler()
                    t0 = time.time()
                    # History budget is init + acquisition only.
                    #   - acquisition(): bsz candidates per round (the "configs recommended")
                    #   - E2E: finetune VAE every e2e_freq rounds, then encode x->z in place.
                    # Native recenter() decoded x' and oracled new strings, which is what
                    # turned 5100 rows into 24k+. We never call it.
                    budget = int(self_inner.max_n_oracle_calls)
                    acq_budget_mode = self_inner._outer.budget_mode == "acquisition"
                    acq_calls = int(self_inner._outer._n_acq_calls)
                    e2e_calls = int(self_inner._outer._n_e2e_calls)
                    n_rounds = int(self_inner._outer._n_rounds)
                    consecutive_stalls = 0
                    tr_restarts = 0
                    # Raised from 40: at failure_tolerance=32 a limit of 40 let a stalling
                    # run halve the trust region exactly once before being killed, well
                    # before TuRBO's own escape (repeated halving -> length_min -> restart)
                    # could act. With stall-driven restarts below, this is now the number of
                    # consecutive dry rounds tolerated *after* the restarts have been tried.
                    max_consecutive_stalls = int(
                        os.environ.get("BOLT_MAX_CONSECUTIVE_STALLS", "120")
                    )
                    # Force a fresh trust region every N consecutive dry rounds.
                    stall_restart_every = int(
                        os.environ.get("BOLT_STALL_RESTART_EVERY", "10")
                    )
                    # Do not declare candidate_exhaustion until this many stall-driven TR
                    # restarts have each failed to surface a new sequence.
                    min_tr_restarts_before_giving_up = int(
                        os.environ.get("BOLT_MIN_TR_RESTARTS", "3")
                    )
                    target_rounds = self_inner._target_acq_rounds()
                    max_rounds = target_rounds if acq_budget_mode else int(
                        os.environ.get("BOLT_MAX_ROUNDS_FACTOR", "12")
                    ) * target_rounds
                    while (
                        n_rounds < target_rounds
                        if acq_budget_mode
                        else int(self_inner.lolbo_state.objective.num_calls) < budget
                    ):
                        self_inner.log_data_to_wandb_on_each_loop()
                        _st = self_inner.lolbo_state
                        _inc_before = (
                            float(_st.best_score_seen)
                            if _st.best_score_seen is not None
                            else None
                        )
                        self_inner.lolbo_state.update_surrogate_model()
                        before_calls = int(self_inner.lolbo_state.objective.num_calls)
                        _n0_acq = len(self_inner.lolbo_state.train_x)
                        old_bsz = int(self_inner.lolbo_state.bsz)
                        if not acq_budget_mode:
                            rem = self_inner._remaining_calls()
                            self_inner.lolbo_state.bsz = max(1, min(old_bsz, rem))
                        try:
                            self_inner.lolbo_state.acquisition()
                        finally:
                            self_inner.lolbo_state.bsz = old_bsz
                        after_calls = int(self_inner.lolbo_state.objective.num_calls)
                        acq_calls += after_calls - before_calls
                        _n1_acq = len(self_inner.lolbo_state.train_x)
                        if _n1_acq > _n0_acq:
                            # THE acquisition batch for this round: the configs the
                            # optimizer actually recommended together.
                            self_inner._outer._spans.append({
                                "start": _n0_acq, "stop": _n1_acq, "round": n_rounds,
                                "phase": "acquisition",
                                "batch_id": f"{self_inner._outer.pep_task_id}-r{n_rounds}-acq",
                                "t": round(time.time() - t0, 3),
                                "incumbent_before": _inc_before,
                            })
                        n_rounds += 1
                        pre_e2e = int(self_inner.lolbo_state.objective.num_calls)
                        self_inner._scheduled_e2e_reencode(n_rounds)
                        e2e_calls += int(self_inner.lolbo_state.objective.num_calls) - pre_e2e
                        self_inner._outer._n_acq_calls = acq_calls
                        self_inner._outer._n_e2e_calls = e2e_calls
                        self_inner._outer._n_rounds = n_rounds
                        if after_calls == before_calls:
                            # Every decoded candidate was already in xs_to_scores_dict, so
                            # this round consumed no oracle budget.
                            #
                            # ROOT CAUSE of the 290/1000 early stops (45% of their budget
                            # unspent): upstream still routes this round through
                            # update_state(), where re-scoring already-seen points cannot
                            # beat best_value, so failure_counter increments. At
                            # failure_tolerance=32 the trust region HALVES -- which draws
                            # the next candidates from a smaller latent box, making duplicate
                            # decodes MORE likely, which stalls again. Self-reinforcing
                            # collapse. Measured: truncated tasks had median
                            # tr_length_final 0.050 vs 0.100 for completed ones, only 0.7%
                            # ended with L>=0.4 vs 25.4%, and 17.9% sat at L=0.0125 -- one
                            # halving short of length_min (0.5**7) triggering the restart
                            # that would have freed them.
                            #
                            # A round with zero NEW candidates is not evidence the region is
                            # bad; it is evidence the region is EXHAUSTED. Those call for
                            # opposite responses. So: neutralise the spurious failure, and
                            # expand -- then force a full TR restart once a few consecutive
                            # stalls show the neighbourhood really is mined out. Only give up
                            # after several restarts have each failed to find anything new.
                            consecutive_stalls += 1
                            tr = self_inner.lolbo_state.tr_state
                            tr.failure_counter = 0
                            if consecutive_stalls % stall_restart_every == 0:
                                tr_restarts += 1
                                self_inner.lolbo_state.initialize_tr_state()
                                print(
                                    json.dumps(
                                        {
                                            "task_id": self_inner._outer.pep_task_id,
                                            "tr_restart_on_stall": tr_restarts,
                                            "consecutive_stalls": consecutive_stalls,
                                            "n_acquisition_calls": acq_calls,
                                        }
                                    ),
                                    flush=True,
                                )
                            else:
                                tr.length = min(2.0 * tr.length, tr.length_max)
                            if (
                                not acq_budget_mode
                                and consecutive_stalls >= max_consecutive_stalls
                                and tr_restarts >= min_tr_restarts_before_giving_up
                            ):
                                self_inner._outer._truncated = "candidate_exhaustion"
                                print(
                                    json.dumps(
                                        {
                                            "task_id": self_inner._outer.pep_task_id,
                                            "candidate_exhaustion": True,
                                            "consecutive_stalls": consecutive_stalls,
                                            "n_acquisition_calls": acq_calls,
                                            "budget": budget,
                                            "n_rounds": n_rounds,
                                            "e2e_updates": int(
                                                self_inner.lolbo_state.tot_num_e2e_updates
                                            ),
                                        }
                                    ),
                                    flush=True,
                                )
                                break
                        else:
                            # Productive round: the region is not exhausted after all, so
                            # give a later stall episode its full restart budget again.
                            consecutive_stalls = 0
                            tr_restarts = 0
                        if (not acq_budget_mode) and n_rounds >= max_rounds:
                            self_inner._outer._truncated = "max_rounds"
                            print(
                                json.dumps(
                                    {
                                        "task_id": self_inner._outer.pep_task_id,
                                        "max_rounds_reached": max_rounds,
                                        "n_acquisition_calls": acq_calls,
                                        "budget": budget,
                                    }
                                ),
                                flush=True,
                            )
                            break
                        if self_inner.lolbo_state.tr_state.restart_triggered:
                            self_inner.lolbo_state.initialize_tr_state()
                        if self_inner.lolbo_state.new_best_found:
                            if self_inner.verbose:
                                print("\nNew best found:", flush=True)
                                self_inner.print_progress_update()
                            self_inner.lolbo_state.new_best_found = False
                        if self_inner._outer.run_mode == "continue_bolt":
                            self_inner._save_runtime_checkpoint()
                    self_inner._outer._n_oracle_calls = int(
                        self_inner.lolbo_state.objective.num_calls
                    )
                    if self_inner._outer.run_mode == "continue_bolt":
                        self_inner._save_runtime_checkpoint(force=True)
                    if self_inner.verbose:
                        print("\nOptimization Run Finished, Final Results:")
                        self_inner.print_progress_update()
                        print(
                            f"Post-init oracle calls: {self_inner._outer._n_oracle_calls} "
                            f"(budget {self_inner.max_n_oracle_calls})"
                        )
                    self_inner._outer._seconds = time.time() - t0
                    self_inner._outer._lolbo_state = self_inner.lolbo_state
                    self_inner._outer._objective = self_inner.objective
                    if self_inner.tracker is not None:
                        self_inner.tracker.finish()
                    return self_inner

                def run_lolbo(self_inner):
                    if self_inner._outer.run_mode == "continue_bolt":
                        hist = self_inner._hist_path()
                        ckpt_file = checkpoint_path(hist)
                        if ckpt_file.is_file() or hist.is_file():
                            if not ckpt_file.is_file():
                                ckpt_file = ensure_checkpoint(hist)
                            ckpt = load_checkpoint(ckpt_file)
                            n_calls = int(ckpt.get("num_calls") or 0)
                            print(
                                json.dumps(
                                    {
                                        "task_id": self_inner._outer.pep_task_id,
                                        "resume_from_checkpoint": str(ckpt_file),
                                        "n_oracle_calls": n_calls,
                                        "max_oracle_budget": int(self_inner.max_n_oracle_calls),
                                    }
                                ),
                                flush=True,
                            )
                            self_inner._restore_checkpoint(ckpt)
                        else:
                            print(
                                json.dumps(
                                    {
                                        "task_id": self_inner._outer.pep_task_id,
                                        "cold_start": True,
                                        "n_init_points": int(self_inner._outer.num_init_points),
                                        "max_oracle_budget": int(self_inner.max_n_oracle_calls),
                                    }
                                ),
                                flush=True,
                            )
                        return self_inner.run_lolbo_bolt_native()
                    return self_inner.run_lolbo_checkpoint()

                def run_lolbo_checkpoint(self_inner):
                    return self_inner.run_lolbo_bolt_native()

            oracle_budget = (
                int(self.max_oracle_budget)
                if self.run_mode == "continue_bolt"
                else max(1, int(self.n_trials))
            )
            self._runner = _BoltNativeRunner(
                self,
                task_id="apex",
                seed=seed,
                track_with_wandb=False,
                minimize=bool(hps["minimize"]),
                max_n_oracle_calls=oracle_budget,
                learning_rte=float(hps["learning_rte"]),
                acq_func=str(hps["acq_func"]),
                bsz=int(hps["bsz"]),
                num_initialization_points=int(hps["num_initialization_points"]),
                init_n_update_epochs=int(hps["init_n_update_epochs"]),
                num_update_epochs=int(hps["num_update_epochs"]),
                e2e_freq=int(hps["e2e_freq"]),
                update_e2e=bool(self._use_e2e),
                k=int(hps["k"]),
                verbose=verbose,
                surrogate_type=str(hps["surrogate_type"]),
                log_table_freq=10**9,
                max_string_length=int(hps["max_string_length"]),
                task_specific_args=list(hps["task_specific_args"]),
                constraint_function_ids=list(hps["constraint_function_ids"]),
                constraint_thresholds=list(hps["constraint_thresholds"]),
                constraint_types=list(hps["constraint_types"]),
                divf_id=str(hps["divf_id"]),
                path_to_vae_statedict=str(hps["path_to_vae_statedict"]),
                init_data_path=str(hps["init_data_path"]),
                init_scores_path=str(hps["init_scores_path"]),
                init_offset_helper=int(self._init_offset_helper),
            )
            self._opt_name = "bolt_native"
            self._oracle_tag = "bolt_bacteria_0"
            if self.pep_idx is not None:
                _, scores = load_seed0_pool(limit=self.num_init_points)
                if 0 <= int(self.pep_idx) < len(scores):
                    self._init_score = float(scores[int(self.pep_idx)])
        else:
            raise ValueError(f"Unsupported oracle_mode={oracle_mode!r}; use bolt_native")

    def _reference(self) -> str:
        return bolt_reference_sequence(constraint_index=self.constraint_index)

    def similarity_to_reference(self, seq: Optional[str]) -> Optional[float]:
        """BOLT similarity 1 - lev(seq, ref)/len(ref). None for a missing sequence."""
        if not isinstance(seq, str) or not seq:
            return None
        return float(template_similarity(seq, self._reference()))

    def _assert_feasible_init(self, init_xs: List[str]) -> None:
        """Fail loudly if no init point satisfies the similarity constraint.

        The whole bo80_bolt_bsz10_r100 sweep (999/1000 tasks) ran to completion with an
        empty feasible set: top_k stayed empty, best_x_seen stayed None, E2E finetuned on
        10 sequences, and export() still reported a plausible-looking unconstrained best.
        This check turns that silent degeneracy into an immediate, named failure.
        """
        ref = self._reference()
        sims = [template_similarity(str(x), ref) for x in init_xs]
        n_feasible = sum(1 for s in sims if s >= self.min_similarity)
        self._n_init_feasible = n_feasible
        self._init_max_similarity = float(max(sims)) if sims else 0.0
        print(
            json.dumps(
                {
                    "task_id": self.pep_task_id,
                    "init_feasible": n_feasible,
                    "init_total": len(sims),
                    "init_max_similarity": round(self._init_max_similarity, 4),
                    "reference": ref,
                    "min_similarity": self.min_similarity,
                }
            ),
            flush=True,
        )
        if n_feasible == 0 and self.require_feasible_init:
            raise RuntimeError(
                f"{self.pep_task_id}: 0/{len(sims)} init points satisfy similarity "
                f">= {self.min_similarity} vs reference {ref!r} (constraint_index="
                f"{self.constraint_index}, max similarity in pool "
                f"{self._init_max_similarity:.3f}). The init pool almost certainly does "
                f"not correspond to this reference -- check init_data_path and "
                f"init_offset_helper (block = constraint_types[0] - init_offset_helper)."
            )

    def run(self) -> Dict[str, Any]:
        self._runner.run_lolbo()
        return self.export()

    def export(self) -> Dict[str, Any]:
        state = self._lolbo_state
        obj = self._objective
        if state is None or obj is None:
            raise RuntimeError("LOLBO did not run")

        minimize = bool(state.minimize)
        train_x: List[str] = list(state.train_x)
        train_y = state.train_y.detach().cpu().float().reshape(-1)
        train_z = state.train_z.detach().cpu().float()
        if train_z.dim() == 3:
            train_z = train_z.reshape(train_z.shape[0], -1)
        dim = int(train_z.shape[-1])
        n_init = int(self._n_init_loaded or self.num_init_points)
        n_oracle = int(self._n_oracle_calls or getattr(obj, "num_calls", 0))

        slim = bool(self.run_mode == "continue_bolt")
        observations: List[Dict[str, Any]] = []
        t0 = time.time()
        # index -> provenance, from the spans recorded during the loop. Rows before n_init
        # are the init pool; anything after with no span is defensive (should not happen).
        prov: List[Optional[Dict[str, Any]]] = [None] * len(train_x)
        for sp in self._spans:
            for k in range(int(sp["start"]), min(int(sp["stop"]), len(train_x))):
                prov[k] = sp
        # Per-row constraint value, which upstream never wrote into the history.
        train_c = None
        if state.train_c is not None:
            tc = state.train_c.detach().cpu().float()
            train_c = tc.reshape(tc.shape[0], -1) if tc.dim() > 1 else tc.reshape(-1, 1)
        ref_seq = bolt_reference_sequence(constraint_index=self.constraint_index)

        for i, (seq, y_val, z_row) in enumerate(zip(train_x, train_y, train_z)):
            score = float(y_val.item())
            bacteria0_score = score if not minimize else -score
            mic0 = bolt_maximize_score_to_mic(bacteria0_score)
            sp = prov[i]
            is_init = i < n_init
            source = "bolt_seed0_init" if is_init else self._opt_name
            cfg = {} if slim else _z_config(z_row, dim)
            cvals = (
                [round(float(v), 6) for v in train_c[i].tolist()]
                if train_c is not None and i < train_c.shape[0]
                else None
            )
            sim = template_similarity(seq, ref_seq)
            phase = "init" if is_init else (sp["phase"] if sp else "acquisition")
            if phase not in {"init", "acquisition"}:
                continue
            observations.append(
                observation(
                    cfg,
                    bacteria0_score if not minimize else mic0,
                    # Real wall-clock offset for the round that produced this row. The old
                    # value was `time.time() - t0` measured inside THIS serialization loop,
                    # i.e. microseconds of JSON building, which made every time-to-solution
                    # analysis meaningless.
                    elapsed_time=0.0 if is_init else float(sp["t"]) if sp else -1.0,
                    extra={
                        "source": source,
                        "artifact": seq,
                        # provenance: lets you rebuild the exact proposed batch
                        "round": -1 if is_init else (int(sp["round"]) if sp else None),
                        "phase": "init" if is_init else (sp["phase"] if sp else None),
                        "batch_id": None if is_init else (sp["batch_id"] if sp else None),
                        "incumbent_before": None if is_init or not sp else sp["incumbent_before"],
                        "info": {
                            "sequence": seq,
                            "oracle": self._oracle_tag,
                            "raw_cost": bacteria0_score if not minimize else mic0,
                            "bacteria0_score": bacteria0_score,
                            "mic_bacteria_0": mic0,
                            "precomputed": bool(is_init),
                            "constraint_values": cvals,
                            "similarity_to_reference": round(float(sim), 6),
                            "feasible": bool(sim >= self.min_similarity - 1e-9),
                        },
                    },
                )
            )

        best_obj = max(float(o["objectives"][0]) for o in observations)

        # The unconstrained best above is what the old code reported, and it is what let a
        # fully infeasible run look healthy. Also compute the best point that actually
        # satisfies the similarity constraint -- this is the number the paper reports.
        ref = self._reference()
        best_feasible_obj: Optional[float] = None
        best_feasible_seq: Optional[str] = None
        n_feasible_obs = 0
        for o in observations:
            seq = str((o.get("extra_info") or {}).get("artifact") or "")
            if template_similarity(seq, ref) < self.min_similarity:
                continue
            n_feasible_obs += 1
            v = float(o["objectives"][0])
            if best_feasible_obj is None or v > best_feasible_obj:
                best_feasible_obj, best_feasible_seq = v, seq

        hps_used = dict(getattr(self, "_hps_used", None) or BOLT_SEED0_HPARAMS)
        out_root = protein_bo80_bolt_root() / self.pep_task_id / self._opt_name / "seed1"
        hist_path = out_root / "history.json"
        meta = {
            "optimizer": self._opt_name,
            "vae": "bolt_uniref_256",
            "oracle": self._oracle_tag,
            "n_init_points": n_init,
            "n_post_init_trials": self.n_trials,
            "n_oracle_calls": n_oracle,
            "n_trials": total_trial_budget(
                n_init_points=n_init, n_post_init_trials=self.n_trials
            ),
            "trust_region": True,
            "e2e_vae": bool(self._use_e2e),
            "e2e_schedule": "every_n_rounds",
            "e2e_reencode_only": True,
            "strict_oracle_budget": self.strict_oracle_budget,
            "run_mode": self.run_mode,
            "max_oracle_budget": self.max_oracle_budget,
            "checkpoint_calls": self.checkpoint_calls,
            "min_similarity": self.min_similarity,
            "constraint_index": self.constraint_index,
            "init_offset_helper": self._init_offset_helper,
            "bolt_reference": ref,
            "objective": "bacteria_0",
            # The hparams actually passed to BOLT, not the module defaults.
            "bolt_hparams": hps_used,
            "init_data_path": str(hps_used.get("init_data_path", "")),
            "init_scores_path": str(hps_used.get("init_scores_path", "")),
            "n_init_feasible": self._n_init_feasible,
            "init_max_similarity": self._init_max_similarity,
            "n_feasible_obs": n_feasible_obs,
            "best_feasible_objective": best_feasible_obj,
            "truncated": self._truncated,
            "slim_history": slim,
        }
        if slim:
            # Compact JSON without 256-d latents: ~0.3–1GB vs ~14GB per finished task.
            hist_path.parent.mkdir(parents=True, exist_ok=True)
            hist_path.write_text(
                json.dumps(
                    {
                        "task_id": self.pep_task_id,
                        "num_objectives": 1,
                        "num_constraints": 0,
                        "ref_point": None,
                        "meta_info": meta,
                        "observations": observations,
                    },
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            write_openbox_history(
                hist_path,
                task_id=self.pep_task_id,
                observations=observations,
                meta=meta,
            )
        # Do not duplicate history.json into fitness/ (~14GB per finished task).
        bench_hist = hist_path

        if self.run_mode in {"checkpoint", "continue_bolt"}:
            save_checkpoint(
                checkpoint_path(hist_path),
                train_x=train_x,
                train_y=state.train_y.detach().cpu(),
                train_z=state.train_z.detach().cpu(),
                train_c=state.train_c.detach().cpu() if state.train_c is not None else None,
                num_calls=n_oracle,
                xs_to_scores_dict=dict(obj.xs_to_scores_dict),
                n_init_loaded=n_init,
                tot_num_e2e_updates=int(state.tot_num_e2e_updates),
                best_x_seen=state.best_x_seen,
                best_score_seen=state.best_score_seen,
                minimize=minimize,
                vae_state_dict=vae_state_dict_from_objective(obj),
                n_acq_calls=int(self._n_acq_calls),
                n_e2e_calls=int(self._n_e2e_calls),
                n_rounds=int(self._n_rounds),
                spans=list(self._spans),
            )

        row = {
            "domain": "protein",
            "task_id": self.pep_task_id,
            "start_seq": self.start_seq,
            "n_init_points": n_init,
            "n_post_init_trials": self.n_trials,
            "n_oracle_calls": n_oracle,
            "n_trials": total_trial_budget(
                n_init_points=n_init, n_post_init_trials=self.n_trials
            ),
            "n_obs": len(observations),
            "best_objective": best_obj,
            "seconds": round(self._seconds, 2),
            "history": str(bench_hist),
            "vae": "bolt_uniref_256",
            "oracle": self._oracle_tag,
            "optimizer": self._opt_name,
            "tr_length_final": float(state.tr_state.length),
            "e2e_updates": int(state.tot_num_e2e_updates),
            # Report what actually ran. These previously read BOLT_SEED0_HPARAMS, so all
            # 531 rows of the bsz=10 sweep claimed bsz=50.
            "max_string_length": int(hps_used.get("max_string_length", 30)),
            "bsz": int(hps_used.get("bsz", 0)),
            "e2e_freq": int(hps_used.get("e2e_freq", 0)),
            # Constraint accounting -- the fields whose absence hid the init-pool bug.
            "constraint_index": self.constraint_index,
            "reference": ref,
            "min_similarity": self.min_similarity,
            "n_init_feasible": self._n_init_feasible,
            "init_max_similarity": self._init_max_similarity,
            "n_feasible_obs": n_feasible_obs,
            "best_feasible_objective": best_feasible_obj,
            "best_feasible_seq": best_feasible_seq,
            "best_feasible_similarity": self.similarity_to_reference(best_feasible_seq),
            "best_x_seen": state.best_x_seen,
            "best_x_seen_similarity": self.similarity_to_reference(state.best_x_seen),
            "truncated": self._truncated,
            # Full budget accounting: acquisition calls are the "configs recommended"
            # (rounds x bsz). E2E remaps x->z in place and must not add oracle rows.
            "budget_mode": self.budget_mode,
            "n_rounds": self._n_rounds,
            "n_acquisition_calls": self._n_acq_calls,
            "n_e2e_calls": self._n_e2e_calls,
            "budget_respected": bool(
                (self._n_acq_calls if self.budget_mode == "acquisition" else n_oracle)
                <= int(self.max_oracle_budget)
            ),
        }
        index = protein_bo80_bolt_root() / "index.jsonl"
        _append_index(index, row)
        return row


def _history_oracle_calls(hist: Path) -> Optional[int]:
    if not hist.is_file():
        return None
    try:
        with open(hist, "rb") as f:
            head = f.read(16384).decode("utf-8", "ignore")
        m = re.search(r'"n_oracle_calls"\s*:\s*(\d+)', head)
        if m:
            return int(m.group(1))
        payload = json.loads(hist.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    meta = payload.get("meta_info") or payload.get("meta") or {}
    if "n_oracle_calls" in meta:
        return int(meta["n_oracle_calls"])
    return None


def _checkpoint_oracle_calls(hist: Path) -> Optional[int]:
    ckpt = checkpoint_path(hist)
    meta = ckpt.with_name("checkpoint.meta.json")
    if meta.is_file():
        try:
            return int(json.loads(meta.read_text(encoding="utf-8")).get("num_calls"))
        except (json.JSONDecodeError, TypeError, OSError):
            pass
    if not ckpt.is_file():
        return None
    try:
        payload = load_checkpoint(ckpt)
    except Exception:
        return None
    if "num_calls" in payload:
        return int(payload["num_calls"])
    return None


def _history_acquisition_calls(hist: Path) -> Optional[int]:
    """Acquisition-call count from history.json, or None if not recorded."""
    if not hist.is_file():
        return None
    try:
        with open(hist, "rb") as f:
            head = f.read(32768).decode("utf-8", "ignore")
    except OSError:
        return None
    m = re.search(r'"n_acquisition_calls"\s*:\s*(\d+)', head)
    return int(m.group(1)) if m else None


def _task_done(hist: Path, *, n_trials: int, run_mode: str = "checkpoint") -> bool:
    if str(run_mode).lower() == "continue_bolt":
        # A finished task always has history.json, because export() writes it after the
        # loop returns. A checkpoint alone only proves the task *started*.
        #
        # Do NOT infer completion from the checkpoint's num_calls: that counts EVERY
        # oracle call, including E2E re-scoring, while the budget under
        # --budget-mode acquisition counts only acquisition candidates. Total calls
        # routinely reach 2-4x the acquisition budget, so an interrupted task shows
        # num_calls >> n_trials and would be skipped forever, leaving no index row.
        # That silently dropped 35 tasks (pep_001..pep_079) from a 1000-task fleet.
        if not hist.is_file():
            return False
        # Prefer the acquisition count, which is what the budget actually caps.
        acq = _history_acquisition_calls(hist)
        if acq is not None:
            return int(acq) >= int(n_trials)
        return True  # completed under an older schema that did not record it
    # checkpoint mode: n_calls was never assigned on this path (UnboundLocalError).
    n_calls = _history_oracle_calls(hist)
    if n_calls is None:
        n_calls = _checkpoint_oracle_calls(hist)
    if n_calls is None:
        return False
    return int(n_calls) >= int(n_trials)


def run_tasks(
    *,
    n_trials: int = 500,
    seed: Optional[int] = None,
    device_id: int = 0,
    task_ids: Optional[List[str]] = None,
    skip_done: bool = True,
    oracle_mode: str = "bolt_native",
    min_similarity: float = 0.75,
    constraint_index: int = 0,
    n_init_points: int = 1000,
    bsz: int = 50,
    e2e_freq: Optional[int] = None,
    n_tasks: int = 1000,
    task_offset: int = 0,
    shard_index: int = 0,
    n_shards: int = 1,
    run_mode: str = "checkpoint",
    max_oracle_budget: int = BOLT_FULL_ORACLE_BUDGET,
    checkpoint_calls: int = CHECKPOINT_ORACLE_CALLS,
    init_data_path: Optional[str] = None,
    init_scores_path: Optional[str] = None,
    init_offset_helper: int = 0,
    require_feasible_init: bool = True,
    budget_mode: str = "acquisition",
) -> List[Dict[str, Any]]:
    opt_name = "bolt_native" if str(oracle_mode).lower() == "bolt_native" else "bolt_lolbo"
    if task_ids:
        tasks = list(task_ids)
    else:
        tasks = _task_ids(n_tasks, task_offset, shard_index, n_shards)
    all_seqs, _ = load_seed0_pool()
    results: List[Dict[str, Any]] = []
    min_free_gb = float(os.environ.get("BOLT_MIN_FREE_GB", "40"))
    data_root = protein_bo80_bolt_root()
    # shutil.disk_usage() below stats this path, so it has to exist first.
    data_root.mkdir(parents=True, exist_ok=True)
    for tid in tasks:
        idx = _parse_pep_idx(tid)
        if idx < 0 or idx >= len(all_seqs):
            print(json.dumps({"task_id": tid, "skip": "idx_out_of_range"}))
            continue
        start_seq = all_seqs[idx]
        out_dir = protein_bo80_bolt_root() / tid / opt_name / "seed1"
        hist = out_dir / "history.json"
        free_gb = shutil.disk_usage(data_root).free / (1024**3)
        if free_gb < min_free_gb:
            print(
                json.dumps(
                    {
                        "task_id": tid,
                        "skip": "disk_low",
                        "free_gb": round(free_gb, 1),
                        "min_free_gb": min_free_gb,
                    }
                ),
                flush=True,
            )
            break
        if skip_done and _task_done(
            hist,
            n_trials=max_oracle_budget if run_mode == "continue_bolt" else n_trials,
            run_mode=run_mode,
        ):
            n_calls = _history_oracle_calls(hist)
            print(json.dumps({"task_id": tid, "skip": "done", "n_oracle_calls": n_calls, "run_mode": run_mode}))
            results.append({"task_id": tid, "skipped": True, "n_oracle_calls": n_calls})
            continue
        print(
            json.dumps(
                {
                    "task_id": tid,
                    "start_seq": start_seq,
                    "n_init_points": n_init_points,
                    "n_post_init_trials": n_trials,
                    "run_mode": run_mode,
                    "max_oracle_budget": max_oracle_budget if run_mode == "continue_bolt" else n_trials,
                    "oracle_mode": oracle_mode,
                }
            ),
            flush=True,
        )
        runner = BenchmarkPeptideLOLBO(
            pep_task_id=tid,
            start_seq=start_seq,
            n_trials=n_trials,
            seed=seed,
            device_id=device_id,
            oracle_mode=oracle_mode,
            min_similarity=min_similarity,
            constraint_index=idx,
            pep_idx=idx,
            num_initialization_points=n_init_points,
            bsz=bsz,
            e2e_freq=int(e2e_freq if e2e_freq is not None else BOLT_SEED0_HPARAMS.get("e2e_freq") or 10),
            run_mode=run_mode,
            max_oracle_budget=max_oracle_budget,
            checkpoint_calls=checkpoint_calls,
            init_data_path=init_data_path,
            init_scores_path=init_scores_path,
            init_offset_helper=init_offset_helper,
            require_feasible_init=require_feasible_init,
            budget_mode=budget_mode,
        )
        row = runner.run()
        print(json.dumps(row), flush=True)
        results.append(row)
    return results


def run_consistent10(
    *,
    n_trials: int = 500,
    seed: Optional[int] = None,
    device_id: int = 0,
    task_ids: Optional[List[str]] = None,
    skip_done: bool = True,
    oracle_mode: str = "bolt_native",
    n_init_points: int = 1000,
    bsz: int = 50,
    n_tasks: int = 1000,
    n_shards: int = 1,
    run_mode: str = "checkpoint",
    max_oracle_budget: int = BOLT_FULL_ORACLE_BUDGET,
    checkpoint_calls: int = CHECKPOINT_ORACLE_CALLS,
) -> List[Dict[str, Any]]:
    return run_tasks(
        n_trials=n_trials,
        seed=seed,
        device_id=device_id,
        task_ids=list(task_ids or CONSISTENT10),
        skip_done=skip_done,
        oracle_mode=oracle_mode,
        n_init_points=n_init_points,
        bsz=bsz,
        n_tasks=len(CONSISTENT10),
        n_shards=1,
        run_mode=run_mode,
        max_oracle_budget=max_oracle_budget,
        checkpoint_calls=checkpoint_calls,
    )
