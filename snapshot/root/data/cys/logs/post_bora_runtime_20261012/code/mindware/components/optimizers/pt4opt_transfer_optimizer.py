"""ML transfer adapters using isolated workers and verified source artifacts."""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from ConfigSpace import Configuration
from openbox.utils.constants import SUCCESS, TIMEOUT

from mindware.components.optimizers.base_optimizer import BaseOptimizer, MAX_INT
from mindware.components.optimizers.llm_optimizer_utils import wrapper_evaluater
from mindware.components.optimizers.utils import MyConfiguration
from mindware.components.optimizers import ml_checkpoint
from mindware.components.optimizers.transfer_hpo.schema import encode_config, group_id, schema_fingerprint, configspace_spec
from mindware.components.optimizers.transfer_hpo.protocol import TRAIN_MANIFEST, sha256_file, sha256_directory, validate_dump
from mindware.knowledge_db.knowledge_database import KnowledgeDatabase

REPO = Path(__file__).resolve().parents[3]
DEFAULT_DUMP_DIR = REPO / "benchmark_data/ml_train_80_transfer_v2"
DEFAULT_FSBO_CKPT_DIR = REPO / "benchmark_data/ml_train_80_fsbo_ckpts"
DEFAULT_RGPE_CACHE_DIR = REPO / "benchmark_data/ml_train_80_rgpe_sources"
FSBO_PYTHON = "/root/data/cys/baseline/envs/bl_fsbo_exact/bin/python"
RGPE_PYTHON = "/root/data/cys/baseline/envs/bl_rgpe/bin/python"
FSBO_REPO = "/root/data/cys/baseline/repos/FSBO"
RGPE_REPO = "/root/data/cys/baseline/repos/transfer-hpo-framework"
OPTFORMER_PYTHON = "/root/data/cys/baseline/envs/bl_optformer/bin/python"
OPTFORMER_REPO = "/root/data/cys/baseline/repos/optformer-neurips22"
T5X_REPO = "/root/data/cys/baseline/repos/t5x-2023-07"


class _TransferHPOOptimizer(BaseOptimizer):
    optimizer_label = "transfer"
    minimum_observations = 1

    def __init__(self, evaluator, config_space, name, eval_type, time_limit=None,
                 evaluation_limit=None, per_run_time_limit=300, per_run_mem_limit=1024,
                 inner_iter_num_per_iter=1, timestamp=None, output_dir="./", seed=1,
                 n_jobs=1, topk=50, optimizer_params=None, task_id="GenTrans",
                 logging_enabled=True, **sampler_params):
        timestamp = time.time() if timestamp is None else timestamp
        super().__init__(evaluator=evaluator, config_space=config_space, name=name,
                         eval_type=eval_type, task_id=task_id, time_limit=time_limit,
                         evaluation_limit=evaluation_limit, per_run_time_limit=per_run_time_limit,
                         per_run_mem_limit=per_run_mem_limit, inner_iter_num_per_iter=inner_iter_num_per_iter,
                         timestamp=timestamp, output_dir=output_dir, seed=seed, topk=topk)
        self.optimizer_params = dict(optimizer_params or {})
        self.optimizer_params.update(sampler_params)
        self.rng = np.random.RandomState(int(seed))
        self.n_jobs = 1
        self.exp_output = {}
        self.KD = KnowledgeDatabase.instance()
        self.configs, self.perfs = self.KD.configs, self.KD.perfs
        self.trial_cnt = len(self.configs)
        self.n_startup = int(self.optimizer_params.get("n_startup_trials", 3))
        self.n_pending = int(self.optimizer_params.get("n_pending", 256))
        if self.n_startup < self.minimum_observations or self.n_pending < 1:
            raise ValueError("Insufficient startup observations or empty candidate pool")
        algorithm = config_space.get_hyperparameter("algorithm")
        choices = list(getattr(algorithm, "choices", []))
        self.fixed_algorithm = getattr(algorithm, "value", None)
        if self.fixed_algorithm is None and len(choices) == 1:
            self.fixed_algorithm = choices[0]
        if self.fixed_algorithm is None:
            raise ValueError("Transfer optimization requires a fixed ML algorithm")
        self.task_type = self.optimizer_params.get("task_type")
        self.group = group_id(self.task_type, self.fixed_algorithm)
        self.schema_fingerprint = schema_fingerprint(self.config_space)
        self.dump_path = Path(self.optimizer_params.get("dump_dir") or DEFAULT_DUMP_DIR) / (self.group.replace(":", "_") + ".json")
        if self.optimizer_label in ("fsbo", "rgpe"):
            matches = [path for path in self.dump_path.parent.glob(self.group.replace(":", "_") + "*.json")
                       if json.loads(path.read_text()).get("schema_fingerprint") == self.schema_fingerprint]
            if len(matches) != 1:
                raise ValueError("Expected one source export matching target schema for %s, found %d" % (self.group, len(matches)))
            self.dump_path = matches[0]
        self.artifact_tag = self.dump_path.stem
        self.logging_enabled = logging_enabled
        self.traj_rows = []
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.traj_csv_path = str(Path(output_dir) / ("%d_opt_%s_traj.csv" % (timestamp, self.optimizer_label)))
        self.start_wall_time = time.time()
        self.outer_iter_idx = 0
        self.bo_mode = "bo_mab"
        self.asset_identity = {}
        self._proposal_diagnostics = None
        self.attempted_configs = set(self._key(config) for config in self.configs)

    @staticmethod
    def _cfg_dict(config):
        return config.get_dictionary() if hasattr(config, "get_dictionary") else dict(config)

    def _key(self, config):
        return json.dumps(self._cfg_dict(config), sort_keys=True)

    def _observed(self):
        return [(c, float(p)) for c, p in zip(self.configs, self.perfs) if np.isfinite(p)]

    def _require_dump(self):
        dump = json.loads(self.dump_path.read_text())
        validate_dump(dump)
        if dump["schema_fingerprint"] != self.schema_fingerprint:
            raise ValueError("Source and target search spaces differ for %s" % self.group)
        return dump

    def _set_assets(self, identity):
        if self.asset_identity and self.asset_identity != identity:
            raise RuntimeError("Transfer artifacts changed during this run")
        self.asset_identity = identity

    def _check_assets(self):
        raise NotImplementedError

    def _pending_configs(self, count):
        found, keys = [], set(self.attempted_configs)
        for _ in range(max(100, count * 20)):
            candidate = self.config_space.sample_configuration()
            key = self._key(candidate)
            if key not in keys:
                keys.add(key)
                found.append(candidate)
                if len(found) == count:
                    break
        if not found:
            raise RuntimeError("No unevaluated configurations found in the search space")
        return found

    def _suggest(self):
        self._proposal_diagnostics = None
        self._check_assets()
        observed = self._observed()
        if self.trial_cnt < self.n_startup:
            return self._pending_configs(1)[0], "configured random initialization"
        if len(observed) < self.minimum_observations:
            raise RuntimeError("Startup ended without enough successful observations for %s" % self.optimizer_label)
        return self._suggest_model(observed)

    def _suggest_model(self, observed):
        raise NotImplementedError

    def _budget_exhausted(self):
        exhausted = ((self.evaluation_num_limit is not None and self.trial_cnt >= self.evaluation_num_limit)
                     or (self.time_limit is not None and time.time() - self.timestamp >= self.time_limit))
        if exhausted:
            self.timeout_flag = True
        return exhausted or self.early_stopped_flag or self.timeout_flag

    def run(self):
        while not self._budget_exhausted():
            self.iterate()
        return self.incumbent_perf

    def iterate(self, budget=MAX_INT):
        start = time.time()
        rows = []
        for _ in range(self.inner_iter_num_per_iter):
            if self._budget_exhausted() or time.time() - start >= budget:
                break
            trial_start = time.time()
            self._check_assets()
            proposal = ml_checkpoint.pending(self)
            if proposal is None:
                config, note = self._suggest()
                proposal = {"config": self._cfg_dict(config), "note": note,
                            "suggest_time": time.time() - trial_start,
                            "diagnostics_dir": self._proposal_diagnostics}
            config = MyConfiguration(Configuration(self.config_space, values=proposal["config"]))
            self.config_space.check_configuration(config)
            if self._budget_exhausted():
                break
            ml_checkpoint.prepare(self, proposal)
            observation = wrapper_evaluater(config, self.evaluator, self.per_run_time_limit)
            objective = float(observation.objectives[0]) if observation.objectives else float("inf")
            success = observation.trial_state == SUCCESS and np.isfinite(objective)
            perf = -objective if success else float("-inf")
            if not success:
                observation.objectives = [float("inf")]
            self.update_history(observation, trail_id=str(self.trial_cnt))
            if success:
                self.exp_output[time.time()] = (config, perf)
                self.update_saver([config], [objective])
            self.attempted_configs.add(self._key(config))
            elapsed = float(observation.elapsed_time or 0.0)
            self._record_event(self.trial_cnt, config, perf, elapsed, proposal["suggest_time"],
                               time.time() - trial_start, proposal["note"], observation.trial_state,
                               proposal.get("diagnostics_dir"))
            self.trial_cnt += 1
            finite = self._observed()
            if finite:
                best, self.incumbent_perf = max(finite, key=lambda pair: pair[1])
                self.incumbent_config = self._cfg_dict(best)
            rows.append({"perf": perf, "config": config, "eval_time": elapsed, "note": proposal["note"]})
            ml_checkpoint.commit(self)
            self._budget_exhausted()
        self.outer_iter_idx += 1
        return self.incumbent_perf, time.time() - start, self.incumbent_config, rows

    def update_history(self, observation, **extra):
        self.KD.update_history(observation, source=self.bo_mode, **extra)

    def _record_event(self, index, config, perf, eval_time, suggest_time, iter_time, note, state,
                      diagnostics_dir=None):
        self.traj_rows.append({
            "ts": pd.Timestamp.now().isoformat(), "stage": "iterate", "iter_idx": index,
            "perf": perf, "best_so_far": max(self.perfs) if self.perfs else float("-inf"),
            "algorithm": self.fixed_algorithm, "smac_time": suggest_time, "eval_time": eval_time,
            "iter_time": iter_time, "wall_elapsed_sec": time.time() - self.start_wall_time,
            "note": note, "trial_state": int(state), "config_json": self._key(config),
            "evaluation_status": "SUCCESS" if state == SUCCESS and np.isfinite(perf)
                else "TIMEOUT" if state == TIMEOUT else "FAILED",
            "evaluation_limit_seconds": self.per_run_time_limit,
            "runtime_policy_id": os.environ.get("ML_RUNTIME_POLICY_ID"),
            "oracle_called": True,
            "schema_fingerprint": self.schema_fingerprint,
            "assets_json": json.dumps(self.asset_identity, sort_keys=True),
            "diagnostics_dir": diagnostics_dir,
        })
        if self.logging_enabled and not getattr(self, "_ml_session", None):
            temporary = self.traj_csv_path + ".tmp"
            pd.DataFrame(self.traj_rows).to_csv(temporary, index=False)
            os.replace(temporary, self.traj_csv_path)

    def _worker_env(self, pythonpath):
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([pythonpath, str(REPO), env.get("PYTHONPATH", "")])
        if not self.optimizer_params.get("allow_cuda"):
            env["CUDA_VISIBLE_DEVICES"] = ""
        env.setdefault("OMP_NUM_THREADS", "1")
        return env

    def _run_worker(self, python, worker, worker_repo, flags, request):
        session = getattr(self, "_ml_session", None)
        root = session.directory if session else Path(self.traj_csv_path).parent
        trial_dir = root / "decisions" / ("trial-%03d" % self.trial_cnt)
        trial_dir.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="attempt-", dir=trial_dir)).resolve()
        self._proposal_diagnostics = str(directory)
        req, out = directory / "request.json", directory / "response.json"
        request = dict(request, schema_fingerprint=self.schema_fingerprint)
        req.write_text(json.dumps(request, indent=2, allow_nan=False))
        command = [python, str(REPO / "examples/benchmark" / worker), "suggest"]
        command.extend(flags + ["--request", str(req), "--out", str(out)])
        record = {"status": "running", "optimizer": self.optimizer_label, "trial_id": self.trial_cnt,
                  "run_id": session.identity if session else None, "command": command,
                  "cwd": str(worker_repo), "started_at": time.time(), "assets": self.asset_identity,
                  "request_sha256": sha256_file(req)}
        metadata = directory / "worker.json"
        ml_checkpoint.atomic_json(metadata, record)
        started = time.monotonic()
        try:
            if self.optimizer_label == "optformer" and os.environ.get("OPTFORMER_REMOTE_CONFIG"):
                from .optformer_remote import suggest_remote
                payload, remote = suggest_remote(request, self.asset_identity["checkpoint_sha256"])
                record.update(remote=remote, returncode=0)
                ml_checkpoint.atomic_json(out, payload)
            else:
                with (directory / "stdout.log").open("w") as stdout, (directory / "stderr.log").open("w") as stderr:
                    result = subprocess.run(command, cwd=worker_repo, env=self._worker_env(worker_repo),
                                            stdout=stdout, stderr=stderr, text=True,
                                            timeout=float(self.optimizer_params.get("worker_timeout", 180)))
                record["returncode"] = result.returncode
                if result.returncode:
                    detail = (directory / "stderr.log").read_text()[-2000:]
                    raise RuntimeError("%s worker failed: %s (diagnostics: %s)" % (
                        self.optimizer_label, detail, directory))
            payload = json.loads(out.read_text())
            if not isinstance(payload, dict):
                raise ValueError("Worker response must be a JSON object")
            record.update(status="succeeded", response_sha256=sha256_file(out))
            return payload
        except subprocess.TimeoutExpired as exc:
            record.update(status="timeout", error=str(exc))
            raise RuntimeError("%s worker timed out before evaluation (diagnostics: %s)" % (
                self.optimizer_label, directory)) from exc
        except Exception as exc:
            record.update(status="failed", error=str(exc))
            raise
        finally:
            record.update(finished_at=time.time(), elapsed_seconds=time.monotonic() - started)
            ml_checkpoint.atomic_json(metadata, record)

    def _selected(self, payload, pending):
        index = payload.get("index")
        directory = self._proposal_diagnostics
        metadata = Path(directory) / "worker.json" if directory else None
        record = json.loads(metadata.read_text()) if metadata else {}
        if type(index) is not int or not 0 <= index < len(pending):
            if metadata:
                record.update(status="invalid_response", error="Worker returned an invalid candidate index")
                ml_checkpoint.atomic_json(metadata, record)
            raise RuntimeError("Worker returned an invalid candidate index")
        if metadata:
            record.update(selected_index=index, selected_config=self._cfg_dict(pending[index]))
            ml_checkpoint.atomic_json(metadata, record)
        return pending[index]

    def get_opt_trajectory(self):
        return {"n_evals": self.trial_cnt, "n_successful": len(self._observed()),
                "final_best": self.incumbent_perf, "csv_path": self.traj_csv_path,
                "group": self.group, "assets": self.asset_identity,
                "detail_perfs": ",".join(str(p) for p in self.perfs)}

    def restore_from_traj(self, *args, **kwargs):
        raise RuntimeError("Transfer methods require the durable ML session checkpoint for resume; CSV-only resume is unsupported")


class FSBOOptimizer(_TransferHPOOptimizer):
    optimizer_label = "fsbo"
    minimum_observations = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ckpt_dir = Path(self.optimizer_params.get("ckpt_dir") or DEFAULT_FSBO_CKPT_DIR) / self.artifact_tag
        self.fsbo_python = str(self.optimizer_params.get("fsbo_python") or FSBO_PYTHON)
        self._check_assets()

    def _check_assets(self):
        self._require_dump()
        meta = json.loads((self.ckpt_dir / "meta.json").read_text())
        weights_hash = sha256_file(self.ckpt_dir / "weights")
        if (meta.get("status") != "complete" or meta.get("method") != "fsbo_paper"
                or meta["schema_fingerprint"] != self.schema_fingerprint
                or meta["weights_sha256"] != weights_hash or meta["dump_sha256"] != sha256_file(self.dump_path)):
            raise ValueError("FSBO checkpoint is incomplete or has incompatible provenance")
        assets = {"checkpoint_sha256": weights_hash, "dump_sha256": meta["dump_sha256"]}
        # Checkpoints trained on per-task min-max labels need min-max target observations.
        self.target_scaling = meta.get("target_scaling", "raw")
        if self.target_scaling != "raw":
            assets["target_scaling"] = self.target_scaling
        self._set_assets(assets)

    def _suggest_model(self, observed):
        pending = self._pending_configs(self.n_pending)
        request = {"X_obs": [encode_config(self.config_space, self._cfg_dict(c), self.fixed_algorithm).tolist() for c, _ in observed],
                   "y_obs": [p for _, p in observed],
                   "X_pen": [encode_config(self.config_space, self._cfg_dict(c), self.fixed_algorithm).tolist() for c in pending],
                   "observed_configs": [self._cfg_dict(c) for c, _ in observed],
                   "pending_configs": [self._cfg_dict(c) for c in pending],
                   "seed": int(self.seed), "finetune_epochs": int(self.optimizer_params.get("finetune_epochs", 100)),
                   "target_scaling": self.target_scaling}
        payload = self._run_worker(self.fsbo_python, "fsbo_worker.py", FSBO_REPO,
                                   ["--ckpt", str(self.ckpt_dir)], request)
        if payload.get("target_scaling", "raw") != self.target_scaling:
            raise RuntimeError("FSBO worker ignored the requested target scaling")
        return self._selected(payload, pending), "FSBO paper deep-kernel EI"


class RGPEOptimizer(_TransferHPOOptimizer):
    optimizer_label = "rgpe"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cache_dir = Path(self.optimizer_params.get("rgpe_cache_dir") or DEFAULT_RGPE_CACHE_DIR)
        self.cache_path = cache_dir / ("%s_seed%d.pkl" % (self.artifact_tag, self.seed))
        self.rgpe_python = str(self.optimizer_params.get("rgpe_python") or RGPE_PYTHON)
        self._check_assets()

    def _check_assets(self):
        dump = self._require_dump()
        meta = json.loads(self.cache_path.with_suffix(".json").read_text())
        cache_hash = sha256_file(self.cache_path)
        if (meta.get("status") != "complete" or meta["schema_fingerprint"] != self.schema_fingerprint
                or meta["dump_sha256"] != sha256_file(self.dump_path)
                or meta["source_tasks"] != sorted(dump["tasks"]) or meta["cache_sha256"] != cache_hash):
            raise ValueError("RGPE source GP cache is incomplete or incompatible")
        self._set_assets({"cache_sha256": cache_hash, "dump_sha256": meta["dump_sha256"],
                          "normalization": meta["normalization"]})

    def _suggest_model(self, observed):
        pending = self._pending_configs(self.n_pending)
        request = {"target_configs": [self._cfg_dict(c) for c, _ in observed],
                   "y_minimize": [-p for _, p in observed],
                   "pending_configs": [self._cfg_dict(c) for c in pending], "seed": int(self.seed),
                   "n_trials": int(self.optimizer_params.get("n_trials") or self.evaluation_num_limit),
                   "num_posterior_samples": 1000}
        payload = self._run_worker(self.rgpe_python, "rgpe_worker.py", RGPE_REPO,
                                   ["--dump", str(self.dump_path), "--cache", str(self.cache_path)], request)
        return self._selected(payload, pending), "RGPE(TAF)"


class OptFormerOptimizer(_TransferHPOOptimizer):
    optimizer_label = "optformer"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        directory = self.optimizer_params.get("optformer_model_dir")
        if not directory:
            raise ValueError("OptFormer requires optformer_model_dir with completed ML training")
        self.model_dir = Path(directory)
        self.optformer_python = str(self.optimizer_params.get("optformer_python") or OPTFORMER_PYTHON)
        self._check_assets()

    def _check_assets(self):
        meta = json.loads((self.model_dir / "meta.json").read_text())
        if (meta.get("status") != "complete" or meta.get("method") != "optformer_ml"
                or meta.get("model_size") != "base"
                or meta.get("numeric_converter") != "continuous_numeric_v1"
                or meta["checkpoint_sha256"] != sha256_directory(self.model_dir / meta["checkpoint"])
                or meta["model_config_sha256"] != sha256_file(self.model_dir / "model.gin")
                or meta["sequences_sha256"] != sha256_file(meta["sequences"])
                or meta["vocabulary_sha256"] != sha256_file(meta["vocabulary"])
                or meta["train_manifest_sha256"] != sha256_file(TRAIN_MANIFEST)):
            raise ValueError("OptFormer needs a verified completed base-model ML checkpoint")
        self._set_assets({key: meta[key] for key in (
            "checkpoint_sha256", "model_config_sha256", "sequences_sha256", "vocabulary_sha256")})

    def _worker_env(self, pythonpath):
        env = super()._worker_env(pythonpath + os.pathsep + T5X_REPO)
        env.setdefault("TF_NUM_INTEROP_THREADS", "1")
        env.setdefault("TF_NUM_INTRAOP_THREADS", "1")
        env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.2")
        return env

    def _suggest(self):
        if self.trial_cnt >= self.n_startup and not self._observed():
            self._proposal_diagnostics = None
            self._check_assets()
            return self.config_space.get_default_configuration(), "labelled default: no successful initialization"
        return super()._suggest()

    def _suggest_model(self, observed):
        request = {
            "task_id": self.task_id, "configspace": configspace_spec(self.config_space),
            "algorithm_condition": "unknown_mixed", "configs": [self._cfg_dict(c) for c, _ in observed],
            "y_maximize": [p for _, p in observed], "seed": int(self.rng.randint(2 ** 31 - 1)),
            "num_samples": int(self.optimizer_params.get("optformer_num_samples", 100)),
        }
        payload = self._run_worker(self.optformer_python, "optformer_worker.py", OPTFORMER_REPO,
                                   ["--model-dir", str(self.model_dir)], request)
        config = Configuration(self.config_space, values=payload["config"])
        self.config_space.check_configuration(config)
        repeated = self._key(config) in self.attempted_configs
        note = "OptFormer prior samples ranked by learned EI"
        resolution = payload.get("configuration_resolution", {})
        if resolution.get("configuration_changed"):
            note += "; ConfigSpace default repair: " + resolution["route"]
            note += "; acquisition scored the original proposal"
        if repeated:
            note += "; repeated proposal evaluated within actual-call budget"
        return config, note
