"""TaskSet FSBO min-max proposals from a reusable isolated CPU subprocess."""
import json
import math
import os
from pathlib import Path
import select
import subprocess
import time

import numpy as np

from mindware.domains.dlhpo.taskset_lolbo import TaskSetCodec
from mindware.domains.search import _optuna_distributions
from mindware.domains.dlhpo.taskset_llm.spec import signature
from mindware.domains.dlhpo.taskset_llm.inference import sha256_file

ROOT = Path(__file__).resolve().parents[4]
FSBO_PYTHON = "/root/data/cys/baseline/envs/bl_fsbo_exact/bin/python"


class PersistentFSBO:
    def __init__(self, assets, directory, *, timeout=300):
        self.assets, self.directory = Path(assets), Path(directory)
        self.timeout, self.process = timeout, None
        self.meta = json.loads((self.assets / "meta.json").read_text())
        if self.meta["target_scaling"] != "minmax" or sha256_file(self.assets / "weights") != self.meta["weights_sha256"]:
            raise ValueError("Wrong FSBO min-max checkpoint")
        self.request_count = 0
        self._log = None

    def request(self, request):
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.process is None:
            self._log = (self.directory / "persistent_worker.log").open("a")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH="/root/data/cys/baseline/repos/FSBO",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            self.process = subprocess.Popen([FSBO_PYTHON, "-u", str(ROOT / "examples/benchmark/domains/taskset_fsbo_persistent_worker.py"),
                                             "--ckpt", str(self.assets), "--log", str(self.directory / "fit.log")],
                                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
                                            text=True, bufsize=1, env=env)
        self.request_count += 1
        message = {"id": self.request_count, "request": request}
        started = time.monotonic()
        self.process.stdin.write(json.dumps(message, allow_nan=False) + "\n")
        self.process.stdin.flush()
        if not select.select([self.process.stdout], [], [], self.timeout)[0]:
            self.close()
            raise TimeoutError("FSBO service exceeded request timeout")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("FSBO service exited; inspect persistent_worker.log")
        response = json.loads(line)
        if response.get("id") != self.request_count or not response.get("ok"):
            raise RuntimeError(f"FSBO service error: {response.get('error', 'response identity mismatch')}")
        result = response["result"]
        if result["checkpoint_sha256"] != self.meta["weights_sha256"] or result["target_scaling"] != "minmax":
            raise ValueError("FSBO service checkpoint/scaling mismatch")
        result.update(service_pid=self.process.pid, request_seconds=time.monotonic() - started,
                      service_compute_seconds=response["seconds"])
        return result

    def close(self):
        if self.process is not None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=5)
            self.process.stdout.close()
            self.process = None
        if self._log is not None:
            self._log.close()
            self._log = None


class FSBOBridge:
    def __init__(self, cs, assets, directory, *, seed=1):
        self.cs, self.seed = cs, seed
        self.codec = TaskSetCodec(cs, _optuna_distributions(cs))
        self.worker = PersistentFSBO(assets, directory)
        self.meta = self.worker.meta
        probe = self.meta["encoding_probe"]
        if self.codec.dim != 43 or not np.allclose(self.codec.encode(probe["config"]), probe["vector"], atol=1e-7, rtol=0):
            raise ValueError("TaskSet FSBO encoding mismatch")

    def __call__(self, observations, trial):
        self.cs.seed(self.seed + 1009 * trial)
        seen = {signature(row["config"]) for row in observations}
        source = [(i + 1, row) for i, row in enumerate(observations)
                  if row["trial_state"] == 0 and math.isfinite(float(row["objectives"][0]))]
        need = 1 if len(observations) < 3 or len(source) < 2 else 256
        candidates = []
        for _ in range(10000):
            config = dict(self.cs.sample_configuration().get_dictionary())
            key = signature(config)
            if key not in seen:
                candidates.append(config)
                seen.add(key)
            if len(candidates) == need:
                break
        if len(candidates) != need:
            raise RuntimeError("Could not construct distinct unseen FSBO candidates")
        if need == 1:
            return candidates[0], {"phase": "random_initialization", "candidate_count": 1}
        request = {"schema_fingerprint": self.meta["schema_fingerprint"],
                   "X_obs": [self.codec.encode(row["config"]).tolist() for _, row in source],
                   "y_obs": [-float(row["objectives"][0]) for _, row in source],
                   "X_pen": [self.codec.encode(config).tolist() for config in candidates],
                   "seed": self.seed, "finetune_epochs": 100, "target_scaling": "minmax"}
        result = self.worker.request(request)
        scores, index = result["acquisition"]["values"], result["index"]
        if (type(index) is not int or not 0 <= index < len(candidates) or len(scores) != len(candidates)
                or not all(math.isfinite(value) for value in scores)
                or index != max(range(len(scores)), key=scores.__getitem__)):
            raise ValueError("FSBO selected index disagrees with candidate EI values")
        result.update(phase="fsbo_ei", candidates=candidates,
                      source_observation_indices=[index for index, _ in source], candidate_count=len(candidates))
        return candidates[index], result

    def close(self):
        self.worker.close()
