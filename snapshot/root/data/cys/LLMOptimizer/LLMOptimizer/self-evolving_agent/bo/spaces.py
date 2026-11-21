from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class HyperparameterSpec:
    name: str
    kind: str
    lower: float | int
    upper: float | int
    default: float | int | str
    log: bool = False

    @property
    def is_integer(self) -> bool:
        return self.kind == "int"

    @property
    def is_float(self) -> bool:
        return self.kind == "float"

    @property
    def is_numeric(self) -> bool:
        return self.kind in {"int", "float"}


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        return float(int(value))
    return float(value)


class ConfigSpaceAdapter:
    """Convert numeric ConfigSpace values to and from normalized vectors."""

    def __init__(self, specs: list[HyperparameterSpec]):
        if not specs:
            raise ValueError("ConfigSpaceAdapter needs at least one hyperparameter.")
        unsupported = [spec.name for spec in specs if not spec.is_numeric]
        if unsupported:
            raise NotImplementedError(
                "This self-evolving HPO pipeline only supports numeric parameters. "
                f"Unsupported parameters: {unsupported}"
            )
        self.specs = specs
        self.name_to_spec = {spec.name: spec for spec in specs}

    @classmethod
    def from_config_space(cls, config_space) -> "ConfigSpaceAdapter":
        specs: list[HyperparameterSpec] = []
        for name, hp in dict(config_space).items():
            hp_type = type(hp).__name__
            if "Integer" in hp_type:
                specs.append(
                    HyperparameterSpec(
                        name=name,
                        kind="int",
                        lower=int(hp.lower),
                        upper=int(hp.upper),
                        default=int(hp.default_value),
                        log=bool(getattr(hp, "log", False)),
                    )
                )
            elif "Float" in hp_type:
                specs.append(
                    HyperparameterSpec(
                        name=name,
                        kind="float",
                        lower=float(hp.lower),
                        upper=float(hp.upper),
                        default=float(hp.default_value),
                        log=bool(getattr(hp, "log", False)),
                    )
                )
            else:
                specs.append(
                    HyperparameterSpec(
                        name=name,
                        kind="categorical",
                        lower=0,
                        upper=0,
                        default=getattr(hp, "default_value", None),
                    )
                )
        return cls(specs)

    @property
    def dim(self) -> int:
        return len(self.specs)

    def numeric_specs(self) -> list[HyperparameterSpec]:
        return [spec for spec in self.specs if spec.is_numeric]

    def _encode_value(self, spec: HyperparameterSpec, value: Any) -> float:
        x = _as_float(value)
        lower = _as_float(spec.lower)
        upper = _as_float(spec.upper)
        if math.isclose(lower, upper):
            return 0.0

        if spec.log:
            lower = math.log(lower)
            upper = math.log(upper)
            x = math.log(max(x, 1e-12))

        normalized = (x - lower) / (upper - lower)
        return float(np.clip(normalized, 0.0, 1.0))

    def _decode_value(self, spec: HyperparameterSpec, value: float) -> float | int:
        value = float(np.clip(value, 0.0, 1.0))
        lower = _as_float(spec.lower)
        upper = _as_float(spec.upper)
        if math.isclose(lower, upper):
            raw = lower
        else:
            if spec.log:
                lower = math.log(lower)
                upper = math.log(upper)
                raw = math.exp(lower + value * (upper - lower))
            else:
                raw = lower + value * (upper - lower)

        if spec.is_integer:
            return int(round(raw))
        return float(raw)

    def config_to_vector(self, config: dict[str, Any]) -> np.ndarray:
        values: list[float] = []
        for spec in self.numeric_specs():
            if spec.name not in config:
                raise KeyError(f"Missing parameter '{spec.name}'")
            values.append(self._encode_value(spec, config[spec.name]))
        return np.asarray(values, dtype=np.float64)

    def vector_to_config(self, vector: np.ndarray | list[float]) -> dict[str, float | int | Any]:
        vec = np.asarray(vector, dtype=np.float64).reshape(-1)
        if vec.shape[0] != len(self.numeric_specs()):
            raise ValueError(
                f"Expected vector of length {len(self.numeric_specs())}, got {vec.shape[0]}"
            )
        config: dict[str, float | int | Any] = {}
        for idx, spec in enumerate(self.numeric_specs()):
            config[spec.name] = self._decode_value(spec, float(vec[idx]))
        return config

    def random_vector(self, rng: np.random.Generator) -> np.ndarray:
        return rng.random(len(self.numeric_specs()), dtype=np.float64)

    def default_config(self) -> dict[str, Any]:
        return {spec.name: spec.default for spec in self.specs}

    def prompt_space_lines(self) -> str:
        lines = ["CONFIG PARAMETERS (return these in warm_start_config):"]
        for spec in self.specs:
            if spec.kind == "categorical":
                lines.append(f"  {spec.name}: categorical (default = {spec.default})")
                continue
            scale = "  [log scale]" if spec.log else ""
            lines.append(
                f"  {spec.name}: {spec.kind} in [{spec.lower}, {spec.upper}]{scale}  (default = {spec.default})"
            )
        return "\n".join(lines)

    def signature(self, config: dict[str, Any]) -> str:
        return json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
