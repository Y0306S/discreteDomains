from mindware.domains.base import DomainAdapter, EvaluationResult, LATENT_DIM
from mindware.domains.dlhpo.adapter import DLHPOAdapter
from mindware.domains.dlhpo.taskset_adapter import TaskSetAdapter
from mindware.domains.protein.adapter import ProteinAdapter
from mindware.domains.queryopt.adapter import QueryOptAdapter
from mindware.domains.queryopt.bolt_exec_adapter import QueryBoltExecAdapter


def get_adapter(domain: str, **kwargs) -> DomainAdapter:
    key = str(domain or "").strip().lower()
    oracle = str(kwargs.pop("oracle", kwargs.pop("oracle_mode", "explain")) or "explain").lower()
    if key in {"protein", "protein_structure", "seq"}:
        return ProteinAdapter(**kwargs)
    if key in {"query", "queryopt", "sql", "join"}:
        if oracle in {"bolt_execute", "bolt_exec", "execution", "runtime"}:
            return QueryBoltExecAdapter(**kwargs)
        return QueryOptAdapter(**kwargs)
    if key in {"dl", "dlhpo", "deep_learning", "nas"}:
        return DLHPOAdapter(**kwargs)
    if key in {"taskset", "task_set", "dl_taskset"}:
        return TaskSetAdapter(**kwargs)
    raise ValueError(f"Unknown domain {domain!r}")


__all__ = [
    "DomainAdapter",
    "EvaluationResult",
    "LATENT_DIM",
    "ProteinAdapter",
    "QueryOptAdapter",
    "QueryBoltExecAdapter",
    "DLHPOAdapter",
    "TaskSetAdapter",
    "get_adapter",
]
