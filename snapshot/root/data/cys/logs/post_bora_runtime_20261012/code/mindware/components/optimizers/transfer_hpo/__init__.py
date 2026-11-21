"""FSBO / RGPE transfer-HPO helpers (ML sklearn spaces, no VAE)."""

def __getattr__(name):
    if name in {"ALGORITHMS", "SOURCE_VARIANT_PRIORITY", "all_group_ids", "group_id"}:
        from . import schema
        return getattr(schema, name)
    raise AttributeError(name)
