"""Explicit source-pinned recorder execution migrations, no tensor changes."""

LEGACY_RUNTIME = "95e25b24ebf61455b80b43f5b1287dc39e6a1f81fa2d0230f2f7b1d0ead249fc"
BASE_RUNTIME = "c69efde1fc6aa47681aa03c69c3aba5ac9df80c6eb211639fe3d97d545f189a0"
V34_RUNTIME = "17ab6bfe6f6a1136cbb6590a33c8afb763c5652743fcb3c8670d8fc760a48dda"
BASE_WRITES = "092039ca008262c6df974a9af27b5072e07d659765399f2fc2b838641d3a7235"
V34_WRITES = "6e558d6504b53ca73369e9db003ce7157381c2204c37c0cfeff77cdc320a800d"


def migrate_sources(old, new, checkpoint):
    allowed = {
        "runtime.py": {
            (LEGACY_RUNTIME, BASE_RUNTIME),
            (LEGACY_RUNTIME, V34_RUNTIME),
            (BASE_RUNTIME, V34_RUNTIME),
        },
        "writes.py": {(BASE_WRITES, V34_WRITES)},
    }
    changed = {}
    for name in set(old) | set(new):
        a, b = old.get(name), new.get(name)
        if a == b:
            continue
        if (a, b) not in allowed.get(name, set()):
            raise ValueError(f"Unverified event execution source migration: {name}")
        changed[name] = {"from": a, "to": b}
    if not changed:
        return None
    result = {
        "source": checkpoint,
        "source_changes": changed,
        "optimizers_preserved": True,
    }
    if "runtime.py" in changed:
        result["from_runtime_sha256"] = old["runtime.py"]
        result["to_runtime_sha256"] = new["runtime.py"]
    # Validate all differences before changing the temporary saved contract.
    old.update({name: new[name] for name in changed})
    return result
