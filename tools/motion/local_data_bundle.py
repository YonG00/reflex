"""Pack/restore the six required motion files for the same licensed user's backup.

The bundle contains restricted third-party data. Do not publish it in a public repo.
Only the script and checksum manifest belong in Git. No GPU or third-party Python
packages are needed to restore a bundle.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile

FILES = (
    "bones_seed/g1/csv/210707/walk_forward_grab_normal_003__A006_M.csv",
    "amass_g1/g1/ACCAD/Male2Walking_c3d/B10 -  Walk turn left 45_poses_120_jpos.npy",
    "motions/walk_then_catch_v17_30s.npz",
    "motions/male2_walk_v2.npz",
    "motions/male2_walk_turn_left_45.npz",
    "amp/amp_dataset.pt",
)
RAW_CHECKSUMS = {
    FILES[0]: "7979e92da40a2241341d02609a37381b92fe00af3f755b20453a65dc3339a8af",
    FILES[1]: "f7be749b1f59500c48a4c1421c00f6fefa36052b154b97dcdb70658ce031404f",
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pack(args):
    records = []
    for name in FILES:
        path = args.data_dir / name
        sha = digest(path)
        if name in RAW_CHECKSUMS and sha != RAW_CHECKSUMS[name]:
            raise ValueError("Unexpected source checksum: " + name)
        records.append({"path": name, "bytes": path.stat().st_size, "sha256": sha})
    manifest = {
        "format_version": 1,
        "purpose": "Internal backup for the licensed user; not for public redistribution",
        "files": records,
    }
    payload = json.dumps(manifest, indent=2).encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w:gz") as tar:
        info = tarfile.TarInfo("manifest.json")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
        for name in FILES:
            tar.add(args.data_dir / name, arcname=name, recursive=False)
    if args.manifest_output:
        args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_output.write_bytes(payload + b"\n")
    print(json.dumps({"bundle": str(args.output), "bytes": args.output.stat().st_size,
                      "sha256": digest(args.output), "files": len(records)}))


def restore(args):
    args.data_dir.mkdir(parents=True, exist_ok=True)
    root = args.data_dir.resolve()
    with tarfile.open(args.bundle, "r:gz") as tar:
        members = tar.getmembers()
        if len(members) != len(FILES) + 1 or {m.name for m in members} != set(FILES) | {"manifest.json"}:
            raise ValueError("Unexpected or duplicate archive members")
        if any(not m.isfile() for m in members):
            raise ValueError("Only regular files are supported")
        manifest = json.load(tar.extractfile("manifest.json"))
        records = {item["path"]: item for item in manifest["files"]}
        if manifest["format_version"] != 1 or set(records) != set(FILES):
            raise ValueError("Unexpected manifest")
        with tempfile.TemporaryDirectory(prefix=".reflex-restore-", dir=root) as temp:
            staging = Path(temp)
            for name in FILES:
                dest = root / name
                if not dest.resolve().is_relative_to(root):
                    raise ValueError("Destination escapes data directory: " + name)
                record = records[name]
                if dest.exists() and digest(dest) != record["sha256"]:
                    raise FileExistsError("Refusing to overwrite a different file: " + str(dest))
                src = tar.extractfile(name)
                target = staging / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("wb") as f:
                    for chunk in iter(lambda: src.read(1024 * 1024), b""):
                        f.write(chunk)
                if target.stat().st_size != record["bytes"] or digest(target) != record["sha256"]:
                    raise ValueError("Bundle checksum mismatch: " + name)
            for name in FILES:
                dest = root / name
                if not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(staging / name, dest)
    print(json.dumps({"state": "restored_and_verified", "files": len(FILES), "data_dir": str(root)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest-output", type=Path)
    p = sub.add_parser("restore")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    (pack if args.command == "pack" else restore)(args)


if __name__ == "__main__":
    main()
