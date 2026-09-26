"""
Vendor the pgsg_1 code that pgsg2 depends on into this repository.

Why: PGSGv2Model (pgsg_v2.py), PLSModel, the Mango loader and the NIR
preprocessor live in the local pgsg_1 working copy, which is not public.
Without them, the code announced in the manuscript's Data and code
availability section cannot be run by a reader.

What it does
------------
1. Entry points: <pgsg1-root>/pgsg_v2.py plus every `pgsg_1` import found in
   this repository's scripts/, src/ and tests/.
2. Resolves imports transitively (absolute `pgsg_1.*` and relative imports
   inside the package, plus parent-package __init__ files) by static AST
   parsing; nothing is executed.
3. Copies the resolved files into vendor/pgsg_1/ with the same layout as the
   pgsg_1 root (pgsg_v2.py at the top, the package under src/pgsg_1/), so
   that `--pgsg1-root vendor/pgsg_1` works unchanged.
4. Writes vendor/pgsg_1/MANIFEST.json: source commit of pgsg_1 (if it is a
   git repository), dirty flag, and the SHA-256 of every vendored file.
5. --check: imports the vendored modules in a fresh interpreter to prove the
   snapshot is self-contained.

Usage:
    python scripts/vendor_pgsg1.py --pgsg1-root /home/clarimar/Dropbox/pgsg/pgsg_1 --check
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PKG = "pgsg_1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module_file(src: Path, dotted: str) -> Path | None:
    """Map 'pgsg_1.a.b' to src/pgsg_1/a/b.py or src/pgsg_1/a/b/__init__.py."""
    base = src.joinpath(*dotted.split("."))
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    return None


def module_name(src: Path, path: Path) -> str:
    rel = path.relative_to(src).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def imports_of(path: Path, current: str | None) -> set[str]:
    """Dotted pgsg_1 module names imported by `path` (current: its module name)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    is_pkg = path.name == "__init__.py"
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == PKG or a.name.startswith(PKG + "."):
                    found.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                mod = node.module or ""
                if not (mod == PKG or mod.startswith(PKG + ".")):
                    continue
            else:
                if current is None:
                    continue
                parts = current.split(".")
                if not is_pkg:
                    parts = parts[:-1]
                parts = parts[: len(parts) - (node.level - 1)]
                mod = ".".join(parts + ([node.module] if node.module else []))
            found.add(mod)
            for a in node.names:  # `from pkg import submodule`
                found.add(f"{mod}.{a.name}")
    return found


def resolve(src: Path, entries: set[str]) -> set[Path]:
    todo, done = list(entries), set()
    files: set[Path] = set()
    while todo:
        mod = todo.pop()
        if mod in done:
            continue
        done.add(mod)
        parts = mod.split(".")
        for k in range(1, len(parts) + 1):  # parent packages first
            parent = ".".join(parts[:k])
            f = module_file(src, parent)
            if f is not None and f not in files:
                files.add(f)
                todo.extend(imports_of(f, parent))
    return files


def git_info(root: Path) -> dict:
    def run(*cmd):
        return subprocess.run(["git", "-C", str(root), *cmd], capture_output=True, text=True)
    head = run("rev-parse", "HEAD")
    if head.returncode != 0:
        return {"git": False}
    dirty = run("status", "--porcelain", "--", "pgsg_v2.py", "src/pgsg_1").stdout.strip()
    return {"git": True, "commit": head.stdout.strip(), "dirty": bool(dirty)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgsg1-root", required=True)
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--dest", default="vendor/pgsg_1")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    root1 = Path(args.pgsg1_root).resolve()
    src = root1 / "src"
    repo = Path(args.repo_root).resolve()
    dest = (repo / args.dest).resolve()
    v2 = root1 / "pgsg_v2.py"
    if not v2.is_file() or not (src / PKG / "__init__.py").is_file():
        raise SystemExit(f"Expected {v2} and {src / PKG / '__init__.py'}")

    entries = imports_of(v2, None)
    consumers = [p for d in ("scripts", "src", "tests") for p in (repo / d).rglob("*.py")
                 if "deprecated" not in p.parts and args.dest.split("/")[0] not in p.parts]
    for p in consumers:
        entries |= imports_of(p, None)
    files = resolve(src, entries)

    if dest.exists():
        shutil.rmtree(dest)
    (dest / "src").mkdir(parents=True)
    shutil.copy2(v2, dest / "pgsg_v2.py")
    manifest = {"pgsg_v2.py": sha256(v2)}
    for f in sorted(files):
        rel = f.relative_to(src)
        (dest / "src" / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest / "src" / rel)
        manifest[f"src/{rel.as_posix()}"] = sha256(f)

    meta = {"source_root": str(root1), "source": git_info(root1),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "entry_modules": sorted(entries), "files": manifest}
    (dest / "MANIFEST.json").write_text(json.dumps(meta, indent=2))
    (dest / "README.md").write_text(
        "# Vendored snapshot of pgsg_1\n\n"
        "Minimal copy of the pgsg_1 code used by this repository (PGSGv2Model, PLSModel,\n"
        "Mango DMC loader, NIR preprocessor), created by `scripts/vendor_pgsg1.py`.\n"
        "Do not edit; regenerate instead. SHA-256 of every file and the source commit are\n"
        "in MANIFEST.json. Use it with `--pgsg1-root vendor/pgsg_1`.\n")
    print(f"Vendored {len(manifest)} files into {dest}")
    for k in sorted(manifest):
        print(f"  {k}")
    if meta["source"].get("dirty"):
        print("WARNING: pgsg_1 has uncommitted changes in the vendored files; "
              "the manifest records the working-copy content, not the commit.")

    if args.check:
        mods = sorted(m for m in {module_name(src, f) for f in files})
        code = ("import sys; sys.path[:0] = [%r, %r]; import pgsg_v2; "
                "[__import__(m) for m in %r]; "
                "import pgsg_1; assert pgsg_1.__file__.startswith(%r), pgsg_1.__file__; "
                "print('import check OK:', len(%r) + 1, 'modules')"
                % (str(dest), str(dest / "src"), mods, str(dest), mods))
        r = subprocess.run([sys.executable, "-I", "-c", code], cwd=str(dest), capture_output=True, text=True)
        print(r.stdout.strip() or r.stderr.strip())
        if r.returncode != 0:
            raise SystemExit("Import check failed: a dependency was not vendored (see message above).")


if __name__ == "__main__":
    main()
