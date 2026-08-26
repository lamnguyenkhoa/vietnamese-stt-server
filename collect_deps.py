"""Copy the runtime dependency closure of requirements.txt out of a local
environment's site-packages into a target directory.

Used by build_local.ps1 to assemble an offline/portable build from packages that
are already installed here, instead of pip-installing them from the internet.
Only the dists reachable from requirements.txt are copied, so the conversion-only
extras that also live in the dev venv (torch, transformers, ...) are left behind.

Usage:
    python collect_deps.py requirements.txt <dest-site-packages> [--list]
"""

import argparse
import shutil
import sys
from importlib import metadata as md
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def parse_requirements(path: Path) -> list[Requirement]:
    reqs = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line and not line.startswith("-"):
            reqs.append(Requirement(line))
    return reqs


def resolve_closure(roots: list[Requirement]) -> dict[str, md.Distribution]:
    """Walk Requires-Dist from the roots, keeping only deps whose environment
    markers hold for this interpreter/platform and the extras actually requested."""
    resolved: dict[str, md.Distribution] = {}
    missing: list[str] = []
    queue = [(r.name, frozenset(r.extras)) for r in roots]
    seen: set[tuple[str, frozenset]] = set()

    while queue:
        name, extras = queue.pop()
        key = (canonicalize_name(name), extras)
        if key in seen:
            continue
        seen.add(key)
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            missing.append(name)
            continue
        resolved[canonicalize_name(name)] = dist
        for spec in dist.metadata.get_all("Requires-Dist") or []:
            dep = Requirement(spec)
            # An extra-gated dep is only pulled in when that extra was requested;
            # a dep with no marker is always required.
            if dep.marker and not any(
                dep.marker.evaluate({"extra": e}) for e in (extras or {""})
            ):
                continue
            queue.append((dep.name, frozenset(dep.extras)))

    if missing:
        sys.exit(
            "Not installed in this environment: "
            + ", ".join(sorted(set(missing)))
            + "\nInstall them here first (pip install -r requirements.txt)."
        )
    return resolved


def copy_dist(dist: md.Distribution, dest_root: Path) -> int:
    # Each dist is copied out of the directory it actually lives in, which is not
    # always the venv's own site-packages (a venv layered on a base install can
    # resolve some dists to the base environment).
    src_root = Path(dist.locate_file(""))
    copied = 0
    for entry in dist.files or []:
        rel = Path(str(entry))
        # RECORD can reference files outside site-packages (Scripts\uvicorn.exe,
        # headers, data files). The app is launched via `python -m`, so skip them.
        if ".." in rel.parts:
            continue
        src = src_root / rel
        if not src.is_file():
            continue
        dest = dest_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied += 1
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requirements", type=Path)
    parser.add_argument("dest", type=Path, nargs="?")
    parser.add_argument("--list", action="store_true", help="print the closure, copy nothing")
    args = parser.parse_args()

    dists = resolve_closure(parse_requirements(args.requirements))

    if args.list or args.dest is None:
        for name in sorted(dists):
            print(f"{dists[name].metadata['Name']}=={dists[name].version}")
        return

    args.dest.mkdir(parents=True, exist_ok=True)
    total = 0
    for name in sorted(dists):
        total += copy_dist(dists[name], args.dest)
    print(f"Copied {len(dists)} packages ({total} files) into {args.dest}")


if __name__ == "__main__":
    main()
