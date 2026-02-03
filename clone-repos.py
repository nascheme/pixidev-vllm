#!/usr/bin/env python3
"""Clone git repositories listed in git-repos.txt and apply patches."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def run(cmd, **kwargs):
    print(f"  $ {shlex.join(str(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", action="append", default=None,
        help="Only clone the specified repo(s). Can be passed multiple times.",
    )
    args = parser.parse_args()
    requested = set(args.repo) if args.repo else None

    root = Path(__file__).resolve().parent
    repos_file = root / "git-repos.txt"
    patches_dir = root / "patches"

    with open(repos_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                print(f"Skipping malformed line: {line!r}", file=sys.stderr)
                continue

            org, repo, commit = parts

            if requested and repo not in requested:
                continue

            dest = root / repo / repo.lower()

            if dest.exists():
                print(f"[skip] {dest} already exists")
                continue

            print(f"[clone] {org}/{repo} @ {commit}")
            dest.parent.mkdir(parents=True, exist_ok=True)

            run(["git", "clone", f"https://github.com/{org}/{repo}.git", str(dest)])
            run(["git", "checkout", commit], cwd=dest)
            run(["git", "submodule", "update", "--init", "--recursive"], cwd=dest)

            # Apply patches if any exist
            repo_patches = patches_dir / repo
            if repo_patches.is_dir():
                patches = sorted(repo_patches.iterdir())
                if patches:
                    print(f"[patch] Applying {len(patches)} patch(es) to {repo}")
                    run(["git", "am"] + [str(p) for p in patches], cwd=dest)

    print("Done.")


if __name__ == "__main__":
    main()
