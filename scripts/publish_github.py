#!/usr/bin/env python3
"""Test, commit and push this release to an explicitly specified GitHub repository.

Create an empty repository on GitHub first and authenticate git (SSH or its
credential helper). This script neither embeds tokens nor force-pushes history.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args, capture=False):
    return subprocess.run(
        args,
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    ).stdout


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--remote",
        required=True,
        help="git@github.com:OWNER/REPO.git or https://github.com/OWNER/REPO.git",
    )
    p.add_argument("--branch", default="main")
    p.add_argument(
        "--message",
        default="Prepare safedelta package and audited reproduction workflows",
    )
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if not re.fullmatch(
        r"(?:git@github\.com:|https://github\.com/)[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?",
        a.remote,
    ):
        p.error("an explicit credential-free GitHub repository URL is required")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", a.branch) or ".." in a.branch:
        p.error("invalid branch")
    run(sys.executable, str(ROOT / "scripts/check_integrity.py"))
    run(sys.executable, "-m", "pytest", "-q")
    run(sys.executable, "-m", "build")
    distributions = sorted(str(x) for x in (ROOT / "dist").glob("*") if x.is_file())
    run(sys.executable, "-m", "twine", "check", *distributions)
    if a.dry_run:
        print(
            "Validated. Would commit this checkout and push HEAD to", a.remote, a.branch
        )
        return
    if not (ROOT / ".git").exists():
        run("git", "init", "--initial-branch", a.branch)
    top = Path(
        run("git", "rev-parse", "--show-toplevel", capture=True).strip()
    ).resolve()
    if top != ROOT:
        raise RuntimeError("refusing to stage a parent repository")
    # Check identity before staging anything. Configure git user.name/email if missing.
    run("git", "var", "GIT_AUTHOR_IDENT", capture=True)
    remotes = run("git", "remote", capture=True).splitlines()
    if "origin" in remotes:
        if run("git", "remote", "get-url", "origin", capture=True).strip() != a.remote:
            raise RuntimeError(
                "origin differs from requested remote; configure it explicitly"
            )
    else:
        run("git", "remote", "add", "origin", a.remote)
    run("git", "add", "--all")
    changed = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=ROOT
    ).returncode
    if changed == 1:
        run("git", "commit", "-m", a.message)
    elif changed != 0:
        raise RuntimeError("cannot inspect staged changes")
    run("git", "push", "--set-upstream", "origin", f"HEAD:refs/heads/{a.branch}")


if __name__ == "__main__":
    main()
