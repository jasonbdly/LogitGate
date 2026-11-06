#!/usr/bin/env python3
"""
Release verification and version synchronization script for LogitGate.

Manages and verifies the 5 authoritative version locations:
1. pyproject.toml             (project.version)
2. src/logitgate/__init__.py  (__version__)
3. CITATION.cff               (version & date-released)
4. README.md                  (BibTeX citation version)
5. CHANGELOG.md               (release header ## [X.Y.Z] - YYYY-MM-DD)

Usage:
    python .agents/skills/logitgate-release/scripts/verify_release.py --check
    python .agents/skills/logitgate-release/scripts/verify_release.py --bump 0.3.1
    python .agents/skills/logitgate-release/scripts/verify_release.py --full-check
"""

from __future__ import annotations

import argparse
import datetime
import re
import subprocess
import sys
from pathlib import Path


def find_repo_root() -> Path:
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").is_file() and (current / ".git").exists():
            return current
        current = current.parent
    return Path.cwd()


def get_versions(root: Path) -> dict[str, str]:
    versions: dict[str, str] = {}

    # 1. pyproject.toml
    pyproject_text = (root / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', pyproject_text, re.MULTILINE)
    if not m:
        raise ValueError("Could not find version in pyproject.toml")
    versions["pyproject.toml"] = m.group(1)

    # 2. src/logitgate/__init__.py
    init_text = (root / "src/logitgate/__init__.py").read_text(encoding="utf-8")
    m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', init_text, re.MULTILINE)
    if not m:
        raise ValueError("Could not find __version__ in src/logitgate/__init__.py")
    versions["src/logitgate/__init__.py"] = m.group(1)

    # 3. CITATION.cff
    citation_text = (root / "CITATION.cff").read_text(encoding="utf-8")
    m = re.search(r'^version:\s*([0-9a-zA-Z.-]+)', citation_text, re.MULTILINE)
    if not m:
        raise ValueError("Could not find version in CITATION.cff")
    versions["CITATION.cff"] = m.group(1)

    # 4. README.md
    readme_text = (root / "README.md").read_text(encoding="utf-8")
    m = re.search(r'version\s*=\s*\{([0-9a-zA-Z.-]+)\}', readme_text)
    if not m:
        raise ValueError("Could not find version in README.md BibTeX citation")
    versions["README.md"] = m.group(1)

    # 5. CHANGELOG.md
    changelog_text = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    m = re.search(r'^##\s*\[([0-9a-zA-Z.-]+)\]', changelog_text, re.MULTILINE)
    if not m:
        raise ValueError("Could not find latest version in CHANGELOG.md")
    versions["CHANGELOG.md"] = m.group(1)

    return versions


def check_sync(root: Path) -> tuple[bool, str]:
    versions = get_versions(root)
    unique = set(versions.values())
    if len(unique) == 1:
        v = next(iter(unique))
        return True, v
    return False, ", ".join(f"{k}: {val}" for k, val in versions.items())


def bump_version(root: Path, new_version: str) -> None:
    if not re.match(r"^\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?$", new_version):
        raise ValueError(f"Invalid semver version: '{new_version}' (expected X.Y.Z)")

    today = datetime.date.today().isoformat()
    print(f"Bumping LogitGate version to {new_version} (Release date: {today})")

    # 1. pyproject.toml
    pyproject_path = root / "pyproject.toml"
    content = pyproject_path.read_text(encoding="utf-8")
    new_content, count = re.subn(
        r'^version\s*=\s*["\'][^"\']+["\']',
        f'version = "{new_version}"',
        content,
        count=1,
        flags=re.MULTILINE,
    )
    if count == 0:
        raise ValueError("Failed to update version in pyproject.toml")
    pyproject_path.write_text(new_content, encoding="utf-8")
    print("  ✓ pyproject.toml")

    # 2. src/logitgate/__init__.py
    init_path = root / "src/logitgate/__init__.py"
    content = init_path.read_text(encoding="utf-8")
    new_content, count = re.subn(
        r'^__version__\s*=\s*["\'][^"\']+["\']',
        f'__version__ = "{new_version}"',
        content,
        count=1,
        flags=re.MULTILINE,
    )
    if count == 0:
        raise ValueError("Failed to update __version__ in src/logitgate/__init__.py")
    init_path.write_text(new_content, encoding="utf-8")
    print("  ✓ src/logitgate/__init__.py")

    # 3. CITATION.cff
    citation_path = root / "CITATION.cff"
    content = citation_path.read_text(encoding="utf-8")
    content, _count_v = re.subn(
        r'^version:\s*[0-9a-zA-Z.-]+',
        f"version: {new_version}",
        content,
        count=1,
        flags=re.MULTILINE,
    )
    content, _count_d = re.subn(
        r'^date-released:\s*["\'][^"\']+["\']',
        f'date-released: "{today}"',
        content,
        count=1,
        flags=re.MULTILINE,
    )
    citation_path.write_text(content, encoding="utf-8")
    print("  ✓ CITATION.cff (version and date-released)")

    # 4. README.md
    readme_path = root / "README.md"
    content = readme_path.read_text(encoding="utf-8")
    new_content, count = re.subn(
        r'version\s*=\s*\{[0-9a-zA-Z.-]+\}',
        f"version = {{{new_version}}}",
        content,
        count=1,
    )
    readme_path.write_text(new_content, encoding="utf-8")
    print("  ✓ README.md (BibTeX citation)")

    # 5. CHANGELOG.md
    changelog_path = root / "CHANGELOG.md"
    content = changelog_path.read_text(encoding="utf-8")
    if "## [Unreleased]" in content:
        content = content.replace("## [Unreleased]", f"## [{new_version}] - {today}", 1)
    else:
        # Prepend a new section right above the latest release
        latest_header = re.search(r'^##\s*\[', content, re.MULTILINE)
        if latest_header:
            pos = latest_header.start()
            new_section = (
                f"## [{new_version}] - {today}\n\n"
                "### Added\n\n"
                "### Changed\n\n"
                "### Fixed\n\n---\n\n"
            )
            content = content[:pos] + new_section + content[pos:]
    changelog_path.write_text(content, encoding="utf-8")
    print("  ✓ CHANGELOG.md")


def run_command(cmd: list[str], cwd: Path, desc: str) -> bool:
    print(f"Running: {desc} ({' '.join(cmd)})...")
    result = subprocess.run(cmd, check=False, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAILED: {desc} (exit code {result.returncode})")
        if result.stdout:
            print("STDOUT:\n" + result.stdout)
        if result.stderr:
            print("STDERR:\n" + result.stderr)
        return False
    print(f"  ✓ {desc} passed.")
    return True


def run_full_check(root: Path) -> bool:
    print("=== Step 1: Checking Version Synchronization ===")
    synced, info = check_sync(root)
    if not synced:
        print(f"FAIL: Version mismatch detected!\n{info}")
        return False
    print(f"  ✓ All 5 metadata locations synchronized at version {info}")

    print("\n=== Step 2: Quality Gates ===")
    if not run_command(["uvx", "ruff", "check", "src/logitgate", "tests"], root, "Ruff lint"):
        return False

    if not run_command(["uv", "run", "--with", "mypy", "mypy", "src/logitgate", "tests"], root, "Mypy type check"):
        return False

    if not run_command(["uv", "run", "pytest"], root, "Pytest test suite"):
        return False

    print("\n=== Step 3: Package Build Verification ===")
    if not run_command(["uv", "build"], root, "uv build (sdist and wheel)"):
        return False

    print("\n=== Step 4: Git Status Check ===")
    git_status = subprocess.run(["git", "status", "--porcelain"], check=False, cwd=root, capture_output=True, text=True)
    if git_status.stdout.strip():
        print("Note: Working directory has uncommitted changes:")
        print(git_status.stdout)
    else:
        print("  ✓ Working tree is clean.")

    print("\n✓ ALL RELEASE PRE-FLIGHT CHECKS PASSED!")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="LogitGate release and version sync manager")
    parser.add_argument("--check", action="store_true", help="Verify all 5 version locations are synchronized")
    parser.add_argument("--bump", type=str, metavar="VERSION", help="Bump version across all 5 files to VERSION (e.g. 0.3.1)")
    parser.add_argument("--full-check", action="store_true", help="Run full pre-flight verification (version sync + ruff + mypy + pytest + build)")

    args = parser.parse_args()
    root = find_repo_root()

    if args.bump:
        bump_version(root, args.bump)
        synced, info = check_sync(root)
        if not synced:
            print(f"Error: Version sync failed after bump: {info}", file=sys.stderr)
            return 1
        print(f"\nSuccessfully synchronized all files to version {info}")
        return 0

    if args.full_check:
        return 0 if run_full_check(root) else 1

    # Default to --check
    synced, info = check_sync(root)
    if synced:
        print(f"LogitGate version is synchronized: {info}")
        for file, ver in get_versions(root).items():
            print(f"  ✓ {file:28s} {ver}")
        return 0
    else:
        print(f"Error: Version mismatch detected!\n{info}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
