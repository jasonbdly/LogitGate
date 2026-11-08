---
name: logitgate-release
description: >-
  Prepares, verifies, or executes a LogitGate version release. Use whenever
  bumping version numbers, cutting a release, validating version metadata
  synchronization, or creating and pushing release tags.
---

# LogitGate Release Runbook & Skill

This skill defines the authoritative procedure for cutting and verifying a LogitGate release to prevent version desynchronization, broken documentation snippets, or unverified builds.

## Release Architecture

LogitGate coordinates versions across five files:
1. `pyproject.toml`
2. `src/logitgate/__init__.py`
3. `CITATION.cff`
4. `README.md`
5. `CHANGELOG.md`

Refer to [Version Locations Reference](./references/version_locations.md) for details on format and regex requirements.

---

## The Automated Release Flow

Use the bundled verification script [verify_release.py](./scripts/verify_release.py) to automate and validate the entire release lifecycle.

### Step 1: Pre-Flight Verification, Clean Working Tree, & Quality Gates

Before touching any version numbers, ensure **all functional changes are already committed in separate, atomic Conventional Commits** with descriptive bodies (`git status` must be completely clean), and that the codebase is 100% green:

```bash
# 0. Verify no uncommitted functional changes exist
git status --porcelain

# 1. Check linter
uvx ruff check src/logitgate tests

# 2. Strict static typing
uv run --with mypy mypy src/logitgate tests

# 3. Full test suite (including executable README snippets)
uv run pytest
```

> **CRITICAL GIT HYGIENE RULE:** Never bundle functional code, test, or CI changes into a `chore(release): bump version to X.Y.Z` commit. Commit all feature, fix, test, and CI changes in dedicated atomic commits *before* running `--bump`.

### Step 2: Atomic Version Bump

Use the helper script to bump all 5 version locations simultaneously:

```bash
# Example: Bumping to 0.3.1
python3 .agents/skills/logitgate-release/scripts/verify_release.py --bump 0.3.1
```

This atomically updates:
- `pyproject.toml` (`version = "0.3.1"`)
- `src/logitgate/__init__.py` (`__version__ = "0.3.1"`)
- `CITATION.cff` (`version: 0.3.1` and `date-released: "YYYY-MM-DD"`)
- `README.md` (BibTeX citation `version = {0.3.1},`)
- `CHANGELOG.md` (converts `## [Unreleased]` to `## [0.3.1] - YYYY-MM-DD` or inserts a new release header)

### Step 3: Verify Changelog Completeness

Manually inspect `CHANGELOG.md` to ensure all user-visible changes are properly categorized:
- `### Added`: New gates, traits, CLI commands, or capabilities.
- `### Changed`: Behavioral changes or API adjustments.
- `### Refactored`: Code deduplication, algorithmic optimizations, and layout improvements.
- `### Fixed`: Bug fixes and regression repairs.

### Step 4: Run Automated Pre-Flight Check

Run the full pre-flight verification script to confirm synchronization, linting, typing, tests, and build artifacts:

```bash
python3 .agents/skills/logitgate-release/scripts/verify_release.py --full-check
```

### Step 5: Git Commit & Annotated Tag

Once `--full-check` passes, stage **only** the 5 release metadata files (never use `git add -A` on a release bump commit):

```bash
# Stage strictly the 5 synchronized release metadata files
git add pyproject.toml src/logitgate/__init__.py CITATION.cff README.md CHANGELOG.md

# Commit using conventional format with a body summarizing the release metadata sync
git commit -m "chore(release): bump version to X.Y.Z" -m "Synchronize version metadata across pyproject.toml, src/logitgate/__init__.py, CITATION.cff, README.md, and CHANGELOG.md for the X.Y.Z release."

# Create annotated release tag
git tag -a vX.Y.Z -m "Release vX.Y.Z"
```

### Step 6: Push to Remote

Always test the push with `--dry-run` first:

```bash
# 1. Dry run
git push --dry-run origin main --tags

# 2. Execute push
git push origin main --tags

# 3. Confirm clean tracking state
git status
```

---

## Quick Reference Commands

| Task | Command |
| :--- | :--- |
| **Verify Version Sync** | `python3 .agents/skills/logitgate-release/scripts/verify_release.py --check` |
| **Bump Version Atomically** | `python3 .agents/skills/logitgate-release/scripts/verify_release.py --bump X.Y.Z` |
| **Full Pre-Flight Suite** | `python3 .agents/skills/logitgate-release/scripts/verify_release.py --full-check` |
| **Test Distribution Build** | `uv build` |
