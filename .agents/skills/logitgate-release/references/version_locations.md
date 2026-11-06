# LogitGate Version Locations Reference

LogitGate tracks version metadata across five authoritative files. Whenever cutting a release or bumping versions, all five locations must be kept strictly synchronized.

---

## The 5 Synchronized Files

| File | Exact Key / Syntax | Purpose |
| :--- | :--- | :--- |
| **`pyproject.toml`** | `version = "X.Y.Z"` under `[project]` | Core package metadata for `pip`, `uv`, and PyPI distribution builds. |
| **`src/logitgate/__init__.py`** | `__version__ = "X.Y.Z"` | Programmatic access at runtime (`import logitgate; logitgate.__version__`), CLI banners, and benchmark metadata. |
| **`CITATION.cff`** | `version: X.Y.Z`<br>`date-released: "YYYY-MM-DD"` | Academic and industry citation metadata conforming to the Citation File Format (CFF). |
| **`README.md`** | `version = {X.Y.Z},` inside the BibTeX citation block | Copy-pasteable BibTeX citation for researchers and papers. |
| **`CHANGELOG.md`** | `## [X.Y.Z] - YYYY-MM-DD` | Keep a Changelog standard release header listing Added, Changed, Refactored, and Fixed items. |

---

## Format & Patterns

### 1. `pyproject.toml`
```toml
[project]
name = "logitgate"
version = "0.3.0"
```

### 2. `src/logitgate/__init__.py`
```python
__version__ = "0.3.0"
```

### 3. `CITATION.cff`
```yaml
cff-version: 1.2.0
message: "If you use LogitGate in your research or production systems, please cite it as below."
title: "LogitGate: Calibrated First-Token Logit Evaluation and Cascaded Routing for LLMs"
version: 0.3.0
date-released: "2026-11-05"
```

### 4. `README.md`
```bibtex
@software{bradley2026logitgate,
  author = {Bradley, Jason},
  title = {LogitGate: Calibrated First-Token Logit Evaluation and Cascaded Routing for LLMs},
  year = {2026},
  version = {0.3.0},
  url = {https://github.com/jasonbdly/LogitGate}
}
```

### 5. `CHANGELOG.md`
```markdown
## [0.3.0] - 2026-11-05

### Added
...

### Changed
...

### Refactored
...

### Fixed
...
```
