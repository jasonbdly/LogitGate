# Contributing to LogitGate

Thank you for your interest in contributing to **LogitGate**! We welcome bug reports, feature proposals, documentation improvements, benchmark contributions, and pull requests.

---

## 1. Project Philosophy & Core Principles

1. **Sub-50ms, Zero-Token Evaluation**: LogitGate's primary goal is to eliminate costly autoregressive generation for System-1 evaluation and routing decisions whenever possible.
2. **Mathematical Debiasing Over Heuristics**: Token and positional biases must be cancelled in mathematically sound log-odds space ($L_1 + L_2$) or probability-weighted continuous spaces ($\mathbb{E}[S] = \sum s \cdot P(s)$), not ad-hoc heuristics.
3. **KV-Cache Prefix Preservation**: Multi-rubric or multi-pass evaluations must keep prompt prefixes byte-identical to ensure >99% GPU KV-cache reuse.
4. **Offline-First, Sub-Second Test Suite**: 100% of core unit and integration tests must run fully offline using deterministic stubs (`StubBackend`) in under 1 second without GPU, cloud API keys, or network connectivity.

---

## 2. Development Setup

LogitGate requires **Python 3.10+**.

### Clone and Environment Creation

```bash
git clone https://github.com/jasonbdly/LogitGate.git
cd LogitGate

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in editable mode with test dependencies
pip install --upgrade pip
pip install -e ".[test]"
```

---

## 3. Running Tests & Quality Checks

### Fast Offline Test Suite

Run the full test suite with `pytest`:

```bash
pytest tests -q
```

All 81 unit tests execute in ~0.3s. If your changes introduce new gates or debiasing math, ensure they include comprehensive offline tests using `StubBackend`.

### Type Checking & Linting

```bash
# Type check with mypy
mypy src/logitgate

# Lint checks
ruff check src/logitgate tests
```

---

## 4. Architecture & Extension Guidelines

### Adding a New Backend

All inference backends must inherit from `LogitBackend` (for first-token logits) and optionally `GenerativeBackend` (if supporting Layer 2 generative critique):

```python
from typing import Any
from logitgate.backends.base import GenerativeBackend, LogitBackend

class MyCustomBackend(LogitBackend, GenerativeBackend):
    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url
        self.model = model

    def query_top_logprobs(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        # Return [{'token': 'A', 'logprob': -0.15}, ...]
        ...

    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> tuple[str, int]:
        # Return (critique_text, tokens_generated)
        ...

    def __repr__(self) -> str:
        return f"MyCustomBackend(model='{self.model}')"
```

Register your backend URI scheme in `resolve_logit_backend()` and `resolve_generative_backend()` inside `logitgate/backends/base.py`.

### Adding a New Gate

Custom gates should inherit from `BaseGate` in `logitgate/gates/base.py`:

```python
from logitgate.gates.base import BaseGate
from logitgate.models import Verdict, DecisionStatus, LayerTrace
import time

class CustomGate(BaseGate):
    def __init__(self, name: str = "CustomGate"):
        self.name = name

    def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        t0 = time.perf_counter()
        # Custom gating logic...
        latency = (time.perf_counter() - t0) * 1000.0
        return Verdict(
            passed=True,
            status=DecisionStatus.PASSED,
            triggered_layer=self.name,
            total_latency_ms=latency,
            total_tokens_generated=0,
        )

    def __repr__(self) -> str:
        return f"CustomGate(name='{self.name}')"
```

Inheriting from `BaseGate` automatically gives your gate `>>` and `|` chaining support into `Cascade` pipelines!

---

## 5. Coding Standards

- **PEP 561 Typing**: All public functions and methods must have strict type annotations.
- **Helpful REPL Representations**: All gates and backends must implement `__repr__` for clear, readable debugging in interactive Python REPLs and pytest outputs.
- **Fail-Fast Input Validation**: Gate constructors must validate parameters (e.g. non-empty rubrics, valid probability thresholds `pass_threshold > fail_threshold`, positive temperatures) and raise descriptive `ValueError` exceptions immediately.
- **Preserve Existing Documentation**: Keep docstrings and citation comments intact.

---

## 6. Pull Request Process

1. Create a feature branch: `git checkout -b feature/my-new-gate`.
2. Implement your feature or fix, including new unit tests in `tests/`.
3. Verify that `pytest tests -q` passes 100% in under 1 second.
4. Ensure `git status` shows clean, targeted changes.
5. Submit your PR with a clear summary of motivation, mathematical debiasing impact, and benchmark results.
