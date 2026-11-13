from __future__ import annotations

import asyncio
import json
import math
import os
import stat
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class CalibrationResult(BaseModel):
    optimal_temperature: float
    brier_score_before: float
    brier_score_after: float
    ece_before: float
    ece_after: float
    multiclass_temperature: float | None = None
    probe_auroc: float | None = None
    median_abs_log_odds: float | None = None


KNOWN_MODEL_TEMPERATURES: dict[str, float] = {
    "qwen2.5-coder:14b": 1.05,
    "qwen2.5-coder:3b": 0.75,
    "qwen2.5-coder-3b": 0.75,
    "qwen2.5-coder:1.5b": 0.85,
    "qwen2.5-coder-1.5b": 0.85,
    "qwen2.5:14b": 1.05,
    "qwen2.5:7b": 1.05,
    "qwen2.5": 1.05,
    "gemma4:e4b": 4.00,
    "gemma4": 2.50,
    "gpt-4o-mini": 1.15,
    "llama3.2:3b": 1.00,
    "llama3.2:1b": 1.40,
    "llama3": 1.10,
}

# Sub-4B instruction models exhibit partial in-context symbol-binding attenuation
# under zero-shot binary A/B polarity inversion (Pass 1: A=PASS/B=FAIL, Pass 2: B=PASS/A=FAIL),
# shrinking the symmetric binary log-odds difference 0.5 * (f_lo + r_lo) relative to
# un-inverted K-ary ChoiceGate / ScoreGate vocabularies.
KNOWN_BINARY_MODEL_TEMPERATURES: dict[str, float] = {
    "qwen2.5-coder:1.5b": 0.06,
    "qwen2.5-coder-1.5b": 0.06,
    "qwen2.5-coder:3b": 0.40,
    "qwen2.5-coder-3b": 0.40,
}

_GENERIC_FAMILY_KEYS = frozenset({"qwen2.5", "gemma4", "llama3"})
_IGNORED_MODEL_KEYS = frozenset({"", "default", "stub"})


def _is_ignored_model_name(m_lower: str) -> bool:
    return m_lower in _IGNORED_MODEL_KEYS or m_lower.startswith(("stub://", "custom://"))

UNIVERSAL_CALIBRATION_DATASET: list[tuple[str, str, int]] = [
    # 1. Grounded RAG Pass (paraphrased factual alignment)
    (
        "Does the answer rely strictly on the provided context without extrapolating unsupported dates or regions?",
        "Context: The v2.4 migration applies to EU-West clusters starting Q3.\nAnswer: Only EU-West clusters are scheduled for the v2.4 migration in Q3.",
        1,
    ),
    # 2. Grounded RAG Fail (subtle regional extrapolation)
    (
        "Does the answer rely strictly on the provided context without extrapolating unsupported dates or regions?",
        "Context: The v2.4 migration applies to EU-West clusters starting Q3.\nAnswer: The v2.4 migration rolls out globally to all US and EU clusters in Q3.",
        0,
    ),
    # 3. Security Policy Pass (parameterized SQL)
    (
        "Does the database query use parameterized placeholders for all user inputs and avoid string interpolation?",
        'cursor.execute("SELECT id, role FROM accounts WHERE email = ? AND active = 1", (user_email,))',
        1,
    ),
    # 4. Security Policy Fail (partial SQL string interpolation)
    (
        "Does the database query use parameterized placeholders for all user inputs and avoid string interpolation?",
        'cursor.execute(f"SELECT id, role FROM accounts WHERE org = \'{org_id}\' AND email = ?", (user_email,))',
        0,
    ),
    # 5. Multi-Constraint Code Contract Pass
    (
        "Does the function include parameter and return type annotations, handle empty sequences without raising ValueError, and avoid mutating its input?",
        "def head_or_none(items: list[int]) -> int | None:\n    return items[0] if items else None",
        1,
    ),
    # 6. Multi-Constraint Code Contract Fail (satisfies 2 of 3 rules, fails mutation rule)
    (
        "Does the function include parameter and return type annotations, handle empty sequences without raising ValueError, and avoid mutating its input?",
        "def head_or_none(items: list[int]) -> int | None:\n    return items.pop(0) if items else None",
        0,
    ),
    # 7. Support Policy Pass (apology + refund confirmation + no sensitive data request)
    (
        "Does the support reply apologize for the duplicate charge, confirm a refund is issued, and avoid asking for full card numbers or passwords?",
        "I apologize for the duplicate charge on invoice #402. I have issued a full refund to your original payment method, which will appear in 3-5 business days.",
        1,
    ),
    # 8. Support Policy Fail (apologizes, but asks for CVV/password before refunding)
    (
        "Does the support reply apologize for the duplicate charge, confirm a refund is issued, and avoid asking for full card numbers or passwords?",
        "I apologize for the duplicate charge on invoice #402. Please reply with your account password and 3-digit CVV so I can issue your refund.",
        0,
    ),
]


UNIVERSAL_MULTICLASS_CALIBRATION_DATASET: list[tuple[str, str, dict[str, str], str]] = [
    (
        "Which category best describes this artifact?",
        "SELECT id, email FROM accounts WHERE org_id = ? AND active = 1",
        {
            "A": "Parameterized SQL database query",
            "B": "HTML CSS stylesheet rule",
            "C": "Customer billing refund email",
        },
        "A",
    ),
    (
        "Which severity classification matches this Python code?",
        "import subprocess\nsubprocess.Popen(user_cmd, shell=True)",
        {
            "A": "Safe read-only pure math function",
            "B": "Critical command injection vulnerability",
            "C": "Valid JSON configuration file",
        },
        "B",
    ),
    (
        "How accurately does the summary reflect the source?",
        "Source: Revenue fell 12% in Q2.\nSummary: Revenue grew 25% in Q2.",
        {
            "A": "Completely faithful paraphrase",
            "B": "Minor formatting difference only",
            "C": "Direct factual contradiction",
        },
        "C",
    ),
    (
        "Rate the engineering quality of this function implementation.",
        "def add(a: int, b: int) -> int:\n    '''Return the sum of a and b.'''\n    return a + b",
        {
            "A": "Broken syntax and wrong logic",
            "B": "Clean, typed, documented implementation",
            "C": "Unrelated network socket script",
        },
        "B",
    ),
]

_CALIBRATION_CACHE_LOCK = threading.RLock()
_FILE_STAT_CACHE: dict[str, tuple[int, int, dict[str, Any]]] = {}
_EPHEMERAL_CALIBRATION_CACHE: dict[str, dict[str, Any]] = {}
_MAX_CACHE_BYTES = 4 * 1024 * 1024


def reset_runtime_calibration_cache() -> None:
    """Clears in-memory file stat and ephemeral calibration caches."""
    with _CALIBRATION_CACHE_LOCK:
        _FILE_STAT_CACHE.clear()
        _EPHEMERAL_CALIBRATION_CACHE.clear()


def get_calibration_cache_path() -> Path:
    cache_dir = os.environ.get("LOGITGATE_CACHE_DIR")
    if cache_dir:
        return Path(cache_dir) / "calibration.json"
    return Path.home() / ".cache" / "logitgate" / "calibration.json"


def _cache_key_for_path(target_cache: Path) -> str:
    try:
        return str(target_cache.resolve())
    except (OSError, RuntimeError):
        return str(target_cache)


def _load_cached_mapping(target_cache: Path) -> dict[str, Any] | None:
    path_key = _cache_key_for_path(target_cache)
    merged: dict[str, Any] = {}
    try:
        st = target_cache.stat()
    except OSError:
        st = None

    if st is not None and stat.S_ISREG(st.st_mode) and st.st_size <= _MAX_CACHE_BYTES:
        with _CALIBRATION_CACHE_LOCK:
            hit = _FILE_STAT_CACHE.get(path_key)
            if hit is not None and hit[0] == st.st_mtime_ns and hit[1] == st.st_size:
                merged.update(hit[2])
                ephemeral = _EPHEMERAL_CALIBRATION_CACHE.get(path_key)
                if ephemeral:
                    merged.update(ephemeral)
                return merged
        try:
            with open(target_cache, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                with _CALIBRATION_CACHE_LOCK:
                    _FILE_STAT_CACHE[path_key] = (st.st_mtime_ns, st.st_size, dict(loaded))
                merged.update(loaded)
        except (OSError, ValueError, TypeError, RecursionError):
            pass

    with _CALIBRATION_CACHE_LOCK:
        ephemeral = _EPHEMERAL_CALIBRATION_CACHE.get(path_key)
        if ephemeral:
            merged.update(ephemeral)
    return merged if merged else None


def _is_valid_temp(vf: float) -> bool:
    return math.isfinite(vf) and 0.01 <= vf <= 100.0


def _extract_temp_from_entry(entry: Any, *, binary: bool | None = None) -> float | None:
    if isinstance(entry, bool):
        return None
    if isinstance(entry, dict):
        if binary is True:
            keys = ("binary_temperature", "optimal_temperature", "multiclass_temperature")
        elif binary is False:
            keys = ("multiclass_temperature", "optimal_temperature", "binary_temperature")
        else:
            keys = ("optimal_temperature", "binary_temperature", "multiclass_temperature")
        for key in keys:
            val = entry.get(key)
            if isinstance(val, bool) or val is None:
                continue
            try:
                vf = float(val)
            except (TypeError, ValueError):
                continue
            if _is_valid_temp(vf):
                if binary is False and key != "multiclass_temperature":
                    return max(0.65, vf)
                return vf
        return None
    try:
        vf = float(entry)
    except (TypeError, ValueError):
        return None
    if _is_valid_temp(vf):
        return max(0.65, vf) if binary is False else vf
    return None


def _lookup_temperature_in_mapping(
    m_lower: str,
    mapping: dict[str, Any],
    *,
    binary: bool | None = None,
    allow_family_fallback: bool = True,
) -> float | None:
    """Finds a temperature by exact case-insensitive match first, then longest-prefix substring match."""
    if _is_ignored_model_name(m_lower):
        return None
    for k, v in mapping.items():
        if not isinstance(k, str):
            continue
        k_low = k.strip().lower()
        if not k_low or k_low in _IGNORED_MODEL_KEYS:
            continue
        if k_low == m_lower:
            vf = _extract_temp_from_entry(v, binary=binary)
            if vf is not None:
                return vf
    for k, v in sorted(mapping.items(), key=lambda kv: len(str(kv[0])), reverse=True):
        if not isinstance(k, str):
            continue
        k_low = k.strip().lower()
        if not k_low or k_low in _IGNORED_MODEL_KEYS or len(k_low) < 3:
            continue
        if not allow_family_fallback and k_low in _GENERIC_FAMILY_KEYS:
            continue
        if k_low in m_lower:
            vf = _extract_temp_from_entry(v, binary=binary)
            if vf is not None:
                return vf
    return None


def has_known_or_cached_temperature(
    model_name: str | None,
    cache_path: Path | None = None,
    *,
    binary: bool | None = None,
) -> bool:
    """Returns True when ``model_name`` matches a cached profile or size-qualified built-in entry.

    Args:
        model_name: Model identifier string to look up.
        cache_path: Optional override path for the local JSON calibration cache.
        binary: When ``True``, check binary-specific temperatures first; when ``False``, prefer
            multiclass temperatures.

    Returns:
        ``True`` if an explicit cached profile or built-in entry matches without falling back
        to catch-all family keys.
    """
    if not model_name:
        return False
    m_lower = model_name.strip().lower()
    if _is_ignored_model_name(m_lower):
        return False
    target_cache = cache_path or get_calibration_cache_path()
    cached = _load_cached_mapping(target_cache)
    if cached is not None and _lookup_temperature_in_mapping(m_lower, cached, binary=binary, allow_family_fallback=True) is not None:
        return True
    if binary is True and _lookup_temperature_in_mapping(
        m_lower, KNOWN_BINARY_MODEL_TEMPERATURES, allow_family_fallback=False
    ) is not None:
        return True
    return _lookup_temperature_in_mapping(m_lower, KNOWN_MODEL_TEMPERATURES, allow_family_fallback=False) is not None


def get_model_temperature(
    model_name: str | None,
    cache_path: Path | None = None,
    *,
    binary: bool | None = None,
) -> float:
    """Retrieves calibrated temperature for a model checking cache, known models, then default 1.0.

    Args:
        model_name: Model identifier string (e.g. ``"qwen2.5-coder:3b"`` or a BYO model name).
        cache_path: Optional path to a custom JSON calibration cache file.
        binary: When ``True``, prefer binary A/B permutation temperature overrides; when
            ``False``, prefer multiclass/score temperatures.

    Returns:
        Positive scaling temperature float, defaulting to ``1.0`` when unknown.
    """
    if not model_name:
        return 1.0

    m_lower = model_name.strip().lower()
    if _is_ignored_model_name(m_lower):
        return 1.0

    # 1. Local calibration cache file (or in-memory ephemeral fallback)
    target_cache = cache_path or get_calibration_cache_path()
    cached = _load_cached_mapping(target_cache)
    if cached is not None:
        found = _lookup_temperature_in_mapping(m_lower, cached, binary=binary, allow_family_fallback=True)
        if found is not None:
            return found

    # 2. Zero-shot binary A/B overrides for compact models with symbol-binding attenuation
    if binary is True:
        found_binary = _lookup_temperature_in_mapping(
            m_lower, KNOWN_BINARY_MODEL_TEMPERATURES, allow_family_fallback=False
        )
        if found_binary is not None:
            return found_binary

    # 3. KNOWN_MODEL_TEMPERATURES
    found_known = _lookup_temperature_in_mapping(
        m_lower, KNOWN_MODEL_TEMPERATURES, allow_family_fallback=True
    )
    if found_known is not None:
        return found_known

    # 4. Default
    return 1.0


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    mid = len(s) // 2
    if len(s) % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def _compute_binary_auroc(mean_los_and_labels: Sequence[tuple[float, int]]) -> float | None:
    pos = [z for z, y in mean_los_and_labels if y == 1]
    neg = [z for z, y in mean_los_and_labels if y == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for zp in pos:
        for zn in neg:
            if zp > zn + 1e-12:
                wins += 1.0
            elif abs(zp - zn) <= 1e-12:
                wins += 0.5
    return wins / float(len(pos) * len(neg))


def _debiased_multiclass_accum(
    top_fwd: list[dict[str, Any]],
    top_perm: list[dict[str, Any]],
    keys: Sequence[str],
) -> dict[str, float]:
    from logitgate.debiasing.softmax import extract_token_logprob

    n = len(keys)
    if n == 0:
        return {}
    shift = max(1, n // 2)
    perms = [list(range(n)), [(i + shift) % n for i in range(n)]]
    accum = {k: 0.0 for k in keys}
    uniform_lp = -math.log(float(n))
    for top, perm in zip((top_fwd, top_perm), perms):
        pass_raw = {keys[i]: extract_token_logprob(top, keys[perm[i]]) for i in range(n)}
        finite_vals = [v for v in pass_raw.values() if math.isfinite(v)]
        if not finite_vals:
            for k in keys:
                accum[k] += 0.5 * uniform_lp
            continue
        m_pass = max(finite_vals)
        sum_exp = sum(math.exp(v - m_pass) for v in pass_raw.values() if math.isfinite(v))
        lse_pass = m_pass + math.log(sum_exp)
        for k in keys:
            val = pass_raw[k]
            accum[k] += 0.5 * ((val - lse_pass) if math.isfinite(val) else -50.0)
    return accum


def _fit_multiclass_temperature(
    mc_samples: Sequence[tuple[dict[str, float], str]],
    *,
    binary_auroc: float | None = None,
    fallback_temp: float = 1.0,
    min_temp: float = 0.65,
    max_temp: float = 10.0,
) -> float:
    from logitgate.debiasing.softmax import softmax_probabilities

    if min_temp <= 0.0 or max_temp <= min_temp:
        min_temp = 0.65
        max_temp = 10.0
    fallback_clamped = min(max_temp, max(min_temp, float(fallback_temp)))
    if not mc_samples:
        return round(fallback_clamped, 4)

    gaps: list[float] = []
    correct = 0
    for accum, target_key in mc_samples:
        best_key = max(accum.items(), key=lambda kv: kv[1])[0]
        if best_key == target_key:
            correct += 1
        vals = sorted(accum.values(), reverse=True)
        gaps.append(vals[0] - vals[1] if len(vals) >= 2 else 0.0)

    mc_acc = correct / float(len(mc_samples))
    max_gap = max(gaps) if gaps else 0.0
    if max_gap <= 1e-6:
        return round(min(max_temp, max(min_temp, 1.0)), 4)
    if mc_acc < 0.50:
        return round(min(max_temp, max(min_temp, max(1.0, fallback_clamped))), 4)

    effective_min = (
        min(max_temp, max(min_temp, 1.0))
        if (binary_auroc is not None and binary_auroc < 0.70 and len(mc_samples) < 16)
        else min_temp
    )
    if max_temp <= effective_min * (1.0 + 1e-12):
        return round(effective_min, 4)

    med_gap = _median(gaps)
    t_prior = min(max_temp, max(effective_min, max(0.80, med_gap / 3.6)))
    reg_weight = 0.08 / max(1.0, len(mc_samples) / 4.0)

    def objective(temp: float) -> float:
        total_brier = 0.0
        for accum, target_key in mc_samples:
            probs = softmax_probabilities(accum, temp)
            total_brier += sum(
                (probs[k] - (1.0 if k == target_key else 0.0)) ** 2 for k in accum
            )
        mean_brier = total_brier / float(len(mc_samples))
        return mean_brier + reg_weight * (math.log(max(temp, 1e-6) / max(t_prior, 1e-6)) ** 2)

    steps = 160
    ratio = max_temp / effective_min
    best_temp = t_prior
    best_obj = objective(t_prior)
    for step in range(steps + 1):
        temp = effective_min * (ratio ** (step / steps))
        obj_val = objective(temp)
        if obj_val < best_obj - 1e-12:
            best_obj = obj_val
            best_temp = temp
    best_temp = min(max_temp, max(min_temp, best_temp))
    return round(best_temp, 4)


def _persist_calibration_profile(
    effective_model: str,
    result: CalibrationResult,
    *,
    cache_path: Path | None = None,
    strict_cache_io: bool = True,
) -> None:
    import tempfile

    target_path = cache_path or get_calibration_cache_path()
    path_key = _cache_key_for_path(target_path)
    profile: dict[str, float] = {
        "optimal_temperature": result.optimal_temperature,
        "binary_temperature": result.optimal_temperature,
    }
    if result.multiclass_temperature is not None:
        profile["multiclass_temperature"] = result.multiclass_temperature
    if result.probe_auroc is not None:
        profile["probe_auroc"] = result.probe_auroc
    if result.median_abs_log_odds is not None:
        profile["median_abs_log_odds"] = result.median_abs_log_odds

    try:
        with _CALIBRATION_CACHE_LOCK:
            if target_path.is_symlink():
                raise OSError(f"Refusing to write calibration cache through symlink: {target_path}")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            cache_data: dict[str, Any] = {}
            try:
                st_existing = target_path.stat()
            except OSError:
                st_existing = None
            if (
                st_existing is not None
                and stat.S_ISREG(st_existing.st_mode)
                and st_existing.st_size <= _MAX_CACHE_BYTES
            ):
                try:
                    with open(target_path, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                    if isinstance(loaded, dict):
                        cache_data = loaded
                except (OSError, ValueError, TypeError, RecursionError):
                    cache_data = {}
            ephemeral = _EPHEMERAL_CALIBRATION_CACHE.get(path_key)
            if ephemeral:
                cache_data.update(ephemeral)
            cache_data[effective_model] = profile
            fd, tmp_path = tempfile.mkstemp(dir=target_path.parent, suffix=".tmp")
            fd_owned = True
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    fd_owned = False
                    json.dump(cache_data, f, indent=2, allow_nan=False)
                os.replace(tmp_path, target_path)
            except BaseException:
                if fd_owned:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
            try:
                st = target_path.stat()
                _FILE_STAT_CACHE[path_key] = (st.st_mtime_ns, st.st_size, dict(cache_data))
                _EPHEMERAL_CALIBRATION_CACHE.pop(path_key, None)
            except OSError:
                pass
    except OSError:
        with _CALIBRATION_CACHE_LOCK:
            _EPHEMERAL_CALIBRATION_CACHE.setdefault(path_key, {})[effective_model] = profile
        if strict_cache_io:
            raise


def _multiclass_probe_prompts(
    question: str,
    candidate: str,
    options: dict[str, str],
) -> tuple[str, str, list[str]]:
    from logitgate.debiasing.inversion import compile_choice_prompt

    keys = list(options.keys())
    values = list(options.values())
    n = len(keys)
    shift = max(1, n // 2)
    perm = [(i + shift) % n for i in range(n)]
    permuted_opts = {keys[key_idx]: values[perm.index(key_idx)] for key_idx in range(n)}
    p_fwd = compile_choice_prompt(question=question, candidate=candidate, options=options, query_first=True)
    p_perm = compile_choice_prompt(question=question, candidate=candidate, options=permuted_opts, query_first=True)
    return p_fwd, p_perm, keys


def auto_calibrate_backend(
    backend: Any,
    model_key: str | None = None,
    save_cache: bool = True,
    cache_path: Path | None = None,
    *,
    strict_cache_io: bool = True,
) -> CalibrationResult:
    """Runs built-in micro-calibration probes, fits binary and multiclass temperatures, and caches them.

    Args:
        backend: Live `LogitBackend` instance supporting ``query_top_logprobs``.
        model_key: Optional cache key override (defaults to ``backend.model`` or ``"default"``).
        save_cache: When ``True``, persists fitted binary and multiclass temperatures to disk.
        cache_path: Optional override path for the calibration cache JSON file.
        strict_cache_io: When ``True``, raises ``OSError`` if persisting to disk fails; when
            ``False``, falls back to the in-memory ephemeral cache for the current process.

    Returns:
        `CalibrationResult` containing ``optimal_temperature``, ``multiclass_temperature``,
        Brier score, ECE, and probe AUROC diagnostics.

    Raises:
        BackendError: If the backend fails to return finite token log-probabilities.
    """
    from logitgate.debiasing.inversion import compile_binary_prompt
    from logitgate.debiasing.softmax import compute_binary_log_odds, has_finite_logprobs
    from logitgate.exceptions import BackendError

    effective_model = model_key or getattr(backend, "model", None) or "default"
    samples: list[tuple[float, float, int]] = []

    for rubric, candidate, y in UNIVERSAL_CALIBRATION_DATASET:
        p1 = compile_binary_prompt(
            rubric=rubric,
            candidate=candidate,
            pass_token="A",
            fail_token="B",
            query_first=True,
        )
        top1 = backend.query_top_logprobs(p1, top_k=10, allowed_tokens=["A", "B"])
        if not has_finite_logprobs(top1):
            raise BackendError(
                f"Backend {type(backend).__name__} returned no token log-probabilities during calibration."
            )
        f_lo, _ = compute_binary_log_odds(top1, pass_token="A", fail_token="B")

        p2 = compile_binary_prompt(
            rubric=rubric,
            candidate=candidate,
            pass_token="B",
            fail_token="A",
            query_first=True,
        )
        top2 = backend.query_top_logprobs(p2, top_k=10, allowed_tokens=["A", "B"])
        if not has_finite_logprobs(top2):
            raise BackendError(
                f"Backend {type(backend).__name__} returned no token log-probabilities during calibration."
            )
        r_lo, _ = compute_binary_log_odds(top2, pass_token="B", fail_token="A")

        samples.append((f_lo, r_lo, y))

    result = fit_calibration(samples)

    mc_temp: float | None = None
    try:
        mc_samples: list[tuple[dict[str, float], str]] = []
        for question, candidate, options, target_key in UNIVERSAL_MULTICLASS_CALIBRATION_DATASET:
            p_fwd, p_perm, keys = _multiclass_probe_prompts(question, candidate, options)
            top_f = backend.query_top_logprobs(p_fwd, top_k=10, allowed_tokens=keys)
            top_p = backend.query_top_logprobs(p_perm, top_k=10, allowed_tokens=keys)
            if not has_finite_logprobs(top_f) or not has_finite_logprobs(top_p):
                mc_samples = []
                break
            mc_samples.append((_debiased_multiclass_accum(top_f, top_p, keys), target_key))
        if mc_samples:
            mc_temp = _fit_multiclass_temperature(
                mc_samples,
                binary_auroc=result.probe_auroc,
                fallback_temp=result.optimal_temperature,
            )
    except (BackendError, ValueError, KeyError, RuntimeError):
        mc_temp = None

    if mc_temp is not None:
        result = result.model_copy(update={"multiclass_temperature": mc_temp})

    if save_cache:
        _persist_calibration_profile(
            effective_model,
            result,
            cache_path=cache_path,
            strict_cache_io=strict_cache_io,
        )

    return result


async def aauto_calibrate_backend(
    backend: Any,
    model_key: str | None = None,
    save_cache: bool = True,
    cache_path: Path | None = None,
    *,
    strict_cache_io: bool = True,
) -> CalibrationResult:
    """Async counterpart to `auto_calibrate_backend`.

    Args:
        backend: Live `LogitBackend` instance supporting ``aquery_top_logprobs``.
        model_key: Optional cache key override (defaults to ``backend.model`` or ``"default"``).
        save_cache: When ``True``, persists fitted binary and multiclass temperatures to disk.
        cache_path: Optional override path for the calibration cache JSON file.
        strict_cache_io: When ``True``, raises ``OSError`` if persisting to disk fails; when
            ``False``, falls back to the in-memory ephemeral cache for the current process.

    Returns:
        `CalibrationResult` containing fitted temperatures and calibration metrics.

    Raises:
        BackendError: If the backend fails to return finite token log-probabilities.
    """
    from logitgate.debiasing.inversion import compile_binary_prompt
    from logitgate.debiasing.softmax import compute_binary_log_odds, has_finite_logprobs
    from logitgate.exceptions import BackendError

    effective_model = model_key or getattr(backend, "model", None) or "default"
    samples: list[tuple[float, float, int]] = []

    for rubric, candidate, y in UNIVERSAL_CALIBRATION_DATASET:
        p1 = compile_binary_prompt(
            rubric=rubric,
            candidate=candidate,
            pass_token="A",
            fail_token="B",
            query_first=True,
        )
        p2 = compile_binary_prompt(
            rubric=rubric,
            candidate=candidate,
            pass_token="B",
            fail_token="A",
            query_first=True,
        )
        top1, top2 = await asyncio.gather(
            backend.aquery_top_logprobs(p1, top_k=10, allowed_tokens=["A", "B"]),
            backend.aquery_top_logprobs(p2, top_k=10, allowed_tokens=["A", "B"]),
        )
        if not has_finite_logprobs(top1) or not has_finite_logprobs(top2):
            raise BackendError(
                f"Backend {type(backend).__name__} returned no token log-probabilities during calibration."
            )
        f_lo, _ = compute_binary_log_odds(top1, pass_token="A", fail_token="B")
        r_lo, _ = compute_binary_log_odds(top2, pass_token="B", fail_token="A")
        samples.append((f_lo, r_lo, y))

    result = fit_calibration(samples)

    mc_temp: float | None = None
    try:
        mc_samples: list[tuple[dict[str, float], str]] = []
        for question, candidate, options, target_key in UNIVERSAL_MULTICLASS_CALIBRATION_DATASET:
            p_fwd, p_perm, keys = _multiclass_probe_prompts(question, candidate, options)
            top_f, top_p = await asyncio.gather(
                backend.aquery_top_logprobs(p_fwd, top_k=10, allowed_tokens=keys),
                backend.aquery_top_logprobs(p_perm, top_k=10, allowed_tokens=keys),
            )
            if not has_finite_logprobs(top_f) or not has_finite_logprobs(top_p):
                mc_samples = []
                break
            mc_samples.append((_debiased_multiclass_accum(top_f, top_p, keys), target_key))
        if mc_samples:
            mc_temp = _fit_multiclass_temperature(
                mc_samples,
                binary_auroc=result.probe_auroc,
                fallback_temp=result.optimal_temperature,
            )
    except (BackendError, ValueError, KeyError, RuntimeError):
        mc_temp = None

    if mc_temp is not None:
        result = result.model_copy(update={"multiclass_temperature": mc_temp})

    if save_cache:
        _persist_calibration_profile(
            effective_model,
            result,
            cache_path=cache_path,
            strict_cache_io=strict_cache_io,
        )

    return result


def combine_symmetric_log_odds(
    forward_log_odds: float,
    reverse_log_odds: float,
    logit_temperature: float = 1.0,
) -> float:
    """
    Combines forward and reverse log-odds to cancel token and positional bias.
    mean_log_odds = 0.5 * (forward_log_odds / t + reverse_log_odds / t)
    Returns 1.0 / (1.0 + exp(-mean_log_odds)).
    """
    from logitgate.debiasing.softmax import clamped_sigmoid

    t = (
        max(float(logit_temperature), 1e-6)
        if (not isinstance(logit_temperature, bool) and math.isfinite(logit_temperature) and logit_temperature > 0.0)
        else 1.0
    )
    mean_log_odds = 0.5 * (float(forward_log_odds) / t) + 0.5 * (float(reverse_log_odds) / t)
    if math.isnan(mean_log_odds):
        return 0.5
    return clamped_sigmoid(mean_log_odds)


def compute_shannon_entropy(p_pass: float, eps: float = 1e-12) -> float:
    """Computes normalized binary Shannon entropy in [0.0, 1.0] (bits)."""
    if math.isnan(p_pass):
        return 1.0
    eps_c = min(max(float(eps), 1e-15), 0.499999)
    p_c = min(max(float(p_pass), eps_c), 1.0 - eps_c)
    return -(p_c * math.log2(p_c) + (1.0 - p_c) * math.log2(1.0 - p_c))


@dataclass(frozen=True, slots=True)
class SurfaceTokenMap:
    """Maps canonical binary poles to all single-token surface form IDs."""

    positive_token_ids: tuple[int, ...]
    negative_token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        for seq_name, seq_vals in (
            ("positive_token_ids", self.positive_token_ids),
            ("negative_token_ids", self.negative_token_ids),
        ):
            for x in seq_vals:
                if isinstance(x, bool) or not isinstance(x, int) or x < 0:
                    raise ValueError(f"{seq_name} must contain non-negative integers, got {x!r}")
        pos_norm = tuple(sorted({int(x) for x in self.positive_token_ids}))
        neg_norm = tuple(sorted({int(x) for x in self.negative_token_ids}))
        object.__setattr__(self, "positive_token_ids", pos_norm)
        object.__setattr__(self, "negative_token_ids", neg_norm)
        if not pos_norm or not neg_norm:
            raise ValueError("positive_token_ids and negative_token_ids must not be empty.")
        overlap = set(pos_norm) & set(neg_norm)
        if overlap:
            raise ValueError(
                f"Positive and negative token IDs must be disjoint, got overlapping IDs: {sorted(overlap)}"
            )

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer: Any,
        pos_label: str = "A",
        neg_label: str = "B",
    ) -> SurfaceTokenMap:
        def _collect_ids(label: str) -> tuple[int, ...]:
            variants = {
                label,
                label.lower(),
                label.upper(),
                label.capitalize(),
                f" {label}",
                f" {label.lower()}",
                f" {label.upper()}",
                f" {label.capitalize()}",
            }
            token_ids: set[int] = set()
            for v in variants:
                try:
                    encoded = tokenizer.encode(v, add_special_tokens=False)
                except TypeError:
                    encoded = tokenizer.encode(v)
                if len(encoded) == 1:
                    token_ids.add(int(encoded[0]))
            if not token_ids:
                raise ValueError(f"Label '{label}' does not map to any single-token ID.")
            return tuple(sorted(token_ids))

        return cls(
            positive_token_ids=_collect_ids(pos_label),
            negative_token_ids=_collect_ids(neg_label),
        )

    def to_logit_bias(self, bias: float = 100) -> dict[str, int | float]:
        """Builds an OpenAI/vLLM/llama.cpp `logit_bias` dictionary for all surface token IDs."""
        all_ids = sorted(set(self.positive_token_ids) | set(self.negative_token_ids))
        return {str(tid): bias for tid in all_ids}


def marginalize_pole_logprob(
    logits: Any,
    pos_ids: Sequence[int],
    neg_ids: Sequence[int],
    temperature: float = 1.0,
) -> tuple[float, float]:
    """Computes logsumexp over all surface-form tokens for each pole, normalized

    strictly over the union of valid verdict tokens (eliminating leakage to irrelevant vocab).

    Args:
        logits: 1D Tensor or sequence of shape [vocab_size] from the final prefill token.
        pos_ids: Token IDs corresponding to the first pole (e.g. token 'A').
        neg_ids: Token IDs corresponding to the second pole (e.g. token 'B').
        temperature: Calibration temperature T > 0.

    Returns:
        Tuple of (log_prob_pos, log_prob_neg)
    """
    if isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("Temperature must be a finite strictly positive float.")
    if not pos_ids or not neg_ids:
        raise ValueError("pos_ids and neg_ids must not be empty.")
    for seq_name, seq_vals in (("pos_ids", pos_ids), ("neg_ids", neg_ids)):
        for idx_val in seq_vals:
            if isinstance(idx_val, bool) or not isinstance(idx_val, int) or idx_val < 0:
                raise ValueError(f"{seq_name} must contain non-negative integer token IDs, got {idx_val!r}")

    lse_pos_raw: float
    lse_neg_raw: float
    used_torch = False
    try:
        import torch

        if isinstance(logits, torch.Tensor):
            pos_tensor = logits[list(pos_ids)].float()
            neg_tensor = logits[list(neg_ids)].float()
            lse_pos_raw = float(torch.logsumexp(pos_tensor, dim=0).item())
            lse_neg_raw = float(torch.logsumexp(neg_tensor, dim=0).item())
            used_torch = True
    except ImportError:
        pass

    if not used_torch:
        # Sequence fallback (marginalize surface-form tokens first, then apply temperature scaling)
        pos_vals = [float(logits[i]) for i in pos_ids]
        neg_vals = [float(logits[i]) for i in neg_ids]

        def _lse(vals: Sequence[float]) -> float:
            if any(math.isnan(v) for v in vals):
                return float("nan")
            m = max(vals)
            if math.isinf(m):
                return m
            return m + math.log(sum(math.exp(v - m) for v in vals))

        lse_pos_raw = _lse(pos_vals)
        lse_neg_raw = _lse(neg_vals)

    lse_pos_f = lse_pos_raw / temperature
    lse_neg_f = lse_neg_raw / temperature

    if math.isnan(lse_pos_f) or math.isnan(lse_neg_f):
        return -math.log(2.0), -math.log(2.0)
    if math.isinf(lse_pos_f) and math.isinf(lse_neg_f):
        if (lse_pos_f > 0) == (lse_neg_f > 0):
            return -math.log(2.0), -math.log(2.0)
        return (0.0, -float("inf")) if lse_pos_f > 0 else (-float("inf"), 0.0)
    if math.isinf(lse_pos_f):
        return (0.0, -float("inf")) if lse_pos_f > 0 else (-float("inf"), 0.0)
    if math.isinf(lse_neg_f):
        return (-float("inf"), 0.0) if lse_neg_f > 0 else (0.0, -float("inf"))

    m_tot = max(lse_pos_f, lse_neg_f)
    lse_total_f = m_tot + math.log(math.exp(lse_pos_f - m_tot) + math.exp(lse_neg_f - m_tot))
    return lse_pos_f - lse_total_f, lse_neg_f - lse_total_f


@dataclass(frozen=True, slots=True)
class DebiasedLogitResult:
    """Output of one symmetric (forward + permuted) logit read; see `compute_symmetric_debiased_verdict`."""

    calibrated_prob_pass: float
    shannon_entropy: float
    log_odds_pass: float
    positional_bias_magnitude: float
    pass_forward_prob: float
    pass_permuted_prob: float


def compute_symmetric_debiased_verdict(
    logits_pass_first: Any,
    logits_fail_first: Any,
    token_map: SurfaceTokenMap,
    temperature: float = 1.0,
) -> DebiasedLogitResult:
    """Executes Symmetric Label Permutation debiasing in log-odds space.

    Pass 1 (Forward):  Token 'A' = PASS, Token 'B' = FAIL
    Pass 2 (Permuted): Token 'A' = FAIL, Token 'B' = PASS

    By binding the physical tokens ('A', 'B') to swapped semantic meanings across
    Pass 1 and Pass 2, subtracting the log-odds cancels both token-identity prior
    (preference for 'A' over 'B') and slot-order bias.
    """
    # Pass 1: A -> Pass, B -> Fail
    lp_A1, lp_B1 = marginalize_pole_logprob(
        logits_pass_first,
        token_map.positive_token_ids,
        token_map.negative_token_ids,
        temperature=temperature,
    )
    raw_delta_1 = lp_A1 - lp_B1
    delta_1 = max(-100.0, min(100.0, raw_delta_1)) if math.isinf(raw_delta_1) else raw_delta_1

    # Pass 2: A -> Fail, B -> Pass
    lp_A2, lp_B2 = marginalize_pole_logprob(
        logits_fail_first,
        token_map.positive_token_ids,
        token_map.negative_token_ids,
        temperature=temperature,
    )
    raw_delta_2 = lp_B2 - lp_A2
    delta_2 = max(-100.0, min(100.0, raw_delta_2)) if math.isinf(raw_delta_2) else raw_delta_2

    from logitgate.debiasing.softmax import clamped_sigmoid

    unbiased_log_odds = max(-100.0, min(100.0, 0.5 * (delta_1 + delta_2)))
    positional_bias = min(100.0, 0.5 * abs(delta_1 - delta_2))
    p_pass = clamped_sigmoid(unbiased_log_odds)

    entropy = compute_shannon_entropy(p_pass)

    return DebiasedLogitResult(
        calibrated_prob_pass=p_pass,
        shannon_entropy=entropy,
        log_odds_pass=unbiased_log_odds,
        positional_bias_magnitude=positional_bias,
        pass_forward_prob=math.exp(lp_A1),
        pass_permuted_prob=math.exp(lp_B2),
    )


def fit_calibration(
    samples: Sequence[tuple[float, float, int]],
    min_temp: float = 0.04,
    max_temp: float = 10.0,
    num_bins: int = 10,
) -> CalibrationResult:
    """Fits optimal logit temperature scaling on symmetric log-odds pairs.

    Minimizes Brier score with AUROC-aware MAP log-temperature regularization on
    ``(forward_log_odds, reverse_log_odds, y)`` tuples and computes Expected
    Calibration Error (ECE).

    Args:
        samples: Sequence of ``(forward_log_odds, reverse_log_odds, label)`` tuples with
            binary ground-truth ``label`` in ``{0, 1}``.
        min_temp: Lower search bound (must be strictly positive and less than ``max_temp``).
        max_temp: Upper search bound.
        num_bins: Number of reliability bins used for ECE calculation.

    Returns:
        `CalibrationResult` containing fitted ``optimal_temperature``, Brier before/after,
        ECE before/after, ``probe_auroc``, and ``median_abs_log_odds``.

    Raises:
        CalibrationError: If bounds, ``num_bins``, or sample values/labels are invalid.
    """
    import warnings

    from logitgate.exceptions import CalibrationError, CalibrationWarning

    if (
        isinstance(min_temp, bool)
        or isinstance(max_temp, bool)
        or not math.isfinite(min_temp)
        or not math.isfinite(max_temp)
        or min_temp <= 0.0
        or max_temp <= min_temp
    ):
        raise CalibrationError(
            f"Expected finite 0 < min_temp < max_temp, got min_temp={min_temp}, max_temp={max_temp}"
        )
    if isinstance(num_bins, bool) or not isinstance(num_bins, int) or num_bins < 1:
        raise CalibrationError(f"num_bins must be an integer >= 1, got {num_bins}")

    if not samples:
        warnings.warn(
            "fit_calibration received an empty sample sequence; returning default optimal_temperature=1.0.",
            CalibrationWarning,
            stacklevel=2,
        )
        return CalibrationResult(
            optimal_temperature=1.0,
            brier_score_before=0.0,
            brier_score_after=0.0,
            ece_before=0.0,
            ece_after=0.0,
        )

    labels_seen: set[int] = set()
    for f_lo, r_lo, y in samples:
        if isinstance(f_lo, bool) or isinstance(r_lo, bool):
            raise CalibrationError(
                f"Calibration samples must contain finite float log-odds, got ({f_lo!r}, {r_lo!r})"
            )
        try:
            f_val = float(f_lo)
            r_val = float(r_lo)
        except (TypeError, ValueError) as exc:
            raise CalibrationError(
                f"Calibration samples must contain finite log-odds, got ({f_lo!r}, {r_lo!r})"
            ) from exc
        if not math.isfinite(f_val) or not math.isfinite(r_val):
            raise CalibrationError(
                f"Calibration samples must contain finite log-odds, got ({f_lo}, {r_lo})"
            )
        if isinstance(y, bool) or y not in (0, 1):
            raise CalibrationError(f"Calibration sample label must be 0 or 1, got {y!r}")
        labels_seen.add(int(y))

    if len(labels_seen) < 2:
        warnings.warn(
            "fit_calibration received samples from only a single class; temperature scaling is degenerate.",
            CalibrationWarning,
            stacklevel=2,
        )
    elif len(samples) < 4:
        warnings.warn(
            "fit_calibration received fewer than 4 samples; temperature estimate has high variance.",
            CalibrationWarning,
            stacklevel=2,
        )

    mean_los: list[tuple[float, int]] = [
        (0.5 * (float(f_lo) + float(r_lo)), int(y)) for f_lo, r_lo, y in samples
    ]
    abs_los = [abs(z) for z, _ in mean_los]
    median_abs_lo = _median(abs_los)
    max_abs_lo = max(abs_los) if abs_los else 0.0
    if max_abs_lo <= 1e-9:
        warnings.warn(
            "fit_calibration received all zero symmetric log-odds; temperature cannot change predictions.",
            CalibrationWarning,
            stacklevel=2,
        )
    probe_auroc = _compute_binary_auroc(mean_los) if len(labels_seen) >= 2 else None

    effective_min = min_temp
    t_prior = min(max_temp, max(effective_min, 1.0))
    base_reg = 1e-2
    if len(labels_seen) >= 2 and max_abs_lo > 1e-9 and probe_auroc is not None:
        if probe_auroc < 0.70 and len(samples) < 32:
            # Weak rank discrimination on small probe sets: prevent T < 1.0 sharpening so uncertain items escalate.
            if min_temp < 1.0 < max_temp:
                effective_min = max(min_temp, 1.0)
            t_prior = max(effective_min, max(1.5, median_abs_lo / 2.5))
        elif median_abs_lo < 2.2:
            # Good rank discrimination with attenuated symmetric log-odds (e.g. sub-4B symbol binding).
            t_prior = max(min_temp, min(1.0, median_abs_lo / 2.2))
        elif median_abs_lo > 4.0:
            # High-magnitude log-odds (e.g. Gemma 4): normalize median scale toward ~2.8 logits.
            t_prior = max(1.0, min(max_temp, median_abs_lo / 2.8))
            base_reg = 0.04
        t_prior = min(max_temp, max(effective_min, t_prior))

    def calc_prob(f_lo: float, r_lo: float, temp: float) -> float:
        return combine_symmetric_log_odds(f_lo, r_lo, logit_temperature=temp)

    def brier_score(temp: float) -> float:
        total = sum((calc_prob(f, r, temp) - float(y)) ** 2 for f, r, y in samples)
        return total / len(samples)

    # Scale- and AUROC-aware log-temperature MAP regularization prevents T -> min_temp collapse
    # on small 100%-separable micro-probes while decaying O(1/N) so large user datasets converge
    # cleanly to the empirical Brier optimum.
    reg_weight = base_reg / max(1.0, len(samples) / 8.0)

    def objective(temp: float) -> float:
        return brier_score(temp) + reg_weight * (math.log(max(temp, 1e-6) / max(t_prior, 1e-6)) ** 2)

    def compute_ece(temp: float) -> float:
        n = len(samples)
        bins: list[list[tuple[float, float]]] = [[] for _ in range(num_bins)]
        for f, r, y in samples:
            p = calc_prob(f, r, temp)
            idx = min(int(p * num_bins), num_bins - 1)
            bins[idx].append((p, float(y)))

        ece = 0.0
        for b in bins:
            if not b:
                continue
            b_size = len(b)
            avg_conf = sum(p for p, _ in b) / b_size
            avg_acc = sum(y for _, y in b) / b_size
            ece += (b_size / n) * abs(avg_acc - avg_conf)
        return ece

    # 1. Log-spaced grid search for coarse minimum across orders of magnitude
    brier_before = brier_score(1.0)
    best_temp = 1.0 if effective_min <= 1.0 <= max_temp else effective_min
    best_obj = objective(best_temp)
    steps = 200
    ratio = max_temp / effective_min
    best_step: int | None = None
    for step in range(steps + 1):
        temp = effective_min * (ratio ** (step / steps))
        obj_val = objective(temp)
        if obj_val < best_obj - 1e-12:
            best_obj = obj_val
            best_temp = temp
            best_step = step

    # 2. Golden-section refinement around coarse minimum
    if best_step is None:
        step_factor = ratio ** (1.0 / steps)
        a = max(effective_min, best_temp / step_factor)
        b = min(max_temp, best_temp * step_factor)
    else:
        a = effective_min * (ratio ** (max(0, best_step - 1) / steps))
        b = effective_min * (ratio ** (min(steps, best_step + 1) / steps))
    phi = (math.sqrt(5) - 1) / 2
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc = objective(c)
    fd = objective(d)

    for _ in range(30):
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - phi * (b - a)
            fc = objective(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + phi * (b - a)
            fd = objective(d)

    optimal_temp = min(max_temp, max(effective_min, (a + b) / 2.0))
    brier_after = brier_score(optimal_temp)
    if (
        (
            max_abs_lo <= 1e-9
            or len(labels_seen) < 2
            or abs(t_prior - 1.0) <= 1e-9
            or (len(samples) >= 32 and brier_after > brier_before + 1e-12)
        )
        and brier_after >= brier_before - 1e-12
        and effective_min <= 1.0 <= max_temp
    ):
        optimal_temp = 1.0
        brier_after = brier_before
    optimal_temp = min(max_temp, max(min_temp, optimal_temp))
    ece_before = compute_ece(1.0)
    ece_after = compute_ece(optimal_temp)

    return CalibrationResult(
        optimal_temperature=round(optimal_temp, 4),
        brier_score_before=round(brier_before, 6),
        brier_score_after=round(brier_after, 6),
        ece_before=round(ece_before, 6),
        ece_after=round(ece_after, 6),
        probe_auroc=round(probe_auroc, 4) if probe_auroc is not None else None,
        median_abs_log_odds=round(median_abs_lo, 4),
    )
