"""Deterministic content-addressed `VerdictCache` with bounded LRU memory and optional SQLite persistence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import stat
import threading
import time
import warnings
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel
from typing_extensions import Self

from logitgate.backends.base import (
    GenerativeBackend,
    LogitBackend,
    StubBackend,
    infer_backend_spec,
)
from logitgate.exceptions import CacheWarning, ConfigurationError
from logitgate.models import DecisionStatus, Verdict

if TYPE_CHECKING:
    from logitgate.gates.base import BaseGate

_CACHE_KEY_PREFIX = "lgcache:v1:"
_MAX_SQLITE_DB_BYTES = 256 * 1024 * 1024  # 256 MiB safety cap on existing SQLite file
_NO_OVERRIDE = object()


@dataclass(frozen=True, slots=True)
class CacheStats:
    """Telemetry snapshot of `VerdictCache` lookups, evictions, and saved compute."""

    hits: int
    misses: int
    puts: int
    evictions: int
    memory_entries: int
    disk_entries: int
    saved_latency_ms: float
    saved_tokens: int

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return round(self.hits / total, 4) if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "puts": self.puts,
            "evictions": self.evictions,
            "hit_rate": self.hit_rate,
            "memory_entries": self.memory_entries,
            "disk_entries": self.disk_entries,
            "saved_latency_ms": round(self.saved_latency_ms, 2),
            "saved_tokens": self.saved_tokens,
        }


@dataclass(slots=True)
class _CacheEntry:
    verdict: Verdict
    created_at: float
    expires_at: float | None
    original_latency_ms: float
    original_tokens: int
    has_normalized: bool = False
    normalized_artifact: Any = None


def _unwrap_backend(backend: Any) -> Any:
    curr = backend
    for _ in range(4):
        inner = getattr(curr, "inner", None)
        if isinstance(inner, (LogitBackend, GenerativeBackend)):
            curr = inner
        else:
            break
    return curr


def _callable_fingerprint(fn: Callable[..., Any]) -> dict[str, Any]:
    qualname = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None) or type(fn).__qualname__
    module = getattr(fn, "__module__", "")
    code_obj = getattr(fn, "__code__", None)
    if code_obj is not None:
        raw_bytes = code_obj.co_code + repr(code_obj.co_consts).encode("utf-8", errors="replace")
        code_hash = hashlib.sha256(raw_bytes).hexdigest()[:24]
        return {"fn": f"{module}.{qualname}", "code": code_hash}
    return {"fn": f"{module}.{qualname}", "repr": repr(fn)[:200]}


def _canonicalize_value(val: Any, *, _depth: int = 0) -> Any:
    if _depth > 16:
        return {"t": "depth_capped", "v": repr(val)[:256]}
    if val is None or isinstance(val, bool):
        return {"t": type(val).__name__, "v": val}
    if isinstance(val, int):
        return {"t": "int", "v": val}
    if isinstance(val, float):
        if math.isnan(val):
            return {"t": "float", "v": "NaN"}
        if math.isinf(val):
            return {"t": "float", "v": "Inf" if val > 0 else "-Inf"}
        return {"t": "float", "v": round(val, 8)}
    if isinstance(val, str):
        return {"t": "str", "v": val}
    if isinstance(val, (bytes, bytearray)):
        return {"t": "bytes", "v": bytes(val).hex()}
    if isinstance(val, BaseModel):
        return {
            "t": f"pydantic:{type(val).__module__}.{type(val).__qualname__}",
            "v": _canonicalize_value(val.model_dump(mode="json"), _depth=_depth + 1),
        }
    if isinstance(val, Mapping):
        items = sorted(
            ((str(k), _canonicalize_value(v, _depth=_depth + 1)) for k, v in val.items()),
            key=lambda kv: kv[0],
        )
        return {"t": "map", "v": items}
    if isinstance(val, tuple):
        return {"t": "tuple", "v": [_canonicalize_value(x, _depth=_depth + 1) for x in val]}
    if isinstance(val, list):
        return {"t": "list", "v": [_canonicalize_value(x, _depth=_depth + 1) for x in val]}
    if isinstance(val, (set, frozenset)):
        canon_elems = [_canonicalize_value(x, _depth=_depth + 1) for x in val]
        canon_elems.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
        return {"t": "set", "v": canon_elems}
    if isinstance(val, type) and issubclass(val, BaseModel):
        return {
            "t": "pydantic_schema",
            "cls": f"{val.__module__}.{val.__qualname__}",
            "schema": val.model_json_schema(),
        }
    if callable(val):
        return {"t": "callable", "v": _callable_fingerprint(val)}
    return {"t": f"obj:{type(val).__module__}.{type(val).__qualname__}", "v": repr(val)}


def _backend_fingerprint(backend: Any, gate_model: str | None = None) -> dict[str, Any]:
    if backend is None:
        return {"spec": None, "model": gate_model}
    spec = infer_backend_spec(backend)
    raw = _unwrap_backend(backend)
    fp: dict[str, Any] = {"spec": spec, "model": gate_model}
    if isinstance(raw, StubBackend):
        fp["stub"] = {
            "default_prob": raw.default_prob,
            "forward": _canonicalize_value(raw.forward_top_logprobs),
            "reverse": _canonicalize_value(raw.reverse_top_logprobs),
            "diag": raw.diagnostic_text,
        }
    elif spec.startswith("custom://"):
        fp["class"] = f"{type(raw).__module__}.{type(raw).__qualname__}"
        if hasattr(raw, "cache_id"):
            fp["cache_id"] = _canonicalize_value(raw.cache_id)
    options = getattr(raw, "options", None)
    if options:
        fp["options"] = _canonicalize_value(options)
    lora_path = getattr(raw, "lora_path", None)
    if lora_path:
        fp["lora_path"] = str(lora_path)
        fp["lora_scale"] = round(float(getattr(raw, "lora_scale", 1.0)), 6)
    return fp


def _profile_fingerprint(profile: Any) -> dict[str, Any] | None:
    if profile is None:
        return None
    if hasattr(profile, "model_dump"):
        dumped = profile.model_dump(mode="json")
        dumped.pop("created_at", None)
        return _canonicalize_value(dumped)
    return _canonicalize_value(str(profile))


def gate_cache_fingerprint(gate: BaseGate) -> dict[str, Any]:
    """Builds a secret-free, deterministic structural dictionary describing ``gate``."""
    cls_name = f"{type(gate).__module__}.{type(gate).__qualname__}"
    fp: dict[str, Any] = {
        "class": cls_name,
        "name": getattr(gate, "name", type(gate).__name__),
        "layer_tier": getattr(gate, "layer_tier", 1),
        "can_fast_path": getattr(gate, "can_fast_path", True),
        "can_override_fail": getattr(gate, "can_override_fail", False),
        "escalation_policy": getattr(gate, "escalation_policy", "on_ambiguous_or_fail"),
    }

    short_cls = type(gate).__name__
    if short_cls == "Cascade":
        child_gates: Sequence[BaseGate] = getattr(gate, "gates", ())
        fp["gates"] = [gate_cache_fingerprint(g) for g in child_gates]
        fp["fuse_logits"] = getattr(gate, "fuse_logits", True)
        fp["prior_weight"] = round(float(getattr(gate, "prior_weight", 0.30)), 6)
        fp["normalize_artifacts"] = getattr(gate, "normalize_artifacts", False)
        return fp

    if short_cls == "BestOfNGate":
        pre = getattr(gate, "pre_gate", None)
        post = getattr(gate, "post_gate", None)
        rg = getattr(gate, "rank_gate", None)
        fp["pre_gate"] = gate_cache_fingerprint(pre) if pre is not None else None
        fp["rank_gate"] = gate_cache_fingerprint(rg) if rg is not None else None
        fp["post_gate"] = gate_cache_fingerprint(post) if post is not None else None
        return fp

    if short_cls == "MultiRubricGate":
        inner = getattr(gate, "_gate", None)
        fp["rubrics"] = list(getattr(gate, "rubrics", ()))
        fp["fail_fast"] = bool(getattr(gate, "fail_fast", False))
        fp["inner"] = gate_cache_fingerprint(inner) if inner is not None else None
        return fp

    if short_cls == "ASTGate":
        fp["syntax"] = getattr(gate, "syntax", False)
        fp["forbidden_imports"] = _canonicalize_value(getattr(gate, "forbidden_imports", None))
        fp["forbidden_patterns"] = _canonicalize_value(getattr(gate, "forbidden_patterns", None))
        fp["required_sections"] = _canonicalize_value(getattr(gate, "required_sections", None))
        fp["acyclic_dag"] = getattr(gate, "acyclic_dag", False)
        fp["json_schema"] = _canonicalize_value(getattr(gate, "json_schema", None))
        fp["min_length"] = getattr(gate, "min_length", None)
        fp["max_length"] = getattr(gate, "max_length", None)
        fp["normalize"] = _canonicalize_value(getattr(gate, "normalize", False))
        validators = getattr(gate, "validators", None) or ()
        fp["validators"] = [_callable_fingerprint(v) for v in validators]
        return fp

    backend = getattr(gate, "backend", None)
    if backend is not None:
        fp["backend"] = _backend_fingerprint(backend, getattr(gate, "model", None))

    profile = getattr(gate, "profile", None)
    if profile is not None:
        fp["profile"] = _profile_fingerprint(profile)

    aci = getattr(gate, "adaptive_controller", None)
    if aci is not None and hasattr(aci, "cache_fingerprint"):
        fp["adaptive_controller"] = _canonicalize_value(aci.cache_fingerprint())

    anchor_bank = getattr(gate, "anchor_bank", None)
    if anchor_bank is not None and hasattr(anchor_bank, "cache_fingerprint"):
        fp["anchor_bank"] = _canonicalize_value(anchor_bank.cache_fingerprint())

    meta_cal = getattr(gate, "meta_calibrator", None)
    if meta_cal is not None and hasattr(meta_cal, "cache_fingerprint"):
        fp["meta_calibrator"] = _canonicalize_value(meta_cal.cache_fingerprint())

    unanch_meta = getattr(gate, "unanchored_meta_calibrator", None)
    if unanch_meta is not None and hasattr(unanch_meta, "cache_fingerprint"):
        fp["unanchored_meta_calibrator"] = _canonicalize_value(unanch_meta.cache_fingerprint())

    replay = gate.replay_config()
    if replay is not None:
        fp["replay"] = _canonicalize_value(replay)

    for attr in (
        "rubric",
        "question",
        "options",
        "context",
        "pass_threshold",
        "fail_threshold",
        "threshold",
        "min_score",
        "max_score",
        "max_score_std",
        "min_confidence",
        "min_margin",
        "max_positional_bias",
        "logit_temperature",
        "symmetric_debiasing",
        "permutation_mode",
        "mode",
        "query_first",
        "max_tokens",
        "think",
        "allow_fail_override",
    ):
        if hasattr(gate, attr):
            fp[attr] = _canonicalize_value(getattr(gate, attr))

    if replay is None and backend is None:
        # Custom BaseGate subclass: capture deterministic public attributes
        custom_attrs: dict[str, Any] = {}
        for k, v in sorted(vars(gate).items()):
            if k.startswith("_") or k in ("telemetry_sink", "cache", "save_on_fail"):
                continue
            custom_attrs[k] = _canonicalize_value(v)
        fp["custom_attrs"] = custom_attrs

    return fp


def compute_verdict_cache_key(
    gate: BaseGate,
    candidate: Any,
    *,
    rubric_override: str | None = None,
    context_override: Any = _NO_OVERRIDE,
    namespace: str = "",
) -> str:
    """Computes a collision-resistant SHA-256 content-addressed cache key."""
    payload: dict[str, Any] = {
        "v": 1,
        "ns": namespace,
        "gate": gate_cache_fingerprint(gate),
        "rubric_override": rubric_override,
        "candidate": _canonicalize_value(candidate),
    }
    if context_override is not _NO_OVERRIDE:
        payload["context_override"] = _canonicalize_value(context_override)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return _CACHE_KEY_PREFIX + hashlib.sha256(encoded).hexdigest()


def _prepare_secure_sqlite_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if str(path) == ":memory:":
        raise ConfigurationError("Pass path=None for in-memory VerdictCache rather than ':memory:'")

    parent = path.parent if str(path.parent) else Path(".")
    for ancestor in (parent, *parent.parents):
        if ancestor.exists() and ancestor.is_symlink():
            raise ConfigurationError(
                f"Refusing to open VerdictCache inside symlinked directory: {ancestor}"
            )
    parent.mkdir(parents=True, exist_ok=True)

    for suffix in ("", "-wal", "-shm", "-journal"):
        check_path = Path(f"{path}{suffix}")
        try:
            st = os.lstat(check_path)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(st.st_mode):
            raise ConfigurationError(f"Refusing to open symlinked VerdictCache file: {check_path}")
        if not stat.S_ISREG(st.st_mode):
            raise ConfigurationError(
                f"Refusing to open non-regular VerdictCache file: {check_path}"
            )
        if suffix == "" and st.st_size > _MAX_SQLITE_DB_BYTES:
            raise ConfigurationError(
                f"VerdictCache SQLite file {check_path} exceeds {_MAX_SQLITE_DB_BYTES} bytes cap"
            )

    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(str(path), flags, 0o600)
    try:
        st_fd = os.fstat(fd)
        if not stat.S_ISREG(st_fd.st_mode):
            raise ConfigurationError(f"VerdictCache target is not a regular file: {path}")
        try:
            os.fchmod(fd, 0o600)
        except OSError:
            pass
    finally:
        os.close(fd)
    return path


class VerdictCache:
    """Thread-safe content-addressed `Verdict` cache with LRU memory and optional SQLite storage.

    Keys hash the full structural fingerprint of the gate (class, backend/model, rubric,
    thresholds, temperature, `CalibrationProfile`, and `ASTGate` validator bytecode) together
    with the canonicalized candidate, preventing cross-model or cross-threshold stale hits.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        max_entries: int = 4096,
        ttl_seconds: float | None = None,
        cache_ambiguous: bool = False,
        max_disk_entries: int = 50_000,
        namespace: str = "",
    ) -> None:
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 1:
            raise ConfigurationError(
                f"max_entries must be an integer >= 1, got {max_entries!r}"
            )
        if (
            isinstance(max_disk_entries, bool)
            or not isinstance(max_disk_entries, int)
            or max_disk_entries < 1
        ):
            raise ConfigurationError(
                f"max_disk_entries must be an integer >= 1, got {max_disk_entries!r}"
            )
        if ttl_seconds is not None:
            if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, (int, float)):
                raise ConfigurationError(
                    f"ttl_seconds must be a positive finite float or None, got {ttl_seconds!r}"
                )
            ttl_f = float(ttl_seconds)
            if not math.isfinite(ttl_f) or ttl_f <= 0.0:
                raise ConfigurationError(
                    f"ttl_seconds must be a positive finite float or None, got {ttl_seconds!r}"
                )
            self.ttl_seconds: float | None = ttl_f
        else:
            self.ttl_seconds = None

        if not isinstance(cache_ambiguous, bool):
            raise ConfigurationError(f"cache_ambiguous must be a bool, got {cache_ambiguous!r}")
        if not isinstance(namespace, str):
            raise ConfigurationError(f"namespace must be a str, got {namespace!r}")

        self.max_entries = max_entries
        self.max_disk_entries = max_disk_entries
        self.cache_ambiguous = cache_ambiguous
        self.namespace = namespace
        self.path: Path | None = _prepare_secure_sqlite_path(path) if path is not None else None

        self._lock = threading.RLock()
        self._mem: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._puts = 0
        self._evictions = 0
        self._saved_latency_ms = 0.0
        self._saved_tokens = 0
        self._closed = False
        self._conn: sqlite3.Connection | None = None

        if self.path is not None:
            self._init_sqlite(self.path)

    def _init_sqlite(self, db_path: Path) -> None:
        conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=5.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS verdict_cache (
                    cache_key TEXT PRIMARY KEY,
                    verdict_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL,
                    original_latency_ms REAL NOT NULL,
                    original_tokens INTEGER NOT NULL,
                    has_normalized INTEGER NOT NULL DEFAULT 0,
                    normalized_json TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_verdict_cache_created ON verdict_cache(created_at)"
            )
            conn.commit()
        except Exception:
            conn.close()
            raise
        self._conn = conn

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if self._conn is not None:
                try:
                    self._conn.close()
                except sqlite3.Error:
                    pass
                self._conn = None

    def compute_key(
        self,
        gate: BaseGate,
        candidate: Any,
        *,
        rubric_override: str | None = None,
        context_override: Any = _NO_OVERRIDE,
    ) -> str:
        return compute_verdict_cache_key(
            gate,
            candidate,
            rubric_override=rubric_override,
            context_override=context_override,
            namespace=self.namespace,
        )

    def _should_store(self, verdict: Verdict) -> bool:
        return verdict.status != DecisionStatus.AMBIGUOUS or self.cache_ambiguous

    @staticmethod
    def _materialize_hit(entry: _CacheEntry, lookup_ms: float, cache_key: str) -> Verdict:
        cloned = entry.verdict.model_copy(deep=True)
        cloned.total_latency_ms = round(lookup_ms, 4)
        if cloned.traces:
            first_meta = dict(cloned.traces[0].metadata or {})
            first_meta["cache_hit"] = True
            first_meta["cache_key"] = cache_key[:24]
            first_meta["original_total_latency_ms"] = round(entry.original_latency_ms, 4)
            cloned.traces[0].metadata = first_meta
            for tr in cloned.traces:
                tr.latency_ms = 0.0
        return cloned

    def _get_entry_locked(
        self,
        key: str,
        now: float,
    ) -> _CacheEntry | None:
        mem_entry = self._mem.get(key)
        if mem_entry is not None:
            if mem_entry.expires_at is not None and now >= mem_entry.expires_at:
                self._mem.pop(key, None)
                self._delete_sqlite_key_locked(key)
                return None
            self._mem.move_to_end(key)
            return mem_entry

        if self._conn is not None and not self._closed:
            try:
                row = self._conn.execute(
                    """
                    SELECT verdict_json, created_at, expires_at, original_latency_ms,
                           original_tokens, has_normalized, normalized_json
                    FROM verdict_cache WHERE cache_key = ?
                    """,
                    (key,),
                ).fetchone()
            except sqlite3.Error as exc:
                warnings.warn(f"VerdictCache SQLite read failed: {exc}", CacheWarning, stacklevel=2)
                row = None

            if row is not None:
                v_json, c_at, exp_at, orig_lat, orig_tok, has_norm_i, norm_json = row
                if exp_at is not None and now >= float(exp_at):
                    self._delete_sqlite_key_locked(key)
                    return None
                try:
                    verdict = Verdict.model_validate_json(v_json)
                    norm_val = json.loads(norm_json) if (has_norm_i and norm_json is not None) else None
                    disk_entry = _CacheEntry(
                        verdict=verdict,
                        created_at=float(c_at),
                        expires_at=float(exp_at) if exp_at is not None else None,
                        original_latency_ms=float(orig_lat),
                        original_tokens=int(orig_tok),
                        has_normalized=bool(has_norm_i),
                        normalized_artifact=norm_val,
                    )
                    self._insert_mem_locked(key, disk_entry)
                    return disk_entry
                except (ValueError, TypeError) as exc:
                    warnings.warn(
                        f"Evicting corrupt VerdictCache row {key[:20]}: {exc}",
                        CacheWarning,
                        stacklevel=2,
                    )
                    self._delete_sqlite_key_locked(key)

        return None

    def _delete_sqlite_key_locked(self, key: str) -> None:
        if self._conn is not None and not self._closed:
            try:
                self._conn.execute("DELETE FROM verdict_cache WHERE cache_key = ?", (key,))
                self._conn.commit()
            except sqlite3.Error:
                pass

    def _insert_mem_locked(self, key: str, entry: _CacheEntry) -> None:
        if key in self._mem:
            self._mem.move_to_end(key)
            self._mem[key] = entry
            return
        while len(self._mem) >= self.max_entries:
            self._mem.popitem(last=False)
            self._evictions += 1
        self._mem[key] = entry

    def _put_sqlite_locked(self, key: str, entry: _CacheEntry) -> None:
        if self._conn is None or self._closed:
            return
        norm_json: str | None = None
        has_norm_int = 0
        if entry.has_normalized:
            try:
                norm_json = json.dumps(entry.normalized_artifact)
                has_norm_int = 1
            except TypeError:
                has_norm_int = 0
                norm_json = None
        try:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO verdict_cache (
                    cache_key, verdict_json, created_at, expires_at,
                    original_latency_ms, original_tokens, has_normalized, normalized_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    entry.verdict.model_dump_json(),
                    entry.created_at,
                    entry.expires_at,
                    entry.original_latency_ms,
                    entry.original_tokens,
                    has_norm_int,
                    norm_json,
                ),
            )
            count_row = self._conn.execute("SELECT COUNT(*) FROM verdict_cache").fetchone()
            total_rows = int(count_row[0]) if count_row else 0
            if total_rows > self.max_disk_entries:
                excess = total_rows - self.max_disk_entries + max(1, self.max_disk_entries // 20)
                self._conn.execute(
                    """
                    DELETE FROM verdict_cache WHERE cache_key IN (
                        SELECT cache_key FROM verdict_cache ORDER BY created_at ASC LIMIT ?
                    )
                    """,
                    (excess,),
                )
                self._evictions += excess
            self._conn.commit()
        except sqlite3.Error as exc:
            warnings.warn(f"VerdictCache SQLite write failed: {exc}", CacheWarning, stacklevel=2)

    def get(
        self,
        gate: BaseGate,
        candidate: Any,
        *,
        rubric_override: str | None = None,
        context_override: Any = _NO_OVERRIDE,
        _normalized_out: list[Any] | None = None,
    ) -> Verdict | None:
        t0 = time.perf_counter()
        key = self.compute_key(
            gate,
            candidate,
            rubric_override=rubric_override,
            context_override=context_override,
        )
        now = time.time()
        with self._lock:
            entry = self._get_entry_locked(key, now)
            if entry is None:
                self._misses += 1
                return None
            self._hits += 1
            self._saved_latency_ms += entry.original_latency_ms
            self._saved_tokens += entry.original_tokens
            if _normalized_out is not None and entry.has_normalized:
                _normalized_out.append(entry.normalized_artifact)
            lookup_ms = (time.perf_counter() - t0) * 1000.0
            return self._materialize_hit(entry, lookup_ms, key)

    def put(
        self,
        gate: BaseGate,
        candidate: Any,
        verdict: Verdict,
        *,
        rubric_override: str | None = None,
        context_override: Any = _NO_OVERRIDE,
        has_normalized: bool = False,
        normalized_artifact: Any = None,
    ) -> bool:
        if not self._should_store(verdict):
            return False
        key = self.compute_key(
            gate,
            candidate,
            rubric_override=rubric_override,
            context_override=context_override,
        )
        now = time.time()
        expires_at = (now + self.ttl_seconds) if self.ttl_seconds is not None else None
        stored_verdict = verdict.model_copy(deep=True)
        entry = _CacheEntry(
            verdict=stored_verdict,
            created_at=now,
            expires_at=expires_at,
            original_latency_ms=float(verdict.total_latency_ms),
            original_tokens=int(verdict.total_tokens_generated),
            has_normalized=has_normalized,
            normalized_artifact=normalized_artifact,
        )
        with self._lock:
            self._puts += 1
            self._insert_mem_locked(key, entry)
            self._put_sqlite_locked(key, entry)
        return True

    def invalidate(
        self,
        gate: BaseGate,
        candidate: Any,
        *,
        rubric_override: str | None = None,
        context_override: Any = _NO_OVERRIDE,
    ) -> bool:
        key = self.compute_key(
            gate,
            candidate,
            rubric_override=rubric_override,
            context_override=context_override,
        )
        with self._lock:
            existed = self._mem.pop(key, None) is not None
            if self._conn is not None and not self._closed:
                try:
                    cur = self._conn.execute("DELETE FROM verdict_cache WHERE cache_key = ?", (key,))
                    self._conn.commit()
                    existed = existed or (cur.rowcount > 0)
                except sqlite3.Error:
                    pass
            return existed

    def clear(self) -> None:
        with self._lock:
            self._mem.clear()
            if self._conn is not None and not self._closed:
                try:
                    self._conn.execute("DELETE FROM verdict_cache")
                    self._conn.commit()
                except sqlite3.Error:
                    pass

    def stats(self) -> CacheStats:
        with self._lock:
            disk_count = 0
            if self._conn is not None and not self._closed:
                try:
                    row = self._conn.execute("SELECT COUNT(*) FROM verdict_cache").fetchone()
                    disk_count = int(row[0]) if row else 0
                except sqlite3.Error:
                    disk_count = 0
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                puts=self._puts,
                evictions=self._evictions,
                memory_entries=len(self._mem),
                disk_entries=disk_count,
                saved_latency_ms=self._saved_latency_ms,
                saved_tokens=self._saved_tokens,
            )

    def evaluate(self, gate: BaseGate, candidate: Any, **kwargs: Any) -> Verdict:
        """Evaluates ``gate`` on ``candidate`` through this cache."""
        return gate.evaluate(candidate, cache=self, **kwargs)

    async def aevaluate(self, gate: BaseGate, candidate: Any, **kwargs: Any) -> Verdict:
        """Asynchronously evaluates ``gate`` on ``candidate`` through this cache."""
        return await gate.aevaluate(candidate, cache=self, **kwargs)


def resolve_verdict_cache(
    spec: VerdictCache | str | Path | Literal[False] | None,
) -> VerdictCache | Literal[False] | None:
    """Resolves a `VerdictCache`, SQLite file path, ``False``, or ``None``."""
    if spec is None or spec is False:
        return spec
    if isinstance(spec, VerdictCache):
        return spec
    if isinstance(spec, (str, Path)):
        s = str(spec).strip()
        if not s:
            raise ConfigurationError("cache path cannot be empty")
        return VerdictCache(path=s)
    raise ConfigurationError(
        f"cache must be a VerdictCache, file path, False, or None, got {type(spec).__name__}"
    )


def _resolve_active_cache(
    gate_cache: Any,
    call_kwargs: Mapping[str, Any] | None,
) -> tuple[VerdictCache | None, bool]:
    """Returns ``(active_cache, owns_temp_cache)`` with fast-return when unconfigured."""
    if call_kwargs is not None and call_kwargs.get("_bypass_cache"):
        return None, False

    call_cache_raw = call_kwargs.get("cache") if call_kwargs is not None else None
    if call_cache_raw is not None:
        if call_cache_raw is False:
            return None, False
        if isinstance(call_cache_raw, VerdictCache):
            return call_cache_raw, False
        resolved = resolve_verdict_cache(call_cache_raw)
        if resolved is False or resolved is None:
            return None, False
        return resolved, True

    if gate_cache is False:
        return None, False
    if isinstance(gate_cache, VerdictCache):
        return gate_cache, False

    from logitgate.config import get_default_verdict_cache

    return get_default_verdict_cache(), False


def lookup_cached_verdict(
    gate: BaseGate,
    candidate: Any,
    *,
    gate_cache: Any = None,
    call_kwargs: Mapping[str, Any] | None = None,
) -> Verdict | None:
    """Sub-microsecond no-op when caching is inactive; returns cached `Verdict` on hit."""
    active_cache, owns_temp = _resolve_active_cache(gate_cache, call_kwargs)
    if active_cache is None:
        return None
    try:
        rubric_override = (
            call_kwargs.get("rubric_override") if call_kwargs is not None else None
        )
        context_override = (
            call_kwargs["context"]
            if (call_kwargs is not None and "context" in call_kwargs)
            else _NO_OVERRIDE
        )
        norm_box = call_kwargs.get("_normalized_out") if call_kwargs is not None else None
        return active_cache.get(
            gate,
            candidate,
            rubric_override=rubric_override,
            context_override=context_override,
            _normalized_out=norm_box if isinstance(norm_box, list) else None,
        )
    finally:
        if owns_temp:
            active_cache.close()


def store_cached_verdict(
    gate: BaseGate,
    candidate: Any,
    verdict: Verdict,
    *,
    gate_cache: Any = None,
    call_kwargs: Mapping[str, Any] | None = None,
    has_normalized: bool = False,
    normalized_artifact: Any = None,
) -> None:
    """Stores ``verdict`` in the active `VerdictCache` if configured."""
    active_cache, owns_temp = _resolve_active_cache(gate_cache, call_kwargs)
    if active_cache is None:
        return
    try:
        rubric_override = (
            call_kwargs.get("rubric_override") if call_kwargs is not None else None
        )
        context_override = (
            call_kwargs["context"]
            if (call_kwargs is not None and "context" in call_kwargs)
            else _NO_OVERRIDE
        )
        active_cache.put(
            gate,
            candidate,
            verdict,
            rubric_override=rubric_override,
            context_override=context_override,
            has_normalized=has_normalized,
            normalized_artifact=normalized_artifact,
        )
    finally:
        if owns_temp:
            active_cache.close()
