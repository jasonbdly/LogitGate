from __future__ import annotations

import ast
import json
import re
import time
from collections import deque
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel, ValidationError

from logitgate.backends.base import strip_reasoning_blocks
from logitgate.exceptions import ConfigurationError
from logitgate.gates.base import TIER_DETERMINISTIC, BaseGate
from logitgate.models import DecisionStatus, LayerTrace, Verdict

# Fallback import scanner for code that doesn't parse (e.g. a snippet with a syntax error
# when only `forbidden_imports` is enabled). Captures `os.path` from `import os.path` and
# `from os.path import join`. Uses horizontal whitespace `[ \t]` and non-empty module names
# to avoid quadratic backtracking on long runs of newlines or identifiers.
_IMPORT_STMT_RE = re.compile(
    r"(?:^|;)[ \t]*(?:from[ \t]+([a-zA-Z0-9_\.]+)[ \t]+import[ \t]+([a-zA-Z0-9_\., \t\*]+)|import[ \t]+([a-zA-Z0-9_\., \t]+))",
    re.MULTILINE,
)
_FENCED_BLOCK_RE = re.compile(r"```[^\r\n`]*\r?\n(.*?)```", re.DOTALL)
_INLINE_FENCE_RE = re.compile(r"```(.*?)```", re.DOTALL)
_MAX_JSON_DECODE_ATTEMPTS = 128


def normalize_candidate(candidate: Any) -> Any:
    """Strips `<think>...</think>` reasoning traces and unwraps markdown code fences if present.

    Non-string candidates (such as pre-parsed `dict` or `list` objects) are returned unchanged.
    """
    if not isinstance(candidate, str):
        return candidate
    cleaned = strip_reasoning_blocks(candidate)
    match = _FENCED_BLOCK_RE.search(cleaned)
    if match:
        return match.group(1).strip()
    inline_match = _INLINE_FENCE_RE.search(cleaned)
    if inline_match:
        return inline_match.group(1).strip()
    return cleaned.strip()


def _extract_json_with_span(text: str, *, prefer_object: bool = False) -> tuple[Any, str]:
    """Attempts `json.loads` first, then falls back to `raw_decode` on embedded JSON.

    Returns `(parsed_object, json_substring)`.
    """
    normalized = normalize_candidate(text)
    try:
        return json.loads(normalized), normalized
    except (json.JSONDecodeError, RecursionError) as err:
        if isinstance(err, RecursionError):
            first_err = json.JSONDecodeError("Maximum JSON nesting depth exceeded", normalized, 0)
        else:
            first_err = err

    decoder = json.JSONDecoder()
    n = len(normalized)
    fallback_obj: Any = None
    fallback_span: str = normalized
    has_fallback = False
    attempts = 0
    idx = 0

    while idx < n:
        ch = normalized[idx]
        if ch in ("{", "["):
            try:
                obj, end = decoder.raw_decode(normalized, idx)
            except (json.JSONDecodeError, RecursionError):
                attempts += 1
                if attempts >= _MAX_JSON_DECODE_ATTEMPTS:
                    break
                idx += 1
                continue
            span = normalized[idx:end]
            if prefer_object:
                if isinstance(obj, dict):
                    return obj, span
                if not has_fallback:
                    fallback_obj = obj
                    fallback_span = span
                    has_fallback = True
            else:
                if isinstance(obj, dict) or (isinstance(obj, list) and (not obj or isinstance(obj[0], dict))):
                    return obj, span
                if not has_fallback:
                    fallback_obj = obj
                    fallback_span = span
                    has_fallback = True
            # Advance past the decoded value so we never re-scan inner brackets of a top-level array/object.
            idx = max(idx + 1, end)
        else:
            idx += 1

    if has_fallback:
        return fallback_obj, fallback_span

    raise first_err


def _extract_json_from_text(text: str, *, prefer_object: bool = False) -> Any:
    """Attempts `json.loads` first, then falls back to `raw_decode` on embedded JSON."""
    obj, _ = _extract_json_with_span(text, prefer_object=prefer_object)
    return obj


def _compile_user_pattern(pattern: str) -> re.Pattern[str] | None:
    """Compiles a user-supplied pattern once. ``None`` means "treat as a literal substring".

    Users routinely pass strings like ``"eval("`` or ``"$HOME"`` that are invalid or
    unintended regexes, so invalid patterns degrade to literal matching instead of raising.
    """
    try:
        return re.compile(pattern)
    except re.error:
        return None


def _detect_dag_cycle(tasks: list[dict[str, Any]]) -> str | None:
    """Validates that every dependency exists and the dependency graph is acyclic.

    Uses Kahn's algorithm with a deque, so it runs in O(V + E). (``list.pop(0)`` is O(V) per
    pop, which makes the whole sort O(V^2) on large plans.) Node order follows the input so
    error messages are stable across runs and don't depend on hash seeds.

    Returns:
        A human-readable error message, or ``None`` if the graph is a valid DAG.
    """
    if not isinstance(tasks, list):
        return f"Expected list of tasks for DAG check, got {type(tasks).__name__}"

    task_ids: dict[str, None] = {}  # insertion-ordered set
    for t in tasks:
        if not isinstance(t, dict):
            return f"Expected task node to be dict, got {type(t).__name__}"
        tid = t.get("id") if t.get("id") is not None else t.get("task_id")
        if tid is None or str(tid).strip() == "":
            return f"Task node missing required 'id' or 'task_id': {t}"
        tid_str = str(tid)
        if tid_str in task_ids:
            return f"Duplicate task ID '{tid_str}' in DAG"
        task_ids[tid_str] = None

    adj: dict[str, list[str]] = {tid: [] for tid in task_ids}
    in_degree: dict[str, int] = dict.fromkeys(task_ids, 0)

    for t in tasks:
        raw_tid = t.get("id") if t.get("id") is not None else t.get("task_id")
        tid = str(raw_tid)
        deps = t.get("dependencies") if t.get("dependencies") is not None else t.get("deps")
        if deps is None:
            deps = []
        if not isinstance(deps, (list, tuple)):
            return f"Task '{tid}' dependencies must be a list, got {type(deps).__name__}"
        for dep in deps:
            dep_str = str(dep)
            if dep_str not in task_ids:
                return f"Missing dependency: task '{tid}' depends on non-existent task '{dep_str}'"
            adj[dep_str].append(tid)
            in_degree[tid] += 1

    ready = deque(tid for tid, deg in in_degree.items() if deg == 0)
    visited_count = 0
    while ready:
        curr = ready.popleft()
        visited_count += 1
        for neighbor in adj[curr]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                ready.append(neighbor)

    if visited_count != len(task_ids):
        unresolved = [tid for tid, deg in in_degree.items() if deg > 0]
        return f"Cycle detected in task dependencies among: {unresolved}"
    return None


class ASTFailureCode:
    """Standardized failure codes for Layer 0 deterministic AST and contract verification."""

    MIN_LENGTH_VIOLATION = "MIN_LENGTH_VIOLATION"
    MAX_LENGTH_VIOLATION = "MAX_LENGTH_VIOLATION"
    JSON_SYNTAX_ERROR = "JSON_SYNTAX_ERROR"
    JSON_TYPE_ERROR = "JSON_TYPE_ERROR"
    JSON_SCHEMA_VALIDATION_ERROR = "JSON_SCHEMA_VALIDATION_ERROR"
    SYNTAX_ERROR = "SYNTAX_ERROR"
    FORBIDDEN_IMPORT = "FORBIDDEN_IMPORT"
    FORBIDDEN_PATTERN = "FORBIDDEN_PATTERN"
    MISSING_SECTION = "MISSING_SECTION"
    DAG_CYCLE_DETECTED = "DAG_CYCLE_DETECTED"
    CUSTOM_VALIDATOR_FAILED = "CUSTOM_VALIDATOR_FAILED"


class ASTGate(BaseGate):
    """Deterministic CPU verification and BYO validator integration gate (Layer 0).

    Executes in under 0.1ms using native CPU parsing or third-party validators without
    invoking any LLM. Designed as a minimal integration surface between LogitGate cascades
    and any bring-your-own (BYO) syntactic parser or schema validator (Pydantic, `jsonschema`,
    Tree-sitter, SQLGlot, `ast.parse`, etc.).

    Validator contract:
        - Returns `True`, `None`, or any non-`False` parsed object without raising -> PASSED
        - Returns `False` or `(False, "reason")` -> FAILED
        - Raises any `Exception` (`ValidationError`, `SyntaxError`, `ValueError`, etc.) -> FAILED

    Args:
        *pos_validators: Positional validator callables, Pydantic `BaseModel` classes, or `TypeAdapter`s.
        syntax: When True, parses the candidate as Python code via `ast.parse`.
        forbidden_imports: Sequence of module names prohibited from being imported.
        forbidden_patterns: Sequence of regular expressions prohibited in the artifact.
        required_sections: Sequence of string headers or regex patterns required to be present.
        acyclic_dag: When True, interprets candidate as a list/dict of task nodes and ensures
            all dependencies exist and form an acyclic directed graph (DAG).
        json_schema: Pydantic `BaseModel` class or JSON schema dictionary to validate against.
        min_length: Minimum allowed character length of the artifact.
        max_length: Maximum allowed character length of the artifact.
        validators: Sequence of custom callables or validators `fn(candidate) -> Any`.
        normalize: When True, strips `<think>...</think>` blocks and unwraps markdown code fences
            (or extracts embedded JSON) before running checks. May also be a custom preprocessor
            callable `fn(candidate) -> Any`.
        name: Identifier for this gate instance.
    """

    layer_tier = TIER_DETERMINISTIC
    # A passing structural check carries no semantic signal: a downstream judge must still run.
    can_fast_path = False

    def __init__(
        self,
        *pos_validators: Any,
        syntax: bool = False,
        forbidden_imports: Sequence[str] | None = None,
        forbidden_patterns: Sequence[str] | None = None,
        required_sections: Sequence[str] | None = None,
        acyclic_dag: bool = False,
        json_schema: dict[str, Any] | type[BaseModel] | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        validators: Sequence[Callable[[Any], Any]] | None = None,
        normalize: bool | Callable[[Any], Any] = False,
        save_on_fail: bool | None = None,
        name: str = "ASTGate",
    ) -> None:
        if min_length is not None and (isinstance(min_length, bool) or not isinstance(min_length, int) or min_length < 0):
            raise ConfigurationError(f"min_length must be a non-negative integer, got {min_length!r}")
        if max_length is not None and (isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 0):
            raise ConfigurationError(f"max_length must be a non-negative integer, got {max_length!r}")
        if min_length is not None and max_length is not None and min_length > max_length:
            raise ConfigurationError(
                f"min_length ({min_length}) cannot exceed max_length ({max_length})"
            )

        combined_validators: list[Any] = []
        for item in pos_validators:
            if isinstance(item, bool):
                syntax = item
            elif isinstance(item, dict) and json_schema is None:
                json_schema = item
            elif isinstance(item, (list, tuple, set)):
                combined_validators.extend(item)
            elif item is not None:
                combined_validators.append(item)
        if validators:
            combined_validators.extend(validators)

        self.syntax = syntax
        self.forbidden_imports = list(forbidden_imports) if forbidden_imports else None
        self.forbidden_patterns = list(forbidden_patterns) if forbidden_patterns else None
        self.required_sections = list(required_sections) if required_sections else None
        self.acyclic_dag = acyclic_dag
        self.json_schema = json_schema
        self.min_length = min_length
        self.max_length = max_length
        self.validators = combined_validators if combined_validators else None
        self.normalize = normalize
        self.save_on_fail = save_on_fail
        self.name = name
        self._forbidden_res = [(p, _compile_user_pattern(p)) for p in self.forbidden_patterns or []]
        self._required_res = [(s, _compile_user_pattern(s)) for s in self.required_sections or []]

    def __repr__(self) -> str:
        features = []
        if self.syntax:
            features.append("syntax=True")
        if self.forbidden_imports:
            features.append(f"forbidden_imports={self.forbidden_imports!r}")
        if self.forbidden_patterns:
            features.append(f"forbidden_patterns={self.forbidden_patterns!r}")
        if self.required_sections:
            features.append(f"required_sections={self.required_sections!r}")
        if self.acyclic_dag:
            features.append("acyclic_dag=True")
        if self.json_schema:
            schema_name = getattr(self.json_schema, "__name__", str(self.json_schema))
            features.append(f"json_schema={schema_name}")
        if self.min_length is not None:
            features.append(f"min_length={self.min_length}")
        if self.max_length is not None:
            features.append(f"max_length={self.max_length}")
        if self.validators:
            features.append(f"validators=[{len(self.validators)} callables]")
        if self.normalize:
            features.append(f"normalize={self.normalize!r}")
        feat_str = f", {', '.join(features)}" if features else ""
        return f"ASTGate(name='{self.name}'{feat_str})"

    def _fail(
        self,
        err: str,
        code: str,
        t0: float,
        metadata: dict[str, Any] | None = None,
        candidate: Any = None,
    ) -> Verdict:
        latency = (time.perf_counter() - t0) * 1000.0
        meta = {"failure_code": code}
        if metadata:
            meta.update(metadata)
        trace = LayerTrace(
            layer_name=self.name,
            status=DecisionStatus.FAILED,
            latency_ms=latency,
            diagnostic_report=err,
            failure_code=code,
            metadata=meta,
        )
        verdict = Verdict(
            passed=False,
            status=DecisionStatus.FAILED,
            triggered_layer=self.name,
            total_latency_ms=latency,
            total_tokens_generated=0,
            reason=err,
            traces=[trace],
        )
        from logitgate.snapshot import persist_failure_snapshot

        persist_failure_snapshot(
            verdict=verdict,
            candidate=candidate,
            save_on_fail=self.save_on_fail,
            gate_type="ASTGate",
            gate_config={"name": self.name, "failure_code": code},
        )
        return verdict

    @staticmethod
    def _extract_imports_from_source(source: str, parsed_ast: ast.AST | None = None) -> set[str]:
        found_imports: set[str] = set()
        if parsed_ast is None:
            try:
                parsed_ast = ast.parse(source)
            except (SyntaxError, ValueError, RecursionError, MemoryError):
                parsed_ast = None
        if parsed_ast is not None:
            for node in ast.walk(parsed_ast):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        found_imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        found_imports.add(node.module)
                        for alias in node.names:
                            if alias.name and alias.name != "*":
                                found_imports.add(f"{node.module}.{alias.name}")
                    else:
                        for alias in node.names:
                            if alias.name and alias.name != "*":
                                found_imports.add(alias.name)
        else:
            for match in _IMPORT_STMT_RE.finditer(source):
                from_mod, from_names, imp_names = match.group(1), match.group(2), match.group(3)
                if imp_names is not None:
                    for part in imp_names.split(","):
                        mod_token = part.strip().split()[0] if part.strip() else ""
                        if mod_token:
                            found_imports.add(mod_token)
                else:
                    base_mod = (from_mod or "").strip(".")
                    if base_mod:
                        found_imports.add(base_mod)
                    for part in (from_names or "").split(","):
                        name_tok = part.strip().split()[0] if part.strip() else ""
                        if name_tok and name_tok != "*":
                            found_imports.add(f"{base_mod}.{name_tok}" if base_mod else name_tok)
        return found_imports

    @staticmethod
    def _run_validator(
        val_fn: Any, candidate: Any, *, normalized: bool = False
    ) -> tuple[bool, str, str | None]:
        val_name = getattr(val_fn, "__name__", getattr(type(val_fn), "__name__", str(val_fn)))
        extracted_span: str | None = None
        try:
            if isinstance(val_fn, type) and issubclass(val_fn, BaseModel):
                if isinstance(candidate, str):
                    if normalized:
                        payload, extracted_span = _extract_json_with_span(candidate, prefer_object=True)
                    else:
                        payload = json.loads(candidate)
                elif isinstance(candidate, BaseModel) and not isinstance(candidate, val_fn):
                    payload = candidate.model_dump()
                else:
                    payload = candidate
                val_fn.model_validate(payload)
                return True, "", extracted_span
            if hasattr(val_fn, "validate_json") and hasattr(val_fn, "validate_python"):
                if isinstance(candidate, str):
                    if normalized:
                        payload, extracted_span = _extract_json_with_span(candidate)
                        val_fn.validate_python(payload)
                    else:
                        val_fn.validate_json(candidate)
                else:
                    val_fn.validate_python(candidate)
                return True, "", extracted_span
            if not callable(val_fn) and hasattr(val_fn, "validate"):
                if isinstance(candidate, str):
                    if normalized:
                        payload, extracted_span = _extract_json_with_span(candidate)
                    else:
                        payload = json.loads(candidate)
                else:
                    payload = candidate
                val_fn.validate(payload)
                return True, "", extracted_span

            res = val_fn(candidate)
            if isinstance(res, Verdict):
                return res.passed, res.reason or f"Validator '{val_name}' rejected artifact", None
            if isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], bool):
                msg = str(res[1]) if res[1] else f"Validator '{val_name}' rejected artifact"
                return res[0], ("" if res[0] else msg), None
            if res is False:
                return False, f"Validator '{val_name}' rejected artifact", None
            return True, "", None
        except Exception as ex:  # noqa: BLE001
            return False, f"Validator raised exception: {type(ex).__name__}: {ex}", None

    def normalize_input(self, candidate: Any) -> Any:
        """Applies configured candidate normalization (`normalize_candidate` or custom callable)."""
        if callable(self.normalize):
            return self.normalize(candidate)
        if self.normalize is True and isinstance(candidate, str):
            if (
                self.required_sections
                and not self.syntax
                and self.json_schema is None
                and not self.acyclic_dag
                and not self.validators
            ):
                return strip_reasoning_blocks(candidate).strip()
            unwrapped = normalize_candidate(candidate)
            needs_json_span = (
                self.json_schema is not None
                or self.acyclic_dag
                or any(
                    (isinstance(v, type) and issubclass(v, BaseModel))
                    or (hasattr(v, "validate_json") and hasattr(v, "validate_python"))
                    or (not callable(v) and hasattr(v, "validate"))
                    for v in (self.validators or ())
                )
            )
            if needs_json_span and isinstance(unwrapped, str):
                prefer_obj = (
                    (isinstance(self.json_schema, type) and issubclass(self.json_schema, BaseModel))
                    or (
                        isinstance(self.json_schema, dict)
                        and self.json_schema.get("type") not in ("array", "string", "integer", "number", "boolean", "null")
                    )
                    or any(isinstance(v, type) and issubclass(v, BaseModel) for v in (self.validators or ()))
                )
                try:
                    _, span = _extract_json_with_span(unwrapped, prefer_object=prefer_obj)
                    return span
                except json.JSONDecodeError:
                    return unwrapped
            return unwrapped
        return candidate

    def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        t0 = time.perf_counter()
        orig_candidate = candidate
        raw_text_without_think = (
            strip_reasoning_blocks(candidate) if isinstance(candidate, str) else None
        )

        if callable(self.normalize):
            try:
                candidate = self.normalize(candidate)
            except Exception as ex:  # noqa: BLE001
                return self._fail(
                    f"Normalizer raised exception: {ex}",
                    ASTFailureCode.CUSTOM_VALIDATOR_FAILED,
                    t0,
                    candidate=orig_candidate,
                )
        elif self.normalize:
            candidate = normalize_candidate(candidate)

        extracted_json_span: str | None = None
        candidate_str = str(candidate)

        # 0. Min / Max Length Checks
        if self.min_length is not None and len(candidate_str) < self.min_length:
            err = f"Artifact length {len(candidate_str)} is less than minimum required length {self.min_length}"
            return self._fail(
                err,
                ASTFailureCode.MIN_LENGTH_VIOLATION,
                t0,
                {"actual_length": len(candidate_str), "min_length": self.min_length},
                candidate=orig_candidate,
            )

        if self.max_length is not None and len(candidate_str) > self.max_length:
            err = f"Artifact length {len(candidate_str)} exceeds maximum allowed length {self.max_length}"
            return self._fail(
                err,
                ASTFailureCode.MAX_LENGTH_VIOLATION,
                t0,
                {"actual_length": len(candidate_str), "max_length": self.max_length},
                candidate=orig_candidate,
            )

        # 0b. JSON Schema Check
        if self.json_schema is not None:
            parsed_json: Any = None
            prefer_obj = not (
                isinstance(self.json_schema, dict)
                and self.json_schema.get("type") in ("array", "string", "integer", "number", "boolean", "null")
            )
            if isinstance(candidate, (dict, list)):
                parsed_json = candidate
            elif isinstance(candidate, BaseModel):
                parsed_json = (
                    candidate
                    if (
                        isinstance(self.json_schema, type)
                        and issubclass(self.json_schema, BaseModel)
                        and isinstance(candidate, self.json_schema)
                    )
                    else candidate.model_dump()
                )
            elif isinstance(candidate, str):
                try:
                    if self.normalize:
                        parsed_json, extracted_json_span = _extract_json_with_span(
                            candidate, prefer_object=prefer_obj
                        )
                    else:
                        parsed_json = json.loads(candidate)
                except (json.JSONDecodeError, RecursionError) as e:
                    if isinstance(e, RecursionError):
                        err = "Invalid JSON syntax: Maximum JSON nesting depth exceeded"
                        return self._fail(err, ASTFailureCode.JSON_SYNTAX_ERROR, t0, candidate=orig_candidate)
                    err = f"Invalid JSON syntax: {e}"
                    return self._fail(
                        err,
                        ASTFailureCode.JSON_SYNTAX_ERROR,
                        t0,
                        {"lineno": e.lineno, "colno": e.colno},
                        candidate=orig_candidate,
                    )
            else:
                err = f"Expected JSON string, dict, or list; got {type(candidate).__name__}"
                return self._fail(err, ASTFailureCode.JSON_TYPE_ERROR, t0, candidate=orig_candidate)

            if isinstance(self.json_schema, type) and issubclass(self.json_schema, BaseModel):
                try:
                    self.json_schema.model_validate(parsed_json)
                except (ValidationError, ValueError, TypeError) as e:
                    err = f"JSON validation failed for schema {self.json_schema.__name__}: {e}"
                    return self._fail(
                        err,
                        ASTFailureCode.JSON_SCHEMA_VALIDATION_ERROR,
                        t0,
                        {"schema": self.json_schema.__name__},
                        candidate=orig_candidate,
                    )
            elif isinstance(self.json_schema, dict):
                type_map: dict[str, type | tuple[type, ...]] = {
                    "string": str,
                    "integer": int,
                    "number": (int, float),
                    "boolean": bool,
                    "array": list,
                    "object": dict,
                    "null": type(None),
                }

                def _matches_json_type(val: Any, expected_spec: Any) -> bool:
                    if isinstance(expected_spec, str):
                        if expected_spec not in type_map:
                            return True
                        if isinstance(val, bool) and expected_spec in ("integer", "number"):
                            return False
                        return isinstance(val, type_map[expected_spec])
                    if isinstance(expected_spec, (list, tuple)):
                        return any(_matches_json_type(val, item) for item in expected_spec)
                    return True

                is_standard_json_schema = any(
                    k in self.json_schema
                    for k in ("$schema", "properties", "required", "items", "additionalProperties")
                ) or (
                    set(self.json_schema.keys()) <= {"type", "title", "description", "enum"}
                    and isinstance(self.json_schema.get("type"), (str, list, tuple))
                )

                if is_standard_json_schema and "type" in self.json_schema:
                    top_type = self.json_schema["type"]
                    if not _matches_json_type(parsed_json, top_type):
                        err = f"Expected JSON {top_type} for schema validation, got {type(parsed_json).__name__}"
                        return self._fail(err, ASTFailureCode.JSON_TYPE_ERROR, t0, candidate=orig_candidate)
                    if top_type == "array" and isinstance(parsed_json, list):
                        items_spec = self.json_schema.get("items")
                        if isinstance(items_spec, dict) and "type" in items_spec:
                            item_type = items_spec["type"]
                            for idx_item, elem in enumerate(parsed_json):
                                if not _matches_json_type(elem, item_type):
                                    err = f"Array item at index {idx_item} has invalid type: expected {item_type}, got {type(elem).__name__}"
                                    return self._fail(
                                        err,
                                        ASTFailureCode.JSON_SCHEMA_VALIDATION_ERROR,
                                        t0,
                                        {"index": idx_item, "expected_type": item_type},
                                        candidate=orig_candidate,
                                    )
                elif not isinstance(parsed_json, dict):
                    err = f"Expected JSON object (dict) for schema validation, got {type(parsed_json).__name__}"
                    return self._fail(err, ASTFailureCode.JSON_TYPE_ERROR, t0, candidate=orig_candidate)

                if isinstance(parsed_json, dict):
                    required_keys = self.json_schema.get("required")
                    if isinstance(required_keys, (list, tuple, set)):
                        for req_key in required_keys:
                            if req_key not in parsed_json:
                                err = f"Missing required JSON key: '{req_key}'"
                                return self._fail(
                                    err,
                                    ASTFailureCode.JSON_SCHEMA_VALIDATION_ERROR,
                                    t0,
                                    {"missing_key": req_key},
                                    candidate=orig_candidate,
                                )

                    props = self.json_schema.get("properties")
                    if isinstance(props, dict):
                        for prop_name, prop_def in props.items():
                            if prop_name in parsed_json and isinstance(prop_def, dict) and "type" in prop_def:
                                expected_type_str = prop_def["type"]
                                val = parsed_json[prop_name]
                                if not _matches_json_type(val, expected_type_str):
                                    err = f"Property '{prop_name}' has invalid type: expected {expected_type_str}, got {type(val).__name__}"
                                    return self._fail(
                                        err,
                                        ASTFailureCode.JSON_SCHEMA_VALIDATION_ERROR,
                                        t0,
                                        {"property": prop_name, "expected_type": expected_type_str},
                                        candidate=orig_candidate,
                                    )
                    elif not is_standard_json_schema:
                        for key, expected in self.json_schema.items():
                            if key not in parsed_json:
                                err = f"Missing required JSON key: '{key}'"
                                return self._fail(
                                    err,
                                    ASTFailureCode.JSON_SCHEMA_VALIDATION_ERROR,
                                    t0,
                                    {"missing_key": key},
                                    candidate=orig_candidate,
                                )
                            val = parsed_json[key]
                            if isinstance(expected, str) and expected in type_map:
                                expected_type = type_map[expected]
                                is_bool_for_num = isinstance(val, bool) and expected in ("integer", "number")
                                if not isinstance(val, expected_type) or is_bool_for_num:
                                    err = f"Key '{key}' expected type {expected}, got {type(val).__name__}"
                                    return self._fail(
                                        err,
                                        ASTFailureCode.JSON_SCHEMA_VALIDATION_ERROR,
                                        t0,
                                        {"property": key, "expected_type": expected},
                                        candidate=orig_candidate,
                                    )
                            elif isinstance(expected, type) and (
                                not isinstance(val, expected)
                                or (isinstance(val, bool) and expected in (int, float))
                            ):
                                err = f"Key '{key}' expected type {expected.__name__}, got {type(val).__name__}"
                                return self._fail(
                                    err,
                                    ASTFailureCode.JSON_SCHEMA_VALIDATION_ERROR,
                                    t0,
                                    {"property": key, "expected_type": expected.__name__},
                                    candidate=orig_candidate,
                                )

        # 1. Syntax Check
        parsed_ast: ast.AST | None = None
        if self.syntax or self.forbidden_imports:
            try:
                parsed_ast = ast.parse(candidate_str)
            except SyntaxError as e:
                if self.syntax:
                    err = f"Syntax error at line {e.lineno}, offset {e.offset}: {e.msg}"
                    return self._fail(
                        err,
                        ASTFailureCode.SYNTAX_ERROR,
                        t0,
                        {"lineno": e.lineno, "offset": e.offset, "msg": e.msg},
                        candidate=orig_candidate,
                    )
            except (ValueError, RecursionError, MemoryError) as e:
                if self.syntax:
                    err = f"Syntax error: {type(e).__name__}: {e}"
                    return self._fail(
                        err,
                        ASTFailureCode.SYNTAX_ERROR,
                        t0,
                        {"msg": str(e)},
                        candidate=orig_candidate,
                    )

        # 2. Forbidden Imports
        if self.forbidden_imports:
            found_imports = self._extract_imports_from_source(candidate_str, parsed_ast=parsed_ast)
            if self.normalize is True and raw_text_without_think and raw_text_without_think != candidate_str:
                for match in _FENCED_BLOCK_RE.finditer(raw_text_without_think):
                    found_imports |= self._extract_imports_from_source(match.group(1).strip())
                for match in _INLINE_FENCE_RE.finditer(raw_text_without_think):
                    found_imports |= self._extract_imports_from_source(match.group(1).strip())

            for imp in sorted(found_imports):
                for forbidden in self.forbidden_imports:
                    if imp == forbidden or imp.startswith(f"{forbidden}."):
                        err = f"Forbidden import detected: '{imp}'"
                        return self._fail(
                            err,
                            ASTFailureCode.FORBIDDEN_IMPORT,
                            t0,
                            {"forbidden_import": imp},
                            candidate=orig_candidate,
                        )

        # 3. Forbidden Patterns
        pattern_haystack = (
            f"{candidate_str}\n{raw_text_without_think}"
            if (self.normalize is True and raw_text_without_think and raw_text_without_think != candidate_str)
            else candidate_str
        )
        for pattern, compiled in self._forbidden_res:
            matched = bool(compiled.search(pattern_haystack)) if compiled else pattern in pattern_haystack
            if matched:
                err = f"Forbidden pattern matched: '{pattern}'"
                return self._fail(
                    err,
                    ASTFailureCode.FORBIDDEN_PATTERN,
                    t0,
                    {"pattern": pattern},
                    candidate=orig_candidate,
                )

        # 4. Required Sections (literal match first: headers like "## Security (v2)" are not
        # meant as regexes, and a literal hit is cheaper than a regex search anyway)
        section_haystack = (
            f"{candidate_str}\n{raw_text_without_think}"
            if (self.normalize and raw_text_without_think)
            else candidate_str
        )
        for section, compiled in self._required_res:
            has_section = section in section_haystack or bool(compiled and compiled.search(section_haystack))
            if not has_section:
                err = f"Missing required section: '{section}'"
                return self._fail(
                    err,
                    ASTFailureCode.MISSING_SECTION,
                    t0,
                    {"missing_section": section},
                    candidate=orig_candidate,
                )

        # 5. Acyclic DAG Check
        if self.acyclic_dag:
            tasks_data: list[dict[str, Any]] | None = None
            if isinstance(candidate, list):
                tasks_data = candidate
            elif isinstance(candidate, dict):
                tasks_data = candidate["tasks"] if "tasks" in candidate else candidate.get("nodes")
            elif isinstance(candidate, str):
                try:
                    if self.normalize:
                        loaded, span = _extract_json_with_span(candidate)
                        if extracted_json_span is None:
                            extracted_json_span = span
                    else:
                        loaded = json.loads(candidate)
                    if isinstance(loaded, list):
                        tasks_data = loaded
                    elif isinstance(loaded, dict):
                        tasks_data = loaded["tasks"] if "tasks" in loaded else loaded.get("nodes")
                except (json.JSONDecodeError, RecursionError):
                    pass

            if tasks_data is None:
                return self._fail(
                    "Candidate does not contain a valid task list ('tasks' or 'nodes') for DAG verification",
                    ASTFailureCode.DAG_CYCLE_DETECTED,
                    t0,
                    candidate=orig_candidate,
                )
            dag_err = _detect_dag_cycle(tasks_data)
            if dag_err:
                return self._fail(
                    dag_err,
                    ASTFailureCode.DAG_CYCLE_DETECTED,
                    t0,
                    candidate=orig_candidate,
                )

        # 6. Custom / BYO Validators
        if self.validators:
            for val_fn in self.validators:
                val_ok, val_msg, val_span = self._run_validator(
                    val_fn, candidate, normalized=bool(self.normalize)
                )
                if val_span is not None and extracted_json_span is None:
                    extracted_json_span = val_span
                if not val_ok:
                    return self._fail(
                        val_msg,
                        ASTFailureCode.CUSTOM_VALIDATOR_FAILED,
                        t0,
                        candidate=orig_candidate,
                    )

        if isinstance(norm_out := kwargs.get("_normalized_out"), list):
            if (
                self.normalize is True
                and raw_text_without_think is not None
                and self.required_sections
                and not self.syntax
                and self.json_schema is None
                and not self.acyclic_dag
                and not self.validators
            ):
                norm_out.append(raw_text_without_think.strip())
            elif extracted_json_span is not None and isinstance(candidate, str):
                norm_out.append(extracted_json_span)
            else:
                norm_out.append(candidate)

        latency = (time.perf_counter() - t0) * 1000.0
        trace = LayerTrace(
            layer_name=self.name,
            status=DecisionStatus.PASSED,
            latency_ms=latency,
        )
        return Verdict(
            passed=True,
            status=DecisionStatus.PASSED,
            triggered_layer=self.name,
            total_latency_ms=latency,
            total_tokens_generated=0,
            reason="All deterministic AST and contract checks passed.",
            traces=[trace],
        )

    async def aevaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        """Asynchronously evaluates candidate artifact (direct sync call for CPU AST validation)."""
        return self.evaluate(candidate, **kwargs)


# Alias
ContractGate = ASTGate
