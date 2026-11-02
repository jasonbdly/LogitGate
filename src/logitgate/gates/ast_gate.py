from __future__ import annotations

import ast
import json
import re
import time
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel, ValidationError

from logitgate.gates.base import BaseGate
from logitgate.models import DecisionStatus, LayerTrace, Verdict


def _detect_dag_cycle(tasks: list[dict[str, Any]]) -> str | None:
    """
    Validates that task dependencies exist and contain no cycles.
    Returns error message if invalid, None if acyclic and all dependencies exist.
    """
    task_ids = set()
    for t in tasks:
        tid = t.get("id") or t.get("task_id")
        if tid:
            task_ids.add(str(tid))

    adj: dict[str, list[str]] = {tid: [] for tid in task_ids}
    in_degree: dict[str, int] = {tid: 0 for tid in task_ids}

    for t in tasks:
        tid = str(t.get("id") or t.get("task_id"))
        deps = t.get("dependencies") or t.get("deps") or []
        for dep in deps:
            dep_str = str(dep)
            if dep_str not in task_ids:
                return f"Missing dependency: task '{tid}' depends on non-existent task '{dep_str}'"
            adj[dep_str].append(tid)
            in_degree[tid] += 1

    # Kahn's algorithm
    queue = [tid for tid, deg in in_degree.items() if deg == 0]
    visited_count = 0

    while queue:
        curr = queue.pop(0)
        visited_count += 1
        for neighbor in adj.get(curr, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if visited_count != len(task_ids):
        unresolved = [tid for tid, deg in in_degree.items() if deg > 0]
        return f"Cycle detected in task dependencies among: {unresolved}"
    return None


class ASTGate(BaseGate):
    """Deterministic zero-token verification gate (Layer 0).

    Executes in under 0.1ms using native CPU parsing without invoking any LLM.
    Supports Python AST syntax validation, forbidden import analysis, regex pattern
    matching, acyclic DAG dependency verification, and custom callable validators.

    Args:
        syntax: When True, parses the candidate as Python code via `ast.parse`.
        forbidden_imports: Sequence of module names prohibited from being imported.
        forbidden_patterns: Sequence of regular expressions prohibited in the artifact.
        required_sections: Sequence of string headers or regex patterns required to be present.
        acyclic_dag: When True, interprets candidate as a list/dict of task nodes and ensures
            all dependencies exist and form an acyclic directed graph (DAG).
        validators: Sequence of custom callables `fn(candidate) -> bool | (bool, str)`.
        name: Identifier for this gate instance.
    """

    def __init__(
        self,
        syntax: bool = False,
        forbidden_imports: Sequence[str] | None = None,
        forbidden_patterns: Sequence[str] | None = None,
        required_sections: Sequence[str] | None = None,
        acyclic_dag: bool = False,
        json_schema: dict[str, Any] | type[BaseModel] | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        validators: Sequence[Callable[[Any], bool | tuple[bool, str]]] | None = None,
        name: str = "ASTGate",
    ) -> None:
        self.syntax = syntax
        self.forbidden_imports = list(forbidden_imports) if forbidden_imports else None
        self.forbidden_patterns = list(forbidden_patterns) if forbidden_patterns else None
        self.required_sections = list(required_sections) if required_sections else None
        self.acyclic_dag = acyclic_dag
        self.json_schema = json_schema
        self.min_length = min_length
        self.max_length = max_length
        self.validators = list(validators) if validators else None
        self.name = name

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
        feat_str = f", {', '.join(features)}" if features else ""
        return f"ASTGate(name='{self.name}'{feat_str})"

    def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        t0 = time.perf_counter()
        candidate_str = str(candidate)

        # 0. Min / Max Length Checks
        if self.min_length is not None and len(candidate_str) < self.min_length:
            latency = (time.perf_counter() - t0) * 1000.0
            err = f"Artifact length {len(candidate_str)} is less than minimum required length {self.min_length}"
            trace = LayerTrace(
                layer_name=self.name,
                status=DecisionStatus.FAILED,
                latency_ms=latency,
                diagnostic_report=err,
            )
            return Verdict(
                passed=False,
                status=DecisionStatus.FAILED,
                triggered_layer=self.name,
                total_latency_ms=latency,
                total_tokens_generated=0,
                reason=err,
                traces=[trace],
            )

        if self.max_length is not None and len(candidate_str) > self.max_length:
            latency = (time.perf_counter() - t0) * 1000.0
            err = f"Artifact length {len(candidate_str)} exceeds maximum allowed length {self.max_length}"
            trace = LayerTrace(
                layer_name=self.name,
                status=DecisionStatus.FAILED,
                latency_ms=latency,
                diagnostic_report=err,
            )
            return Verdict(
                passed=False,
                status=DecisionStatus.FAILED,
                triggered_layer=self.name,
                total_latency_ms=latency,
                total_tokens_generated=0,
                reason=err,
                traces=[trace],
            )

        # 0b. JSON Schema Check
        if self.json_schema is not None:
            parsed_json: Any = None
            if isinstance(candidate, (dict, list)):
                parsed_json = candidate
            elif isinstance(candidate, str):
                try:
                    parsed_json = json.loads(candidate)
                except json.JSONDecodeError as e:
                    latency = (time.perf_counter() - t0) * 1000.0
                    err = f"Invalid JSON syntax: {e}"
                    trace = LayerTrace(
                        layer_name=self.name,
                        status=DecisionStatus.FAILED,
                        latency_ms=latency,
                        diagnostic_report=err,
                    )
                    return Verdict(
                        passed=False,
                        status=DecisionStatus.FAILED,
                        triggered_layer=self.name,
                        total_latency_ms=latency,
                        total_tokens_generated=0,
                        reason=err,
                        traces=[trace],
                    )
            else:
                latency = (time.perf_counter() - t0) * 1000.0
                err = f"Expected JSON string, dict, or list; got {type(candidate).__name__}"
                trace = LayerTrace(
                    layer_name=self.name,
                    status=DecisionStatus.FAILED,
                    latency_ms=latency,
                    diagnostic_report=err,
                )
                return Verdict(
                    passed=False,
                    status=DecisionStatus.FAILED,
                    triggered_layer=self.name,
                    total_latency_ms=latency,
                    total_tokens_generated=0,
                    reason=err,
                    traces=[trace],
                )

            if isinstance(self.json_schema, type) and issubclass(self.json_schema, BaseModel):
                try:
                    if isinstance(parsed_json, dict):
                        self.json_schema.model_validate(parsed_json)
                    else:
                        self.json_schema.model_validate_json(candidate_str)
                except (ValidationError, ValueError, TypeError) as e:
                    latency = (time.perf_counter() - t0) * 1000.0
                    err = f"JSON validation failed for schema {self.json_schema.__name__}: {e}"
                    trace = LayerTrace(
                        layer_name=self.name,
                        status=DecisionStatus.FAILED,
                        latency_ms=latency,
                        diagnostic_report=err,
                    )
                    return Verdict(
                        passed=False,
                        status=DecisionStatus.FAILED,
                        triggered_layer=self.name,
                        total_latency_ms=latency,
                        total_tokens_generated=0,
                        reason=err,
                        traces=[trace],
                    )
            elif isinstance(self.json_schema, dict):
                if not isinstance(parsed_json, dict):
                    latency = (time.perf_counter() - t0) * 1000.0
                    err = f"Expected JSON object (dict) for schema validation, got {type(parsed_json).__name__}"
                    trace = LayerTrace(
                        layer_name=self.name,
                        status=DecisionStatus.FAILED,
                        latency_ms=latency,
                        diagnostic_report=err,
                    )
                    return Verdict(
                        passed=False,
                        status=DecisionStatus.FAILED,
                        triggered_layer=self.name,
                        total_latency_ms=latency,
                        total_tokens_generated=0,
                        reason=err,
                        traces=[trace],
                    )

                required_keys = self.json_schema.get("required")
                if isinstance(required_keys, (list, tuple, set)):
                    for req_key in required_keys:
                        if req_key not in parsed_json:
                            latency = (time.perf_counter() - t0) * 1000.0
                            err = f"Missing required JSON key: '{req_key}'"
                            trace = LayerTrace(
                                layer_name=self.name,
                                status=DecisionStatus.FAILED,
                                latency_ms=latency,
                                diagnostic_report=err,
                            )
                            return Verdict(
                                passed=False,
                                status=DecisionStatus.FAILED,
                                triggered_layer=self.name,
                                total_latency_ms=latency,
                                total_tokens_generated=0,
                                reason=err,
                                traces=[trace],
                            )

                props = self.json_schema.get("properties")
                if isinstance(props, dict):
                    type_map: dict[str, type | tuple[type, ...]] = {
                        "string": str,
                        "integer": int,
                        "number": (int, float),
                        "boolean": bool,
                        "array": list,
                        "object": dict,
                    }
                    for prop_name, prop_def in props.items():
                        if prop_name in parsed_json and isinstance(prop_def, dict) and "type" in prop_def:
                            expected_type_str = prop_def["type"]
                            expected_type = type_map.get(expected_type_str)
                            if expected_type and not isinstance(parsed_json[prop_name], expected_type):
                                latency = (time.perf_counter() - t0) * 1000.0
                                err = f"Property '{prop_name}' has invalid type: expected {expected_type_str}, got {type(parsed_json[prop_name]).__name__}"
                                trace = LayerTrace(
                                    layer_name=self.name,
                                    status=DecisionStatus.FAILED,
                                    latency_ms=latency,
                                    diagnostic_report=err,
                                )
                                return Verdict(
                                    passed=False,
                                    status=DecisionStatus.FAILED,
                                    triggered_layer=self.name,
                                    total_latency_ms=latency,
                                    total_tokens_generated=0,
                                    reason=err,
                                    traces=[trace],
                                )
                elif "required" not in self.json_schema:
                    for key, expected in self.json_schema.items():
                        if key not in parsed_json:
                            latency = (time.perf_counter() - t0) * 1000.0
                            err = f"Missing required JSON key: '{key}'"
                            trace = LayerTrace(
                                layer_name=self.name,
                                status=DecisionStatus.FAILED,
                                latency_ms=latency,
                                diagnostic_report=err,
                            )
                            return Verdict(
                                passed=False,
                                status=DecisionStatus.FAILED,
                                triggered_layer=self.name,
                                total_latency_ms=latency,
                                total_tokens_generated=0,
                                reason=err,
                                traces=[trace],
                            )
                        if isinstance(expected, type) and not isinstance(parsed_json[key], expected):
                            latency = (time.perf_counter() - t0) * 1000.0
                            err = f"Key '{key}' expected type {expected.__name__}, got {type(parsed_json[key]).__name__}"
                            trace = LayerTrace(
                                layer_name=self.name,
                                status=DecisionStatus.FAILED,
                                latency_ms=latency,
                                diagnostic_report=err,
                            )
                            return Verdict(
                                passed=False,
                                status=DecisionStatus.FAILED,
                                triggered_layer=self.name,
                                total_latency_ms=latency,
                                total_tokens_generated=0,
                                reason=err,
                                traces=[trace],
                            )

        # 1. Syntax Check
        parsed_ast: ast.AST | None = None
        if self.syntax or self.forbidden_imports:
            try:
                parsed_ast = ast.parse(candidate_str)
            except SyntaxError as e:
                if self.syntax:
                    latency = (time.perf_counter() - t0) * 1000.0
                    err = f"Syntax error at line {e.lineno}, offset {e.offset}: {e.msg}"
                    trace = LayerTrace(
                        layer_name=self.name,
                        status=DecisionStatus.FAILED,
                        latency_ms=latency,
                        diagnostic_report=err,
                    )
                    return Verdict(
                        passed=False,
                        status=DecisionStatus.FAILED,
                        triggered_layer=self.name,
                        total_latency_ms=latency,
                        total_tokens_generated=0,
                        reason=err,
                        traces=[trace],
                    )

        # 2. Forbidden Imports
        if self.forbidden_imports:
            found_imports: set[str] = set()
            if parsed_ast is not None:
                for node in ast.walk(parsed_ast):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            found_imports.add(alias.name)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        found_imports.add(node.module)
            else:
                # Regex fallback if AST not parsed
                for match in re.finditer(r"^\s*(?:import|from)\s+([a-zA-Z0-9_\.]+)", candidate_str, re.MULTILINE):
                    found_imports.add(match.group(1))

            for imp in found_imports:
                for forbidden in self.forbidden_imports:
                    if imp == forbidden or imp.startswith(f"{forbidden}."):
                        latency = (time.perf_counter() - t0) * 1000.0
                        err = f"Forbidden import detected: '{imp}'"
                        trace = LayerTrace(
                            layer_name=self.name,
                            status=DecisionStatus.FAILED,
                            latency_ms=latency,
                            diagnostic_report=err,
                        )
                        return Verdict(
                            passed=False,
                            status=DecisionStatus.FAILED,
                            triggered_layer=self.name,
                            total_latency_ms=latency,
                            total_tokens_generated=0,
                            reason=err,
                            traces=[trace],
                        )

        # 3. Forbidden Patterns
        if self.forbidden_patterns:
            for pattern in self.forbidden_patterns:
                if re.search(pattern, candidate_str):
                    latency = (time.perf_counter() - t0) * 1000.0
                    err = f"Forbidden pattern matched: '{pattern}'"
                    trace = LayerTrace(
                        layer_name=self.name,
                        status=DecisionStatus.FAILED,
                        latency_ms=latency,
                        diagnostic_report=err,
                    )
                    return Verdict(
                        passed=False,
                        status=DecisionStatus.FAILED,
                        triggered_layer=self.name,
                        total_latency_ms=latency,
                        total_tokens_generated=0,
                        reason=err,
                        traces=[trace],
                    )

        # 4. Required Sections
        if self.required_sections:
            for section in self.required_sections:
                if section not in candidate_str and not re.search(section, candidate_str):
                    latency = (time.perf_counter() - t0) * 1000.0
                    err = f"Missing required section: '{section}'"
                    trace = LayerTrace(
                        layer_name=self.name,
                        status=DecisionStatus.FAILED,
                        latency_ms=latency,
                        diagnostic_report=err,
                    )
                    return Verdict(
                        passed=False,
                        status=DecisionStatus.FAILED,
                        triggered_layer=self.name,
                        total_latency_ms=latency,
                        total_tokens_generated=0,
                        reason=err,
                        traces=[trace],
                    )

        # 5. Acyclic DAG Check
        if self.acyclic_dag:
            tasks_data: list[dict[str, Any]] | None = None
            if isinstance(candidate, list):
                tasks_data = candidate
            elif isinstance(candidate, dict):
                tasks_data = candidate.get("tasks") or candidate.get("nodes")
            elif isinstance(candidate, str):
                try:
                    loaded = json.loads(candidate)
                    if isinstance(loaded, list):
                        tasks_data = loaded
                    elif isinstance(loaded, dict):
                        tasks_data = loaded.get("tasks") or loaded.get("nodes")
                except json.JSONDecodeError:
                    pass

            if tasks_data is not None:
                dag_err = _detect_dag_cycle(tasks_data)
                if dag_err:
                    latency = (time.perf_counter() - t0) * 1000.0
                    trace = LayerTrace(
                        layer_name=self.name,
                        status=DecisionStatus.FAILED,
                        latency_ms=latency,
                        diagnostic_report=dag_err,
                    )
                    return Verdict(
                        passed=False,
                        status=DecisionStatus.FAILED,
                        triggered_layer=self.name,
                        total_latency_ms=latency,
                        total_tokens_generated=0,
                        reason=dag_err,
                        traces=[trace],
                    )

        # 6. Custom Validators
        if self.validators:
            for val_fn in self.validators:
                try:
                    res = val_fn(candidate)
                    if isinstance(res, tuple):
                        val_ok, val_msg = res
                    else:
                        val_ok, val_msg = bool(res), f"Validator '{getattr(val_fn, '__name__', str(val_fn))}' rejected artifact"
                except (ValueError, TypeError, AssertionError, KeyError, IndexError, AttributeError, RuntimeError, OSError) as ex:
                    val_ok = False
                    val_msg = f"Validator raised exception: {ex}"

                if not val_ok:
                    latency = (time.perf_counter() - t0) * 1000.0
                    trace = LayerTrace(
                        layer_name=self.name,
                        status=DecisionStatus.FAILED,
                        latency_ms=latency,
                        diagnostic_report=val_msg,
                    )
                    return Verdict(
                        passed=False,
                        status=DecisionStatus.FAILED,
                        triggered_layer=self.name,
                        total_latency_ms=latency,
                        total_tokens_generated=0,
                        reason=val_msg,
                        traces=[trace],
                    )

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
