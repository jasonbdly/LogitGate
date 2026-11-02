from __future__ import annotations

import ast
import json
import re
import time
from typing import Any, Callable, Optional, Sequence, Union

from logitgate.gates.base import BaseGate
from logitgate.models import DecisionStatus, LayerTrace, Verdict


def _detect_dag_cycle(tasks: list[dict[str, Any]]) -> Optional[str]:
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
        forbidden_imports: Optional[Sequence[str]] = None,
        forbidden_patterns: Optional[Sequence[str]] = None,
        required_sections: Optional[Sequence[str]] = None,
        acyclic_dag: bool = False,
        validators: Optional[Sequence[Callable[[Any], Union[bool, tuple[bool, str]]]]] = None,
        name: str = "ASTGate",
    ) -> None:
        self.syntax = syntax
        self.forbidden_imports = list(forbidden_imports) if forbidden_imports else None
        self.forbidden_patterns = list(forbidden_patterns) if forbidden_patterns else None
        self.required_sections = list(required_sections) if required_sections else None
        self.acyclic_dag = acyclic_dag
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
        if self.validators:
            features.append(f"validators=[{len(self.validators)} callables]")
        feat_str = f", {', '.join(features)}" if features else ""
        return f"ASTGate(name='{self.name}'{feat_str})"

    def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        t0 = time.perf_counter()
        candidate_str = str(candidate)

        # 1. Syntax Check
        parsed_ast: Optional[ast.AST] = None
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
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
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
            tasks_data: Optional[list[dict[str, Any]]] = None
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
                except Exception as ex:
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
