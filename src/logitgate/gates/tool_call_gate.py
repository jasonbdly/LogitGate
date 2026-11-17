from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import json
import posixpath
import re
import time
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Any, Literal

from logitgate.exceptions import ConfigurationError
from logitgate.gates.base import (
    TIER_DETERMINISTIC,
    TIER_LOGIT,
    BaseGate,
    _safe_aclose,
    _safe_close,
)
from logitgate.models import DecisionStatus, LayerTrace, Verdict

if TYPE_CHECKING:
    from logitgate.backends.base import LogitBackend
    from logitgate.gates.logit_gate import LogitGate

_PATH_KEYS = {"path", "file", "filepath", "file_path", "filename", "dir", "directory", "dir_path", "target_path", "source_path", "dest", "destination", "cwd", "root", "output_path", "target_dir"}
_URL_KEYS = {"url", "uri", "endpoint", "webhook", "callback", "callback_url", "target_url", "href", "base_url", "hostname"}
_CMD_KEYS = {"command", "cmd", "shell", "bash", "script", "args", "command_line"}
_SQL_KEYS = {"sql", "query", "statement", "operation"}
_BLOCKED_HOSTS = {"localhost", "metadata.google.internal", "metadata", "instance-data"}

_DESTRUCTIVE_CMD_PATS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE) for p in (
        r"\brm\s+-[a-zA-Z]*[rf][a-zA-Z]*\s+[/~]", r"\brm\s+.*--no-preserve-root",
        r"\b(mkfs|mkfs\.[a-zA-Z0-9]+)\b", r"\bdd\s+if=", r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",
        r"\bchmod\s+(-[a-zA-Z]*R[a-zA-Z]*\s+)?777\s+[/~]", r"(curl|wget)\s+[^|;&]+\|\s*(ba)?sh",
        r"/dev/tcp/", r"\bnc\s+.*-e\b", r"\b(shutdown|reboot|poweroff|halt)\b",
        r">\s*/dev/(sd[a-z]|nvme[0-9]|zero|null|urandom|mem|kmem)",
    )
]
_CHAINED_DANGEROUS_PATS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE) for p in (r"\b(rm|curl|wget)\s+", r"\bcat\s+/etc/", r"\b(env|printenv)\b", r"\b(nmap|hydra|mkfs|dd)\b")
]
_DESTRUCTIVE_SQL_PATS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE) for p in (
        r"\bDROP\s+(TABLE|DATABASE|SCHEMA|ROLE|USER|VIEW)\b", r"\bTRUNCATE\s+(TABLE\s+)?[a-zA-Z0-9_\.]+",
        r"\bALTER\s+(USER|ROLE|DATABASE)\b", r"\bGRANT\s+ALL\b", r"\bINTO\s+OUTFILE\b",
        r"\bCOPY\s+.*\s+FROM\s+PROGRAM\b", r"\bxp_cmdshell\b", r";\s*(DROP|DELETE|UPDATE|INSERT|TRUNCATE|ALTER)\b",
        r"\bWHERE\s+1\s*=\s*1\b", r"\bOR\s+1\s*=\s*1\b", r"\bOR\s+['\"]1['\"]\s*=\s*['\"]1['\"]\b",
    )
]
_DEFAULT_INTENT_RUBRIC = (
    "Does the proposed tool call safely, accurately, and faithfully fulfill "
    "the user request without unauthorized actions, malicious payloads, or unintended side effects?"
)


@dataclass(frozen=True, slots=True)
class NormalizedToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str | None = None
    source_format: Literal[
        "openai_tool_call",
        "openai_function_call",
        "mcp_jsonrpc",
        "anthropic_tool_use",
        "raw_dict",
        "transcript",
    ] = "raw_dict"
    extracted_user_intent: str | None = None


class ToolCallFailureCode(str, Enum):
    MALFORMED_ENVELOPE = "malformed_envelope"
    INVALID_ARGUMENTS_JSON = "invalid_arguments_json"
    DISALLOWED_TOOL = "disallowed_tool"
    SCHEMA_VIOLATION = "schema_violation"
    PATH_TRAVERSAL = "path_traversal"
    SSRF_BLOCKED = "ssrf_blocked"
    DESTRUCTIVE_COMMAND = "destructive_command"
    SQL_INJECTION_OR_DESTRUCTIVE = "sql_injection_or_destructive"
    CUSTOM_POLICY_FAILED = "custom_policy_failed"


class ToolCallNormalizationError(Exception):
    def __init__(self, code: ToolCallFailureCode, message: str) -> None:
        super().__init__(message)
        self.code, self.message = code, message


def _looks_like_cmd(tool_name: str) -> bool:
    return any(tok in tool_name.lower() for tok in ("command", "bash", "shell", "exec", "terminal", "run_cmd", "execute"))


def _looks_like_sql_tool(tool_name: str) -> bool:
    return any(tok in tool_name.lower() for tok in ("sql", "db", "query", "postgres", "mysql", "sqlite", "table"))


def _looks_like_sql_val(val: str) -> bool:
    return val.strip().upper().startswith(("SELECT", "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "EXPLAIN", "VACUUM", "GRANT"))


def _check_path_security(val: str, allowed_roots: list[str] | None) -> tuple[bool, str]:
    if "\x00" in val:
        return False, f"Null byte detected in path: {val!r}"
    unquoted = urllib.parse.unquote(val)
    if "\x00" in unquoted:
        return False, f"Null byte detected in unquoted path: {val!r}"
    if any(seg == ".." for seg in PurePosixPath(unquoted).parts + PureWindowsPath(unquoted).parts) or "/../" in unquoted or unquoted.startswith("../") or unquoted.endswith("/..") or unquoted == "..":
        return False, f"Path traversal sequence '..' detected in path: {val!r}"

    norm = posixpath.normpath(unquoted)
    if (norm == "/etc" or norm.startswith(("/etc/", "/root", "/proc", "/sys", "/dev", "/boot"))) and (
        allowed_roots is None or not any(norm == r or norm.startswith(r.rstrip("/") + "/") for r in allowed_roots)
    ):
        return False, f"Access to sensitive system path is blocked: {val!r}"
    if any(tok in norm for tok in (".ssh", ".aws/credentials", ".kube/config")):
        return False, f"Access to sensitive credential path is blocked: {val!r}"
    if allowed_roots is not None and not any(norm == r or norm.startswith(r.rstrip("/") + "/") for r in allowed_roots):
        return False, f"Path '{val}' resolves to '{norm}', which is outside allowed roots: {allowed_roots!r}"
    return True, ""


def _check_ssrf_security(val: str, allowed_hosts: list[str] | None) -> tuple[bool, str]:
    if "://" in val or val.lower().startswith("javascript:"):
        parsed = urllib.parse.urlparse(val)
        if parsed.scheme.lower() not in ("http", "https"):
            return False, f"Disallowed URI scheme '{parsed.scheme}': only HTTP/HTTPS are permitted"
        raw_host, path = parsed.hostname or "", parsed.path
    else:
        raw_host, path = val.split("/")[0].split(":")[0].strip(), ""

    host = raw_host.lower().strip()
    if not host:
        return False, f"Missing or empty host in URL: {val!r}"
    if allowed_hosts is not None:
        if not any((host == ah.lstrip(".") or host.endswith("." + ah.lstrip("."))) if ah.startswith(".") else (host == ah) for ah in allowed_hosts):
            return False, f"Host '{host}' is not in allowed_hosts: {allowed_hosts!r}"
        return True, ""
    if host in _BLOCKED_HOSTS or host.endswith((".internal", ".local")):
        return False, f"SSRF blocked: prohibited host '{host}'"

    ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address | None = None
    try:
        ip_obj = ipaddress.ip_address(host)
    except ValueError:
        if host.isdigit() or (host.startswith("0x") and len(host) <= 10):
            with contextlib.suppress(ValueError):
                ip_obj = ipaddress.ip_address(int(host, 0))
    if ip_obj is not None and (ip_obj.is_loopback or ip_obj.is_link_local or ip_obj.is_private or ip_obj.is_unspecified or ip_obj.is_reserved or ip_obj.is_multicast):
        return False, f"SSRF blocked: non-public IP literal '{host}'"
    if "metadata" in path.lower() and ("meta-data" in path.lower() or "computemetadata" in path.lower()):
        return False, f"SSRF blocked: cloud metadata path in '{val}'"
    return True, ""


def _check_destructive_cmd(val: str, tool_name: str) -> tuple[bool, str]:
    if any(pat.search(val) for pat in _DESTRUCTIVE_CMD_PATS):
        return False, f"Destructive command detected in '{val}': matched pattern"
    if re.search(r"(?:;|&&|\|\||`|\$\()", val):
        if not _looks_like_cmd(tool_name):
            return False, f"Command chaining or substitution in non-shell tool '{tool_name}': {val!r}"
        if any(dpat.search(val) for dpat in _CHAINED_DANGEROUS_PATS):
            return False, f"Destructive or probing chained command detected in '{val}'"
    return True, ""


def _check_destructive_sql(val: str, tool_name: str) -> tuple[bool, str]:
    if any(pat.search(val) for pat in _DESTRUCTIVE_SQL_PATS):
        return False, f"Destructive or injection SQL pattern detected in '{val}'"
    if re.search(r"\bEXPLAIN\s+ANALYZE\s+(DELETE|UPDATE|INSERT|DROP|TRUNCATE)\b", val, re.IGNORECASE):
        return False, f"EXPLAIN ANALYZE with mutating statement detected in '{val}'"
    clean_sql = re.sub(r"/\*.*?\*/", "", re.sub(r"--.*$", "", val, flags=re.MULTILINE), flags=re.DOTALL)
    for stmt in (s.strip() for s in clean_sql.split(";") if s.strip()):
        s_upper = stmt.upper()
        if (re.search(r"\bDELETE\s+FROM\b", s_upper) or re.search(r"\bUPDATE\s+[a-zA-Z0-9_\.]+\s+SET\b", s_upper)) and not re.search(r"\bWHERE\b", s_upper):
            return False, f"Unqualified statement without WHERE clause: {stmt!r}"
    return True, ""


def _validate_json_schema(val: Any, schema: dict[str, Any], path: str = "root") -> None:
    t = schema.get("type")
    if t == "string" and not isinstance(val, str):
        raise ValueError(f"At {path}: expected string, got {type(val).__name__}")
    if t == "integer" and (isinstance(val, bool) or not isinstance(val, int)):
        raise ValueError(f"At {path}: expected integer, got {type(val).__name__}")
    if t == "number" and (isinstance(val, bool) or not isinstance(val, (int, float))):
        raise ValueError(f"At {path}: expected number, got {type(val).__name__}")
    if t == "boolean" and not isinstance(val, bool):
        raise ValueError(f"At {path}: expected boolean, got {type(val).__name__}")
    if t == "array":
        if not isinstance(val, list):
            raise ValueError(f"At {path}: expected array, got {type(val).__name__}")
        if isinstance(schema.get("items"), dict):
            for i, item in enumerate(val):
                _validate_json_schema(item, schema["items"], f"{path}[{i}]")
    elif t == "object" or "properties" in schema:
        if not isinstance(val, dict):
            raise ValueError(f"At {path}: expected object/dict, got {type(val).__name__}")
        for req in schema.get("required") or ():
            if req not in val:
                raise ValueError(f"At {path}: missing required property '{req}'")
        if schema.get("additionalProperties") is False:
            allowed = set((schema.get("properties") or {}).keys())
            for k in val:
                if k not in allowed:
                    raise ValueError(f"At {path}: additional property '{k}' is not allowed")
        for k, pval in val.items():
            if isinstance((schema.get("properties") or {}).get(k), dict):
                _validate_json_schema(pval, schema["properties"][k], f"{path}.{k}")
    if "enum" in schema and isinstance(schema["enum"], (list, tuple)) and val not in schema["enum"]:
        raise ValueError(f"At {path}: value {val!r} not in enum {schema['enum']!r}")


def _unpack_single_envelope(item: Any, default_source: str, extracted_intent: str | None) -> list[NormalizedToolCall]:
    if not isinstance(item, dict):
        raise ToolCallNormalizationError(ToolCallFailureCode.MALFORMED_ENVELOPE, f"Tool call item must be dict, got {type(item).__name__}")

    fn_obj = item.get("function") if item.get("type") == "function" else item.get("function_call")
    if isinstance(fn_obj, dict):
        name, raw_args, call_id = fn_obj.get("name"), fn_obj.get("arguments", {}), item.get("id")
        source = "openai_tool_call" if item.get("type") == "function" else "openai_function_call"
    elif item.get("type") == "function_call":
        name, raw_args, call_id = item.get("name"), item.get("arguments", {}), item.get("call_id") or item.get("id")
        source = "openai_function_call"
    elif item.get("method") == "tools/call" or (item.get("jsonrpc") == "2.0" and "method" in item):
        if item.get("method") != "tools/call" or not isinstance(item.get("params"), dict):
            raise ToolCallNormalizationError(ToolCallFailureCode.MALFORMED_ENVELOPE, "Malformed MCP JSON-RPC call")
        name, raw_args, call_id = item["params"].get("name"), item["params"].get("arguments", {}), str(item["id"]) if "id" in item else None
        source = "mcp_jsonrpc"
    elif item.get("type") == "tool_use":
        name, raw_args, call_id = item.get("name"), item.get("input", {}), item.get("id")
        source = "anthropic_tool_use"
    elif "name" in item:
        name, raw_args, call_id = item.get("name"), item.get("arguments", item.get("input", {})), item.get("id") or item.get("call_id")
        source = default_source if default_source in ("transcript", "mcp_jsonrpc", "openai_tool_call", "openai_function_call", "anthropic_tool_use") else "raw_dict"
    else:
        raise ToolCallNormalizationError(ToolCallFailureCode.MALFORMED_ENVELOPE, f"Unrecognized envelope: {list(item.keys())}")

    if not name or not isinstance(name, str):
        raise ToolCallNormalizationError(ToolCallFailureCode.MALFORMED_ENVELOPE, f"Tool name must be non-empty string, got {name!r}")

    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
        except Exception as exc:
            raise ToolCallNormalizationError(ToolCallFailureCode.INVALID_ARGUMENTS_JSON, f"Invalid arguments JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ToolCallNormalizationError(ToolCallFailureCode.INVALID_ARGUMENTS_JSON, f"Arguments must decode to dict, got {type(parsed).__name__}")
        arguments = parsed
    elif isinstance(raw_args, dict):
        arguments = raw_args
    elif raw_args is None:
        arguments = {}
    else:
        raise ToolCallNormalizationError(ToolCallFailureCode.INVALID_ARGUMENTS_JSON, f"Arguments must be dict or JSON str, got {type(raw_args).__name__}")

    src = "transcript" if default_source == "transcript" else source
    return [NormalizedToolCall(name=name, arguments=arguments, call_id=str(call_id) if call_id is not None else None, source_format=src, extracted_user_intent=extracted_intent)]  # type: ignore[arg-type]


def _unpack_raw_payload(payload: Any, default_source: str, extracted_intent: str | None) -> tuple[list[NormalizedToolCall], str | None]:
    if isinstance(payload, list):
        if not payload:
            raise ToolCallNormalizationError(ToolCallFailureCode.MALFORMED_ENVELOPE, "Tool calls list cannot be empty")
        return [c for elem in payload for c in _unpack_single_envelope(elem, default_source, extracted_intent)], extracted_intent

    if isinstance(payload, dict):
        if payload.get("choices") and isinstance(payload["choices"], list):
            msg = payload["choices"][0].get("message", {}) if isinstance(payload["choices"][0], dict) else {}
            if isinstance(msg.get("tool_calls"), list):
                return _unpack_raw_payload(msg["tool_calls"], "openai_tool_call", extracted_intent)
            if isinstance(msg.get("function_call"), dict):
                return _unpack_single_envelope(msg["function_call"], "openai_function_call", extracted_intent), extracted_intent
        if isinstance(payload.get("tool_calls"), list):
            return _unpack_raw_payload(payload["tool_calls"], "openai_tool_call", extracted_intent)
        if isinstance(payload.get("content"), list):
            t_items = [it for it in payload["content"] if isinstance(it, dict) and it.get("type") == "tool_use"]
            if t_items:
                return _unpack_raw_payload(t_items, "anthropic_tool_use", extracted_intent)
        return _unpack_single_envelope(payload, default_source, extracted_intent), extracted_intent

    raise ToolCallNormalizationError(ToolCallFailureCode.MALFORMED_ENVELOPE, f"Unexpected payload type: {type(payload).__name__}")


def normalize_tool_calls(candidate: Any) -> tuple[list[NormalizedToolCall], str | None]:
    if candidate is None:
        raise ToolCallNormalizationError(ToolCallFailureCode.MALFORMED_ENVELOPE, "Candidate cannot be None")
    if hasattr(candidate, "model_dump") and callable(candidate.model_dump):
        candidate = candidate.model_dump()
    if isinstance(candidate, str):
        text = candidate.strip()
        if "[TOOL CALL]" in text:
            parts = [p.strip() for p in text.split("[TOOL CALL]", 1)]
            intent_part, call_part = parts[0], parts[1]
            intent_text = None
            for pfx in ("[USER REQUEST]", "[USER INTENT]", "User:"):
                if pfx in intent_part:
                    intent_text = intent_part.split(pfx, 1)[1].strip()
                    break
            call_str = re.sub(r"^```[^\r\n]*\r?\n|```$", "", call_part, flags=re.DOTALL).strip() if call_part.startswith("```") else call_part
            try:
                parsed_json = json.loads(call_str)
            except Exception as exc:
                raise ToolCallNormalizationError(ToolCallFailureCode.MALFORMED_ENVELOPE, f"Failed to parse tool call JSON: {exc}") from exc
            return _unpack_raw_payload(parsed_json, "transcript", intent_text or (intent_part if intent_part else None))

        cleaned = re.sub(r"^```[^\r\n]*\r?\n|```$", "", text, flags=re.DOTALL).strip() if text.startswith("```") else text
        try:
            parsed_json = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ToolCallNormalizationError(ToolCallFailureCode.MALFORMED_ENVELOPE, f"String is neither tool transcript nor valid JSON: {exc}") from exc
        return _unpack_raw_payload(parsed_json, "raw_dict", None)

    return _unpack_raw_payload(candidate, "raw_dict", None)


def _validate_prob(param_name: str, val: float | None) -> None:
    if val is not None and (isinstance(val, bool) or not isinstance(val, (int, float)) or not (0.0 <= val <= 1.0)):
        raise ConfigurationError(f"{param_name} must be a float between 0.0 and 1.0, got {val!r}")


class ToolCallGate(BaseGate):
    name: str = "ToolCallGate"
    layer_tier: int = TIER_DETERMINISTIC
    can_fast_path: bool = False

    def __init__(
        self,
        *,
        allowed_tools: Sequence[str] | None = None,
        blocked_tools: Sequence[str] | None = None,
        tool_schemas: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]] | None = None,
        allowed_roots: Sequence[str] | None = None,
        allowed_hosts: Sequence[str] | None = None,
        block_path_traversal: bool = True,
        block_ssrf: bool = True,
        block_destructive_commands: bool = True,
        block_destructive_sql: bool = True,
        max_calls: int | None = None,
        custom_validators: Sequence[Callable[[NormalizedToolCall], bool | str | None]] | None = None,
        check_intent: bool = False,
        intent_rubric: str | None = None,
        user_intent: str | None = None,
        backend: str | LogitBackend | None = None,
        model: str | None = None,
        pass_threshold: float | None = None,
        fail_threshold: float | None = None,
        symmetric_debiasing: bool = True,
        on_unsupported_logprobs: Literal["raise", "escalate", "sidecar"] = "raise",
        sidecar_backend: str | LogitBackend | None = None,
        save_on_fail: bool | None = None,
        telemetry_sink: Any = None,
        cache: Any = None,
        name: str = "ToolCallGate",
    ) -> None:
        from logitgate.cache import resolve_verdict_cache
        from logitgate.telemetry.sinks import resolve_telemetry_sink

        if max_calls is not None and (isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls < 0):
            raise ConfigurationError(f"max_calls must be a non-negative integer or None, got {max_calls!r}")
        _validate_prob("pass_threshold", pass_threshold)
        _validate_prob("fail_threshold", fail_threshold)
        if pass_threshold is not None and fail_threshold is not None and pass_threshold < fail_threshold:
            raise ConfigurationError(f"pass_threshold ({pass_threshold}) cannot be less than fail_threshold ({fail_threshold})")

        resolved_sink = resolve_telemetry_sink(telemetry_sink)
        owns_sink = isinstance(telemetry_sink, str) or hasattr(telemetry_sink, "__fspath__")
        try:
            resolved_cache = resolve_verdict_cache(cache)
        except BaseException:
            if owns_sink and resolved_sink not in (None, False):
                resolved_sink.close()  # type: ignore[union-attr]
            raise

        self.allowed_tools = list(allowed_tools) if allowed_tools is not None else None
        self.blocked_tools = list(blocked_tools) if blocked_tools is not None else None
        self.tool_schemas, self.intent_rubric, self.user_intent, self.save_on_fail = tool_schemas, intent_rubric, user_intent, save_on_fail
        self.allowed_roots = [posixpath.normpath(r) for r in allowed_roots] if allowed_roots is not None else None
        self.allowed_hosts = [h.strip().lower() for h in allowed_hosts] if allowed_hosts is not None else None
        self.block_path_traversal, self.block_ssrf = block_path_traversal, block_ssrf
        self.block_destructive_commands, self.block_destructive_sql = block_destructive_commands, block_destructive_sql
        self.max_calls, self.custom_validators = max_calls, list(custom_validators) if custom_validators is not None else None
        self.check_intent = check_intent or (intent_rubric is not None)
        self.telemetry_sink, self.cache, self.name = resolved_sink, resolved_cache, name
        self._owns_sink, self._owns_cache, self._owns_backend, self._owns_logit_gate = owns_sink, isinstance(cache, (str, Path)), False, False
        self._logit_gate: LogitGate | None = None
        self._normalized_schemas: dict[str, dict[str, Any]] = {}
        self._tool_descriptions: dict[str, str] = {}

        if isinstance(tool_schemas, Mapping):
            for tname, s in tool_schemas.items():
                self._normalized_schemas[tname] = dict(s)
                if "description" in s:
                    self._tool_descriptions[tname] = str(s["description"])
        elif isinstance(tool_schemas, Sequence):
            for spec in tool_schemas:
                if isinstance(spec, Mapping):
                    fn = spec.get("function") if spec.get("type") == "function" and isinstance(spec.get("function"), Mapping) else None
                    tname = str((fn or spec).get("name", ""))
                    s = (fn.get("parameters") if fn else spec.get("inputSchema", spec.get("parameters"))) or {}
                    if tname:
                        self._normalized_schemas[tname] = dict(s) if isinstance(s, Mapping) else {}
                        if "description" in (fn or spec):
                            self._tool_descriptions[tname] = str((fn or spec)["description"])

        if self.check_intent:
            self.layer_tier, self.can_fast_path = TIER_LOGIT, True
            from logitgate.gates.logit_gate import LogitGate

            self._logit_gate = LogitGate(
                rubric=self.intent_rubric or _DEFAULT_INTENT_RUBRIC, backend=backend, model=model,
                pass_threshold=pass_threshold, fail_threshold=fail_threshold, symmetric_debiasing=symmetric_debiasing,
                on_unsupported_logprobs=on_unsupported_logprobs, sidecar_backend=sidecar_backend,
                save_on_fail=False, telemetry_sink=self.telemetry_sink, name=f"{self.name}:intent",
            )
            self._owns_logit_gate, self._gate = True, self._logit_gate

    def replay_config(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "gate_type": "ToolCallGate",
            "name": self.name,
            "block_path_traversal": self.block_path_traversal,
            "block_ssrf": self.block_ssrf,
            "block_destructive_commands": self.block_destructive_commands,
            "block_destructive_sql": self.block_destructive_sql,
            "check_intent": self.check_intent,
        }
        for k, v in (
            ("allowed_tools", self.allowed_tools),
            ("blocked_tools", self.blocked_tools),
            ("allowed_roots", self.allowed_roots),
            ("allowed_hosts", self.allowed_hosts),
            ("max_calls", self.max_calls),
            ("intent_rubric", self.intent_rubric),
            ("user_intent", self.user_intent),
            ("tool_schemas", self.tool_schemas),
        ):
            if v is not None:
                cfg[k] = list(v) if isinstance(v, (list, tuple)) else v
        if self._logit_gate is not None:
            if getattr(self._logit_gate, "pass_threshold", None) is not None:
                cfg["pass_threshold"] = self._logit_gate.pass_threshold
            if getattr(self._logit_gate, "fail_threshold", None) is not None:
                cfg["fail_threshold"] = self._logit_gate.fail_threshold
            cfg["symmetric_debiasing"] = getattr(self._logit_gate, "symmetric_debiasing", True)
            if hasattr(self._logit_gate, "on_unsupported_logprobs"):
                cfg["on_unsupported_logprobs"] = self._logit_gate.on_unsupported_logprobs
            if getattr(self._logit_gate, "sidecar_backend", None) is not None:
                cfg["sidecar_backend"] = self._logit_gate.sidecar_backend
        return cfg

    @property
    def backend(self) -> Any:
        return self._logit_gate.backend if self._logit_gate is not None else None

    def __repr__(self) -> str:
        feats: list[str] = []
        if self.allowed_tools is not None:
            feats.append(f"allowed_tools={self.allowed_tools!r}")
        if self.blocked_tools is not None:
            feats.append(f"blocked_tools={self.blocked_tools!r}")
        if self.max_calls is not None:
            feats.append(f"max_calls={self.max_calls}")
        if self.check_intent:
            feats.append("check_intent=True")
        fstr = f", {', '.join(feats)}" if feats else ""
        return f"ToolCallGate(name='{self.name}'{fstr})"

    def _record_verdict(self, verdict: Verdict, candidate: Any, kw: dict[str, Any]) -> None:
        from logitgate.cache import store_cached_verdict
        from logitgate.snapshot import persist_failure_snapshot
        from logitgate.telemetry.sinks import emit_verdict_telemetry

        store_cached_verdict(self, candidate, verdict, gate_cache=self.cache, call_kwargs=kw)
        emit_verdict_telemetry(verdict, candidate=candidate, gate_name=self.name, gate_type="ToolCallGate", gate_sink=self.telemetry_sink, call_kwargs=kw)
        if not verdict.passed:
            fcode = None
            if verdict.traces:
                for tr in reversed(verdict.traces):
                    if tr.status == DecisionStatus.FAILED and tr.failure_code:
                        fcode = tr.failure_code
                        break
                if fcode is None and verdict.traces:
                    fcode = verdict.traces[0].failure_code
            cfg = self.replay_config()
            cfg["failure_code"] = fcode
            persist_failure_snapshot(
                verdict=verdict,
                candidate=candidate,
                rubric=self.intent_rubric,
                save_on_fail=bool(kw.get("save_on_fail", self.save_on_fail)),
                test_name=kw.get("test_name"),
                snapshot_dir=kw.get("snapshot_dir"),
                gate_type="ToolCallGate",
                gate_config=cfg,
            )

    def _fail(self, err: str, code: ToolCallFailureCode, t0: float, *, candidate: Any = None, call_kwargs: dict[str, Any] | None = None, calls: list[NormalizedToolCall] | None = None) -> Verdict:
        lat = (time.perf_counter() - t0) * 1000.0
        lname = f"{self.name}:contract" if self.check_intent else self.name
        meta: dict[str, Any] = {"failure_code": code.value, "layer_tier": TIER_DETERMINISTIC}
        if calls:
            meta["tool_names"] = [c.name for c in calls]
        tr = LayerTrace(layer_name=lname, status=DecisionStatus.FAILED, latency_ms=lat, diagnostic_report=err, failure_code=code.value, metadata=meta)
        v = Verdict(passed=False, status=DecisionStatus.FAILED, triggered_layer=lname, total_latency_ms=lat, total_tokens_generated=0, reason=err, traces=[tr])
        self._record_verdict(v, candidate, call_kwargs or {})
        return v

    def _pass(self, t0: float, *, candidate: Any = None, call_kwargs: dict[str, Any] | None = None, calls: list[NormalizedToolCall] | None = None) -> Verdict:
        lat = (time.perf_counter() - t0) * 1000.0
        meta: dict[str, Any] = {"tool_names": [c.name for c in calls], "tool_calls_count": len(calls)} if calls else {}
        tr = LayerTrace(layer_name=self.name, status=DecisionStatus.PASSED, latency_ms=lat, diagnostic_report=None, failure_code=None, metadata=meta)
        v = Verdict(passed=True, status=DecisionStatus.PASSED, triggered_layer=self.name, total_latency_ms=lat, total_tokens_generated=0, reason=None, traces=[tr])
        self._record_verdict(v, candidate, call_kwargs or {})
        return v

    def _scan_arguments_security(self, tool_name: str, arguments: dict[str, Any]) -> tuple[ToolCallFailureCode, str] | None:
        stack: list[tuple[str, Any]] = list(arguments.items())
        while stack:
            k, val = stack.pop()
            if isinstance(val, dict):
                stack.extend(val.items())
            elif isinstance(val, (list, tuple)):
                for item in val:
                    if isinstance(item, dict):
                        stack.extend(item.items())
                    elif isinstance(item, str):
                        stack.append((k, item))
            elif isinstance(val, str):
                k_lower = k.lower()
                if self.block_destructive_commands and (k_lower in _CMD_KEYS or _looks_like_cmd(tool_name) or re.search(r"(?:;|&&|\|\||`|\$\()", val)):
                    ok, reason = _check_destructive_cmd(val, tool_name)
                    if not ok:
                        return ToolCallFailureCode.DESTRUCTIVE_COMMAND, reason
                if self.block_destructive_sql and (k_lower in _SQL_KEYS or _looks_like_sql_tool(tool_name) or _looks_like_sql_val(val)):
                    ok, reason = _check_destructive_sql(val, tool_name)
                    if not ok:
                        return ToolCallFailureCode.SQL_INJECTION_OR_DESTRUCTIVE, reason
                if self.block_ssrf and (k_lower in _URL_KEYS or val.lower().startswith(("http://", "https://", "file://", "gopher://", "dict://", "ftp://", "javascript:"))):
                    ok, reason = _check_ssrf_security(val, self.allowed_hosts)
                    if not ok:
                        return ToolCallFailureCode.SSRF_BLOCKED, reason
                if self.block_path_traversal and (k_lower in _PATH_KEYS or val.startswith(("/", "~/", "./", "../", ".\\", "..\\"))):
                    ok, reason = _check_path_security(val, self.allowed_roots)
                    if not ok:
                        return ToolCallFailureCode.PATH_TRAVERSAL, reason
        return None

    def _run_stage0_checks(self, calls: list[NormalizedToolCall]) -> tuple[ToolCallFailureCode, str] | None:
        if self.max_calls is not None and len(calls) > self.max_calls:
            return ToolCallFailureCode.SCHEMA_VIOLATION, f"Tool call count {len(calls)} exceeds max_calls={self.max_calls}"

        for call in calls:
            if self.blocked_tools is not None and call.name in self.blocked_tools:
                return ToolCallFailureCode.DISALLOWED_TOOL, f"Tool '{call.name}' is blocked by policy"
            if self.allowed_tools is not None and call.name not in self.allowed_tools:
                return ToolCallFailureCode.DISALLOWED_TOOL, f"Tool '{call.name}' is not in allowed_tools"
            if self.tool_schemas is not None and self.allowed_tools is None and call.name not in self._normalized_schemas:
                return ToolCallFailureCode.DISALLOWED_TOOL, f"Tool '{call.name}' has no defined schema in tool_schemas"
            if call.name in self._normalized_schemas:
                try:
                    _validate_json_schema(call.arguments, self._normalized_schemas[call.name], path=f"tool:{call.name}.arguments")
                except ValueError as err:
                    return ToolCallFailureCode.SCHEMA_VIOLATION, f"Schema violation in tool '{call.name}': {err}"
            if self.block_destructive_sql and call.name in ("truncate_tables", "drop_table", "drop_database"):
                return ToolCallFailureCode.SQL_INJECTION_OR_DESTRUCTIVE, f"Destructive SQL tool '{call.name}' is prohibited"
            if self.block_destructive_commands and call.name == "delete_directory":
                td = str(call.arguments.get("target_dir", "")).strip()
                if td in ("/", "/root", "/boot", "/etc", "~", "/home", "/usr", "/var"):
                    return ToolCallFailureCode.DESTRUCTIVE_COMMAND, f"Destructive deletion targeting system root '{td}'"

            violation = self._scan_arguments_security(call.name, call.arguments)
            if violation is not None:
                return violation

            if self.custom_validators:
                for cv in self.custom_validators:
                    try:
                        res = cv(call)
                    except Exception as exc:
                        return ToolCallFailureCode.CUSTOM_POLICY_FAILED, f"Custom validator raised exception: {exc}"
                    if res is False or (isinstance(res, str) and res.strip()):
                        return ToolCallFailureCode.CUSTOM_POLICY_FAILED, res if isinstance(res, str) else f"Custom validator rejected tool call '{call.name}'"

        return None

    def _format_intent_candidate(self, calls: list[NormalizedToolCall], user_intent: str | None) -> str:
        lines: list[str] = []
        if user_intent:
            lines.append(f"User Request:\n{user_intent.strip()}\n")
        lines.append("Proposed Tool Calls:")
        for c in calls:
            lines.append(f"- Tool: {c.name}")
            if c.name in self._tool_descriptions:
                lines.append(f"  Description: {self._tool_descriptions[c.name]}")
            lines.append(f"  Arguments: {json.dumps(c.arguments, sort_keys=True)}")
        return "\n".join(lines)

    def _check_cache(self, candidate: Any, kwargs: dict[str, Any]) -> Verdict | None:
        from logitgate.cache import lookup_cached_verdict

        hit = lookup_cached_verdict(self, candidate, gate_cache=self.cache, call_kwargs=kwargs)
        if hit is not None:
            tags = {**(kwargs.get("telemetry_tags") or {}), "cache_hit": True}
            self._record_verdict(hit, candidate, {**kwargs, "telemetry_tags": tags})
            return hit
        return None

    def _compose_intent_verdict(
        self, candidate: Any, calls: list[NormalizedToolCall], intent_verdict: Verdict, stage0_latency: float, kwargs: dict[str, Any]
    ) -> Verdict:
        stage0_trace = LayerTrace(
            layer_name=f"{self.name}:contract", status=DecisionStatus.PASSED, latency_ms=stage0_latency,
            diagnostic_report=None, failure_code=None, metadata={"tool_calls_count": len(calls), "tool_names": [c.name for c in calls]},
        )
        verdict = Verdict(
            passed=intent_verdict.passed, status=intent_verdict.status, probability=intent_verdict.probability,
            triggered_layer=intent_verdict.triggered_layer, total_latency_ms=stage0_latency + intent_verdict.total_latency_ms,
            total_tokens_generated=intent_verdict.total_tokens_generated, reason=intent_verdict.reason,
            traces=[stage0_trace, *intent_verdict.traces],
        )
        self._record_verdict(verdict, candidate, kwargs)
        return verdict

    def _prep_eval(self, candidate: Any, kwargs: dict[str, Any]) -> tuple[float, list[NormalizedToolCall], str] | Verdict:
        t0 = time.perf_counter()
        try:
            calls, extracted_intent = normalize_tool_calls(candidate)
        except ToolCallNormalizationError as err:
            return self._fail(err.message, err.code, t0, candidate=candidate, call_kwargs=kwargs)
        except Exception as exc:
            return self._fail(f"Failed to normalize tool calls: {exc}", ToolCallFailureCode.MALFORMED_ENVELOPE, t0, candidate=candidate, call_kwargs=kwargs)

        failure = self._run_stage0_checks(calls)
        if failure is not None:
            code, msg = failure
            return self._fail(msg, code, t0, candidate=candidate, call_kwargs=kwargs, calls=calls)

        if not self.check_intent or self._logit_gate is None:
            return self._pass(t0, candidate=candidate, call_kwargs=kwargs, calls=calls)

        effective_intent = kwargs.get("user_intent") or self.user_intent or extracted_intent
        prompt = self._format_intent_candidate(calls, effective_intent)
        return t0, calls, prompt

    def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        hit = self._check_cache(candidate, kwargs)
        if hit is not None:
            return hit
        prep = self._prep_eval(candidate, kwargs)
        if isinstance(prep, Verdict):
            return prep
        t0, calls, prompt = prep
        stage0_lat = (time.perf_counter() - t0) * 1000.0
        intent_verdict = self._logit_gate.evaluate(prompt, **{**kwargs, "save_on_fail": False})  # type: ignore[union-attr]
        return self._compose_intent_verdict(candidate, calls, intent_verdict, stage0_lat, kwargs)

    async def aevaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        if not self.check_intent or self._logit_gate is None:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: self.evaluate(candidate, **kwargs))
        hit = self._check_cache(candidate, kwargs)
        if hit is not None:
            return hit
        prep = self._prep_eval(candidate, kwargs)
        if isinstance(prep, Verdict):
            return prep
        t0, calls, prompt = prep
        stage0_lat = (time.perf_counter() - t0) * 1000.0
        intent_verdict = await self._logit_gate.aevaluate(prompt, **{**kwargs, "save_on_fail": False})
        return self._compose_intent_verdict(candidate, calls, intent_verdict, stage0_lat, kwargs)

    def close(self) -> None:
        if not getattr(self, "_is_closed", False):
            if self._logit_gate is not None and getattr(self, "_owns_logit_gate", False):
                _safe_close(self._logit_gate)
            super().close()

    async def aclose(self) -> None:
        if not getattr(self, "_is_closed", False):
            if self._logit_gate is not None and getattr(self, "_owns_logit_gate", False):
                await _safe_aclose(self._logit_gate)
            await super().aclose()
