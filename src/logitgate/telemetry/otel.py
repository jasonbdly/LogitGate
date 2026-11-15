from __future__ import annotations

import math
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from logitgate.exceptions import ConfigurationError
from logitgate.models import LayerTrace
from logitgate.telemetry.models import (
    ErrorPolicy,
    PrivacyMode,
    TelemetryRecord,
    redact_text,
)
from logitgate.telemetry.sinks import (
    TelemetrySink,
    _handle_sink_error,
    _validate_privacy_and_bounds,
)

OtelAttributeValue = (
    bool
    | int
    | float
    | str
    | list[bool]
    | list[int]
    | list[float]
    | list[str]
)


def _sanitize_otel_value(val: Any) -> OtelAttributeValue | None:
    """Coerces ``val`` into a strictly OpenTelemetry-compliant primitive or homogeneous list.

    Returns ``None`` for ``None``, non-finite floats (``NaN``/``+-inf``), or unsupported
    types so callers can cleanly omit invalid attributes.
    """
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return int(val)
    if isinstance(val, float):
        return float(val) if math.isfinite(val) else None
    if isinstance(val, str):
        return val
    if isinstance(val, Sequence) and not isinstance(val, (bytes, bytearray)):
        items = list(val)
        if not items:
            return []
        if all(isinstance(x, bool) for x in items):
            return [bool(x) for x in items]
        if all(isinstance(x, int) and not isinstance(x, bool) for x in items):
            return [int(x) for x in items]
        if all(
            isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))
            for x in items
        ):
            return [float(x) for x in items]
        if all(isinstance(x, str) for x in items):
            return [str(x) for x in items]
    return None


def _put_attr(attrs: dict[str, OtelAttributeValue], key: str, val: Any) -> None:
    sanitized = _sanitize_otel_value(val)
    if sanitized is not None:
        attrs[key] = sanitized


def _infer_gen_ai_system(backend_spec: str | None) -> str | None:
    if not backend_spec:
        return None
    if "://" in backend_spec:
        prefix = backend_spec.split("://", 1)[0].strip().lower()
        return prefix or None
    return backend_spec.strip().lower() or None


class OtelSpanSink(TelemetrySink):
    """Zero-dependency OpenTelemetry span sink for `LogitGate` evaluations.

    Bridges structured `TelemetryRecord` verdicts into OpenTelemetry spans and per-layer
    span events:
    - Accepts an explicit ``tracer`` (any OpenTelemetry ``Tracer`` or duck-typed recording
      tracer exposing ``start_as_current_span`` or ``start_span``), OR
    - Lazily resolves ``opentelemetry.trace.get_tracer(...)`` when ``opentelemetry-api``
      (``pip install 'logitgate[otel]'``) is installed, OR
    - Degrades to a zero-overhead no-op when ``opentelemetry-api`` is absent and
      ``require_otel=False`` (default).
    """

    def __init__(
        self,
        tracer: Any = None,
        *,
        instrumentation_name: str = "logitgate",
        instrumentation_version: str | None = None,
        span_prefix: str = "logitgate.eval",
        privacy: PrivacyMode = "omit_candidate",
        max_candidate_chars: int | None = 4096,
        record_layer_events: bool = True,
        error_on_fail: bool = False,
        require_otel: bool = False,
        filter_fn: Callable[[TelemetryRecord], bool] | None = None,
        on_error: ErrorPolicy = "warn",
    ) -> None:
        val_priv, val_chars, val_err = _validate_privacy_and_bounds(
            privacy, max_candidate_chars, on_error
        )
        if not isinstance(instrumentation_name, str) or not instrumentation_name.strip():
            raise ConfigurationError(
                f"instrumentation_name must be a non-empty string, got {instrumentation_name!r}"
            )
        if instrumentation_version is not None and not isinstance(instrumentation_version, str):
            raise ConfigurationError(
                "instrumentation_version must be a string or None, "
                f"got {instrumentation_version!r}"
            )
        if not isinstance(span_prefix, str) or not span_prefix.strip():
            raise ConfigurationError(
                f"span_prefix must be a non-empty string, got {span_prefix!r}"
            )
        for flag_name, flag_val in (
            ("record_layer_events", record_layer_events),
            ("error_on_fail", error_on_fail),
            ("require_otel", require_otel),
        ):
            if not isinstance(flag_val, bool):
                raise ConfigurationError(f"{flag_name} must be a bool, got {flag_val!r}")
        if filter_fn is not None and not callable(filter_fn):
            raise ConfigurationError("filter_fn must be callable or None")

        if instrumentation_version is None:
            from logitgate import __version__

            resolved_version = __version__
        else:
            resolved_version = instrumentation_version

        resolved_tracer: Any = None
        if tracer is not None:
            if not (
                callable(getattr(tracer, "start_as_current_span", None))
                or callable(getattr(tracer, "start_span", None))
            ):
                raise ConfigurationError(
                    "OtelSpanSink tracer must implement 'start_as_current_span' or 'start_span'."
                )
            resolved_tracer = tracer
        else:
            try:
                from opentelemetry import trace as otel_trace
            except ImportError as exc:
                if require_otel:
                    raise ConfigurationError(
                        "OtelSpanSink(require_otel=True) requires 'opentelemetry-api'. "
                        "Install with `pip install 'logitgate[otel]'` or pass `tracer=`."
                    ) from exc
                resolved_tracer = None
            else:
                resolved_tracer = otel_trace.get_tracer(
                    instrumentation_name.strip(), resolved_version
                )

        self.instrumentation_name: str = instrumentation_name.strip()
        self.instrumentation_version: str = resolved_version
        self.span_prefix: str = span_prefix.strip().rstrip(".")
        self.privacy: PrivacyMode = val_priv
        self.max_candidate_chars: int | None = val_chars
        self.record_layer_events: bool = record_layer_events
        self.error_on_fail: bool = error_on_fail
        self.require_otel: bool = require_otel
        self.filter_fn = filter_fn
        self.on_error: ErrorPolicy = val_err

        self._tracer: Any = resolved_tracer
        self._lock = threading.Lock()
        self._closed = False
        self._spans_emitted = 0

    @property
    def is_enabled(self) -> bool:
        """Returns ``True`` when an active tracer is bound and the sink is open."""
        return self._tracer is not None and not self._closed

    @property
    def spans_emitted(self) -> int:
        """Total number of spans emitted by this sink instance."""
        with self._lock:
            return self._spans_emitted

    def _build_attributes_from_prepared(
        self, prepared: TelemetryRecord
    ) -> dict[str, OtelAttributeValue]:
        rubric_val = prepared.rubric
        if rubric_val is not None and self.privacy == "redact_pii":
            rubric_val = redact_text(rubric_val, max_chars=self.max_candidate_chars)

        attrs: dict[str, OtelAttributeValue] = {}
        _put_attr(attrs, "logitgate.schema_version", prepared.schema_version)
        _put_attr(attrs, "logitgate.event_id", prepared.event_id)
        _put_attr(attrs, "logitgate.timestamp", prepared.timestamp)
        _put_attr(attrs, "logitgate.gate.name", prepared.gate_name)
        _put_attr(attrs, "logitgate.gate.type", prepared.gate_type)
        _put_attr(attrs, "logitgate.verdict.passed", prepared.passed)
        _put_attr(attrs, "logitgate.verdict.status", prepared.status.value)
        _put_attr(attrs, "logitgate.verdict.fast_pathed", prepared.fast_pathed)
        _put_attr(attrs, "logitgate.verdict.escalated", prepared.escalated)
        _put_attr(attrs, "logitgate.verdict.latency_ms", prepared.total_latency_ms)
        _put_attr(attrs, "logitgate.verdict.tokens_generated", prepared.total_tokens_generated)
        _put_attr(attrs, "logitgate.candidate.sha256", prepared.candidate_sha256)
        _put_attr(attrs, "logitgate.candidate.length", prepared.candidate_length)

        _put_attr(attrs, "gen_ai.system", _infer_gen_ai_system(prepared.backend))
        _put_attr(attrs, "gen_ai.request.model", prepared.model)
        _put_attr(attrs, "logitgate.model", prepared.model)
        _put_attr(attrs, "logitgate.backend", prepared.backend)
        _put_attr(attrs, "logitgate.teacher.model", prepared.teacher_model)
        _put_attr(attrs, "logitgate.teacher.passed", prepared.teacher_passed)
        _put_attr(attrs, "logitgate.verdict.probability", prepared.probability)
        _put_attr(attrs, "logitgate.verdict.score", prepared.score)
        _put_attr(attrs, "logitgate.verdict.choice", prepared.choice)
        _put_attr(attrs, "logitgate.verdict.triggered_layer", prepared.triggered_layer)
        _put_attr(attrs, "logitgate.verdict.reason", prepared.reason)
        _put_attr(attrs, "logitgate.rubric", rubric_val)
        _put_attr(attrs, "logitgate.candidate.text", prepared.candidate)

        if prepared.traces:
            _put_attr(attrs, "logitgate.layers.count", len(prepared.traces))
            _put_attr(attrs, "logitgate.layers.names", [tr.layer_name for tr in prepared.traces])
            _put_attr(
                attrs,
                "logitgate.layers.latencies_ms",
                [round(float(tr.latency_ms), 4) for tr in prepared.traces],
            )

        if isinstance(prepared.tags, Mapping):
            for raw_key, raw_val in prepared.tags.items():
                clean_key = str(raw_key).strip()
                if not clean_key:
                    continue
                tag_val = (
                    redact_text(raw_val, max_chars=self.max_candidate_chars)
                    if isinstance(raw_val, str) and self.privacy == "redact_pii"
                    else raw_val
                )
                _put_attr(attrs, f"logitgate.tag.{clean_key}", tag_val)

        return attrs

    def build_span_attributes(self, record: TelemetryRecord) -> dict[str, OtelAttributeValue]:
        """Builds a sanitized OpenTelemetry attribute dictionary for ``record``."""
        prepared = record.with_privacy(
            privacy=self.privacy,
            max_candidate_chars=self.max_candidate_chars,
        )
        return self._build_attributes_from_prepared(prepared)

    def _build_layer_event_attributes(
        self, index: int, trace: LayerTrace
    ) -> dict[str, OtelAttributeValue]:
        ev_attrs: dict[str, OtelAttributeValue] = {}
        _put_attr(ev_attrs, "logitgate.layer.index", index)
        _put_attr(ev_attrs, "logitgate.layer.name", trace.layer_name)
        _put_attr(ev_attrs, "logitgate.layer.status", trace.status.value)
        _put_attr(ev_attrs, "logitgate.layer.latency_ms", round(float(trace.latency_ms), 4))
        _put_attr(ev_attrs, "logitgate.layer.tokens_generated", trace.tokens_generated)
        _put_attr(ev_attrs, "logitgate.layer.calibrated_prob", trace.calibrated_prob)
        _put_attr(ev_attrs, "logitgate.layer.log_odds_pass", trace.log_odds_pass)
        _put_attr(
            ev_attrs,
            "logitgate.layer.positional_bias_magnitude",
            trace.positional_bias_magnitude,
        )
        _put_attr(ev_attrs, "logitgate.layer.entropy", trace.shannon_entropy)
        _put_attr(ev_attrs, "logitgate.layer.failure_code", trace.failure_code)
        _put_attr(ev_attrs, "logitgate.layer.diagnostic_report", trace.diagnostic_report)
        return ev_attrs

    def _populate_span(self, span: Any, prepared: TelemetryRecord) -> None:
        if self.record_layer_events and callable(getattr(span, "add_event", None)):
            for idx, trace in enumerate(prepared.traces):
                span.add_event(
                    "logitgate.layer",
                    attributes=self._build_layer_event_attributes(idx, trace),
                )

        if self.error_on_fail and not prepared.passed and callable(
            getattr(span, "set_status", None)
        ):
            desc = prepared.reason or f"Gate '{prepared.gate_name}' failed ({prepared.status.value})"
            try:
                from opentelemetry.trace import Status, StatusCode

                span.set_status(Status(StatusCode.ERROR, desc))
            except ImportError:
                span.set_status("ERROR", desc)

    def emit(self, record: TelemetryRecord) -> None:
        try:
            if self._closed or self._tracer is None:
                return
            if self.filter_fn is not None and not self.filter_fn(record):
                return

            prepared = record.with_privacy(
                privacy=self.privacy,
                max_candidate_chars=self.max_candidate_chars,
            )
            attrs = self._build_attributes_from_prepared(prepared)
            gate_suffix = (prepared.gate_name or prepared.gate_type or "gate").strip()
            span_name = f"{self.span_prefix}.{gate_suffix}"

            start_ctx = getattr(self._tracer, "start_as_current_span", None)
            if callable(start_ctx):
                with start_ctx(span_name, attributes=attrs) as span:
                    if span is not None:
                        self._populate_span(span, prepared)
            else:
                span = self._tracer.start_span(span_name, attributes=attrs)
                try:
                    if span is not None:
                        self._populate_span(span, prepared)
                finally:
                    end_fn = getattr(span, "end", None)
                    if callable(end_fn):
                        end_fn()

            with self._lock:
                self._spans_emitted += 1
        except Exception as exc:  # noqa: BLE001 - tracer/callback failures routed via on_error
            _handle_sink_error(exc, self.on_error, "OtelSpanSink")

    def close(self) -> None:
        with self._lock:
            self._closed = True
