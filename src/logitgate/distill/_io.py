from __future__ import annotations

import contextlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

from logitgate.exceptions import ConfigurationError

_DEFAULT_MAX_ARTIFACT_BYTES = 16 * 1024 * 1024  # 16 MiB


def atomic_write_json_nofollow(
    path: str | Path,
    payload_obj: dict[str, Any],
    *,
    artifact_name: str,
) -> Path:
    """Atomically writes ``payload_obj`` as JSON, rejecting symlinks on target and parent."""
    if isinstance(path, str) and not path.strip():
        raise ConfigurationError(f"{artifact_name} save path cannot be empty.")
    target = Path(path)
    try:
        for anc in (target.parent, *target.parent.parents):
            if anc.exists() and anc.is_symlink():
                raise ConfigurationError(
                    f"Refusing to write {artifact_name} under symlinked directory: {anc}"
                )
        if target.is_symlink():
            raise ConfigurationError(
                f"Refusing to write {artifact_name} through symlink: {target}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(payload_obj, indent=2, allow_nan=False) + "\n"
        fd, tmp_name = tempfile.mkstemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
        tmp_path = Path(tmp_name)
        fd_owned = True
        try:
            with contextlib.suppress(AttributeError, OSError):
                os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                fd_owned = False
                handle.write(payload)
                handle.flush()
            if target.is_symlink():
                raise ConfigurationError(
                    f"Refusing to overwrite symlinked {artifact_name}: {target}"
                )
            os.replace(tmp_path, target)
        except BaseException:
            if fd_owned:
                with contextlib.suppress(OSError):
                    os.close(fd)
            with contextlib.suppress(OSError):
                tmp_path.unlink(missing_ok=True)
            raise
    except ConfigurationError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"Failed to save {artifact_name} to '{target}': {exc}"
        ) from exc
    return target


def read_json_object_nofollow(
    path: str | Path,
    *,
    artifact_name: str,
    max_bytes: int = _DEFAULT_MAX_ARTIFACT_BYTES,
) -> dict[str, Any]:
    """Loads a JSON object from a regular file using ``O_NOFOLLOW`` and size bounds."""
    if isinstance(path, str) and not path.strip():
        raise ConfigurationError(f"{artifact_name} load path cannot be empty.")
    target = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        if target.is_symlink():
            raise ConfigurationError(
                f"Refusing to load {artifact_name} from symlink: {target}"
            )
        fd = os.open(target, flags)
    except ConfigurationError:
        raise
    except OSError as exc:
        raise ConfigurationError(
            f"Failed to open {artifact_name} at '{target}': {exc}"
        ) from exc

    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise ConfigurationError(
                f"{artifact_name} path is not a regular file: {target}"
            )
        if st.st_size > max_bytes:
            raise ConfigurationError(
                f"{artifact_name} file '{target}' exceeds maximum size ({st.st_size} > {max_bytes} bytes)"
            )
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            fd = -1
            raw = handle.read(max_bytes + 1)
    except ConfigurationError:
        raise
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigurationError(
            f"Failed to read {artifact_name} from '{target}': {exc}"
        ) from exc
    finally:
        if fd >= 0:
            with contextlib.suppress(OSError):
                os.close(fd)

    if len(raw.encode("utf-8")) > max_bytes:
        raise ConfigurationError(
            f"{artifact_name} file '{target}' exceeds maximum size ({max_bytes} bytes)"
        )
    try:
        parsed = json.loads(raw)
    except (ValueError, RecursionError) as exc:
        raise ConfigurationError(
            f"Invalid JSON in {artifact_name} '{target}': {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ConfigurationError(
            f"{artifact_name} '{target}' must contain a top-level JSON object, got {type(parsed).__name__}"
        )
    return parsed
