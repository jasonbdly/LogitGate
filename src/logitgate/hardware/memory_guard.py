from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VRAMWatermarkStatus:
    """Point-in-time CUDA memory reading, in MiB. All zeros means "no CUDA device visible"."""

    allocated_mb: float
    reserved_mb: float
    free_physical_mb: float
    total_physical_mb: float
    spillover_risk_detected: bool


class PCIeSpilloverGuard:
    """Monitors physical CUDA VRAM headroom before Layer 1 / Layer 2 prefill passes

    to prevent silent Windows/WSL2/Linux CUDA Unified Memory paging over PCIe.
    """

    def __init__(
        self,
        device_index: int = 0,
        min_free_headroom_mb: float = 512.0,
        auto_empty_cache: bool = True,
    ) -> None:
        self.device_index = device_index
        self.min_free_headroom_mb = min_free_headroom_mb
        self.auto_empty_cache = auto_empty_cache

    def inspect(self) -> VRAMWatermarkStatus:
        try:
            import torch

            if not torch.cuda.is_available():
                return VRAMWatermarkStatus(0.0, 0.0, 0.0, 0.0, False)

            free_bytes, total_bytes = torch.cuda.mem_get_info(self.device_index)
            alloc_bytes = torch.cuda.memory_allocated(self.device_index)
            reserved_bytes = torch.cuda.memory_reserved(self.device_index)

            free_mb = free_bytes / (1024 * 1024)
            total_mb = total_bytes / (1024 * 1024)
            alloc_mb = alloc_bytes / (1024 * 1024)
            reserved_mb = reserved_bytes / (1024 * 1024)

            spillover_risk = (
                free_mb < self.min_free_headroom_mb or reserved_mb > (total_mb * 0.96)
            )
            return VRAMWatermarkStatus(
                allocated_mb=round(alloc_mb, 2),
                reserved_mb=round(reserved_mb, 2),
                free_physical_mb=round(free_mb, 2),
                total_physical_mb=round(total_mb, 2),
                spillover_risk_detected=spillover_risk,
            )
        except (ImportError, RuntimeError, AttributeError):
            return VRAMWatermarkStatus(0.0, 0.0, 0.0, 0.0, False)

    @contextmanager
    def guard_forward_pass(
        self, estimated_kv_mb: float = 128.0
    ) -> Iterator[VRAMWatermarkStatus]:
        status = self.inspect()
        has_risk = status.spillover_risk_detected or (
            status.total_physical_mb > 0
            and status.free_physical_mb < (self.min_free_headroom_mb + estimated_kv_mb)
        )
        if has_risk and self.auto_empty_cache:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    status = self.inspect()
            except (ImportError, RuntimeError, AttributeError):
                pass
        yield status
