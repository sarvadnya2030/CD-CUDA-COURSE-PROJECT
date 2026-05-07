"""
occupancy.py — CUDA occupancy and register pressure analysis for sm_75.

Parses ptxas -v output captured during nvcc compilation to extract
per-kernel resource usage, then applies sm_75 hardware limits to
compute theoretical occupancy.

RTX 2070 / sm_75 hardware limits
  max_registers_per_block  = 65 536
  max_shared_mem_per_block = 49 152 bytes
  max_warps_per_sm         = 32
  max_blocks_per_sm        = 16
  warp_size                = 32

Register-to-warp limit:
  max_warps_from_regs = floor(max_registers_per_block / (regs * warp_size))
                        (capped at max_warps_per_sm)

Shared-memory-to-warp limit:
  max_blocks_from_smem = floor(max_shared_mem_per_block / smem_per_block)
  max_warps_from_smem  = min(max_blocks_from_smem, max_blocks_per_sm)
                         * (block_size / warp_size)

Occupancy = active_warps / max_warps_per_sm
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ── sm_75 (RTX 2070) hardware limits ──────────────────────────────────────

SM75_MAX_REGISTERS_PER_BLOCK  = 65_536
SM75_MAX_SHARED_MEM_PER_BLOCK = 49_152   # bytes
SM75_MAX_WARPS_PER_SM         = 32
SM75_MAX_BLOCKS_PER_SM        = 16
SM75_WARP_SIZE                = 32


@dataclass
class OccupancyInfo:
    """Resource usage and derived occupancy for one compiled kernel."""
    kernel_name: str
    registers_per_thread: int       # from ptxas info
    shared_mem_bytes: int           # bytes of static shared memory
    spill_stores: int               # register-spill stores to local memory
    spill_loads: int                # register-spill loads from local memory
    has_register_spill: bool        # True if spill_stores > 0
    occupancy: float                # theoretical occupancy [0, 1]
    active_warps: int               # theoretical active warps per SM
    block_size: int                 # threads per block (from params)


# ── ptxas output parsing ───────────────────────────────────────────────────

_RE_PTXAS_INFO = re.compile(
    r"ptxas\s+info\s*:\s*"
    r"Function properties for (?P<name>\w+).*?$",
    re.MULTILINE,
)

_RE_REGS = re.compile(r"(\d+)\s+registers")
_RE_SMEM = re.compile(r"(\d+)\s+bytes\s+smem")
_RE_SPILL_S = re.compile(r"(\d+)\s+bytes\s+stack\s+frame.*?(\d+)\s+bytes\s+spill\s+stores")
_RE_SPILL_L = re.compile(r"(\d+)\s+bytes\s+spill\s+loads")

# Simpler per-line patterns
_RE_REGS_LINE  = re.compile(r"(\d+)\s+registers?", re.IGNORECASE)
_RE_SMEM_LINE  = re.compile(r"(\d+)\s+bytes\s+(?:smem|shared\s+mem)", re.IGNORECASE)
_RE_SPILLS_LINE = re.compile(r"(\d+)\s+bytes?\s+spill\s+stores?", re.IGNORECASE)
_RE_SPILLL_LINE = re.compile(r"(\d+)\s+bytes?\s+spill\s+loads?",  re.IGNORECASE)


def parse_ptxas_stderr(stderr: str) -> dict[str, dict]:
    """
    Parse nvcc -Xptxas -v stderr output.

    Returns a dict keyed by kernel (function) name, each value being a
    sub-dict with keys: registers, smem_bytes, spill_stores, spill_loads.

    ptxas emits lines like::

        ptxas info    : Function properties for matmul_opt
        ptxas info    : Used 32 registers, 8192 bytes smem, ...
        ptxas info    : 0 bytes spill stores, 0 bytes spill loads
    """
    results: dict[str, dict] = {}
    current: Optional[str] = None
    defaults: dict = {"registers": 0, "smem_bytes": 0,
                      "spill_stores": 0, "spill_loads": 0}

    for line in stderr.splitlines():
        # Detect function header
        if "Function properties for" in line:
            m = re.search(r"Function properties for\s+(\w+)", line)
            if m:
                current = m.group(1)
                results[current] = dict(defaults)
            continue

        if current is None:
            continue

        # Registers
        m = _RE_REGS_LINE.search(line)
        if m:
            results[current]["registers"] = int(m.group(1))

        # Shared memory
        m = _RE_SMEM_LINE.search(line)
        if m:
            results[current]["smem_bytes"] = int(m.group(1))

        # Spill stores
        m = _RE_SPILLS_LINE.search(line)
        if m:
            results[current]["spill_stores"] = int(m.group(1))

        # Spill loads
        m = _RE_SPILLL_LINE.search(line)
        if m:
            results[current]["spill_loads"] = int(m.group(1))

    return results


# ── Occupancy calculator ───────────────────────────────────────────────────

def compute_occupancy(
    registers_per_thread: int,
    smem_bytes: int,
    block_size: int,
    max_regs_per_block: int  = SM75_MAX_REGISTERS_PER_BLOCK,
    max_smem_per_block: int  = SM75_MAX_SHARED_MEM_PER_BLOCK,
    max_warps_per_sm: int    = SM75_MAX_WARPS_PER_SM,
    max_blocks_per_sm: int   = SM75_MAX_BLOCKS_PER_SM,
    warp_size: int           = SM75_WARP_SIZE,
) -> tuple[float, int]:
    """
    Compute theoretical occupancy for sm_75 given resource usage.

    Args:
        registers_per_thread: Registers used by each thread.
        smem_bytes:           Static shared memory per block (bytes).
        block_size:           Threads per block.

    Returns:
        (occupancy_fraction, active_warps_per_sm)
    """
    warps_per_block = math.ceil(block_size / warp_size)

    # Limit from register file
    if registers_per_thread > 0:
        max_blocks_from_regs = max_regs_per_block // (registers_per_thread * block_size)
    else:
        max_blocks_from_regs = max_blocks_per_sm

    # Limit from shared memory
    if smem_bytes > 0:
        max_blocks_from_smem = max_smem_per_block // smem_bytes
    else:
        max_blocks_from_smem = max_blocks_per_sm

    # Active blocks = min of all limits, then active warps
    active_blocks = min(
        max_blocks_from_regs,
        max_blocks_from_smem,
        max_blocks_per_sm,
    )
    active_warps = min(active_blocks * warps_per_block, max_warps_per_sm)
    occupancy = active_warps / max_warps_per_sm if max_warps_per_sm > 0 else 0.0

    return occupancy, active_warps


import math   # needed for math.ceil above


# ── Main analyzer class ────────────────────────────────────────────────────

class OccupancyAnalyzer:
    """
    Compile a CUDA source with -Xptxas -v and extract occupancy info.

    Integrates with BenchmarkResult to populate occupancy-related fields.

    Usage::

        oa = OccupancyAnalyzer()
        info = oa.analyze(src_path, kernel_name="matmul_opt", block_size=256)
        result.occupancy              = info.occupancy
        result.registers_per_thread  = info.registers_per_thread
        result.shared_mem_bytes      = info.shared_mem_bytes
        result.has_register_spill    = info.has_register_spill
    """

    def __init__(self, arch: str = "sm_75") -> None:
        """
        Args:
            arch: Target GPU architecture (e.g. "sm_75" for RTX 2070).
        """
        self._arch = arch

    def compile_and_parse(
        self,
        src_path: Path,
        out_path: Optional[Path] = None,
        extra_flags: Optional[list] = None,
    ) -> tuple[bool, dict[str, dict]]:
        """
        Compile *src_path* and parse ptxas occupancy info from stderr.

        If *out_path* is None, uses a temporary file name.

        Returns:
            (compile_success, {kernel_name: resource_dict})
        """
        if out_path is None:
            out_path = src_path.with_suffix(".tmp_occ_bin")

        flags = [
            "nvcc", "-O3", f"-arch={self._arch}",
            "--use_fast_math",
            "--ptxas-options=-v",
            str(src_path), "-o", str(out_path),
        ]
        if extra_flags:
            flags.extend(extra_flags)

        result = subprocess.run(flags, capture_output=True, text=True)
        parsed = parse_ptxas_stderr(result.stderr)

        # Clean up temp binary
        if out_path.name.endswith(".tmp_occ_bin"):
            try:
                out_path.unlink()
            except FileNotFoundError:
                pass

        return result.returncode == 0, parsed

    def analyze_from_stderr(
        self,
        stderr: str,
        kernel_function: str,
        block_size: int,
    ) -> Optional[OccupancyInfo]:
        """
        Derive OccupancyInfo from already-captured ptxas stderr.

        Used when the binary has already been compiled (avoiding recompile).

        Args:
            stderr:           The captured nvcc stderr from a previous compile.
            kernel_function:  Name of the __global__ function to analyse.
            block_size:       Threads per block for the launch config.
        """
        parsed = parse_ptxas_stderr(stderr)
        # Try exact match, then prefix match
        entry = parsed.get(kernel_function)
        if entry is None:
            for name, data in parsed.items():
                if kernel_function in name or name in kernel_function:
                    entry = data
                    break
        if entry is None and parsed:
            entry = next(iter(parsed.values()))
        if entry is None:
            return None

        return self._make_info(kernel_function, entry, block_size)

    def analyze(
        self,
        src_path: Path,
        kernel_function: str,
        block_size: int,
    ) -> Optional[OccupancyInfo]:
        """
        Compile *src_path* and return occupancy for *kernel_function*.

        Args:
            src_path:         CUDA source file.
            kernel_function:  Name of the __global__ function.
            block_size:       Threads per block.
        """
        ok, parsed = self.compile_and_parse(src_path)
        entry = parsed.get(kernel_function)
        if entry is None and parsed:
            entry = next(iter(parsed.values()))
        if entry is None:
            return None
        return self._make_info(kernel_function, entry, block_size)

    @staticmethod
    def _make_info(name: str, entry: dict, block_size: int) -> OccupancyInfo:
        """Build an OccupancyInfo from a parsed ptxas entry."""
        regs       = entry.get("registers", 0)
        smem       = entry.get("smem_bytes", 0)
        spill_s    = entry.get("spill_stores", 0)
        spill_l    = entry.get("spill_loads", 0)
        occ, warps = compute_occupancy(regs, smem, block_size)

        return OccupancyInfo(
            kernel_name=name,
            registers_per_thread=regs,
            shared_mem_bytes=smem,
            spill_stores=spill_s,
            spill_loads=spill_l,
            has_register_spill=spill_s > 0,
            occupancy=occ,
            active_warps=warps,
            block_size=block_size,
        )

    # ── Reporting ──────────────────────────────────────────────────────────

    @staticmethod
    def print_table(
        rows: list,   # list of BenchmarkResult or OccupancyInfo
        title: str = "Occupancy Analysis",
    ) -> None:
        """
        Print a formatted occupancy table.

        Accepts either BenchmarkResult objects (reads .occupancy etc.)
        or OccupancyInfo objects.
        """
        print(f"\n{'='*72}")
        print(f"  {title}")
        print(f"{'='*72}")
        hdr = (f"{'Config':<28} {'Occ%':>6} {'Regs':>5} "
               f"{'SMEM':>7} {'Spill':>6} {'Speedup':>8}")
        print(hdr)
        print("-" * 72)
        for row in rows:
            if hasattr(row, "variant"):  # BenchmarkResult
                config  = row.variant[:27]
                occ     = row.occupancy or 0.0
                regs    = row.registers_per_thread or 0
                smem    = row.shared_mem_bytes or 0
                spill   = "YES" if row.has_register_spill else "no"
                speedup = f"{row.speedup:.2f}x" if row.speedup else "—"
            else:                        # OccupancyInfo
                config  = row.kernel_name[:27]
                occ     = row.occupancy
                regs    = row.registers_per_thread
                smem    = row.shared_mem_bytes
                spill   = "YES" if row.has_register_spill else "no"
                speedup = "—"
            print(f"{config:<28} {occ*100:>5.1f}% {regs:>5} "
                  f"{smem:>6}B {spill:>6} {speedup:>8}")
        print("=" * 72 + "\n")
