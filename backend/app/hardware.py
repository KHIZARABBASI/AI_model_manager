import ctypes
import os
import platform
import shutil
import subprocess
from typing import Optional

from pydantic import BaseModel, Field


class GpuInfo(BaseModel):
    name: str
    vram_bytes: Optional[int] = Field(default=None)
    vram_source: Optional[str] = Field(default=None)


class HardwareInfo(BaseModel):
    os_name: str
    os_version: str
    architecture: str
    cpu_name: str
    logical_cpus: int
    system_ram_bytes: Optional[int]
    free_disk_bytes: Optional[int]
    disk_path: str
    gpus: list[GpuInfo]
    detection_warnings: list[str]


def _windows_total_memory() -> Optional[int]:
    if platform.system() != "Windows":
        return None

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("memory_load", ctypes.c_ulong),
            ("total_phys", ctypes.c_ulonglong),
            ("avail_phys", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong),
            ("avail_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("avail_virtual", ctypes.c_ulonglong),
            ("avail_extended_virtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(MemoryStatus)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return int(status.total_phys)
    return None


def _linux_total_memory() -> Optional[int]:
    try:
        with open("/proc/meminfo", encoding="utf-8") as memory_file:
            for line in memory_file:
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _detect_system_ram() -> Optional[int]:
    if platform.system() == "Windows":
        return _windows_total_memory()
    if platform.system() == "Linux":
        return _linux_total_memory()
    return None


def _run_command(command: list[str]) -> Optional[str]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _detect_nvidia_gpus() -> list[GpuInfo]:
    output = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return []

    gpus: list[GpuInfo] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",", 1)]
        if not parts:
            continue
        name = parts[0]
        vram_bytes: Optional[int] = None
        if len(parts) == 2:
            try:
                vram_bytes = int(parts[1]) * 1024 * 1024
            except ValueError:
                pass
        gpus.append(
            GpuInfo(
                name=name,
                vram_bytes=vram_bytes,
                vram_source="nvidia-smi" if vram_bytes is not None else None,
            )
        )
    return gpus


def _detect_windows_gpus() -> list[GpuInfo]:
    output = _run_command(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | "
            "Select-Object Name,AdapterRAM | ConvertTo-Csv -NoTypeInformation",
        ]
    )
    if not output:
        return []

    gpus: list[GpuInfo] = []
    for line in output.splitlines()[1:]:
        if not line:
            continue
        columns = next(
            (row for row in _parse_csv_rows(line)),
            None,
        )
        if not columns or len(columns) < 2:
            continue
        name = columns[0].strip('"')
        adapter_ram = columns[1].strip('"')
        vram_bytes: Optional[int] = None
        try:
            vram_bytes = int(adapter_ram)
        except (TypeError, ValueError):
            pass
        gpus.append(
            GpuInfo(
                name=name,
                vram_bytes=vram_bytes,
                vram_source="Windows device information" if vram_bytes else None,
            )
        )
    return gpus


def _parse_csv_rows(line: str) -> list[list[str]]:
    import csv
    import io

    return list(csv.reader(io.StringIO(line)))


def detect_gpus() -> list[GpuInfo]:
    nvidia_gpus = _detect_nvidia_gpus()
    if nvidia_gpus:
        return nvidia_gpus
    if platform.system() == "Windows":
        return _detect_windows_gpus()
    return []


def detect_hardware() -> HardwareInfo:
    disk_path = os.getenv("SystemDrive", "C:\\") if platform.system() == "Windows" else os.path.abspath(os.sep)
    warnings: list[str] = []

    try:
        free_disk_bytes = shutil.disk_usage(disk_path).free
    except OSError:
        free_disk_bytes = None
        warnings.append("Free disk space could not be detected.")

    system_ram = _detect_system_ram()
    if system_ram is None:
        warnings.append("System RAM could not be detected.")

    gpus = detect_gpus()
    if not gpus:
        warnings.append("No GPU was detected. A CPU-only setup may be available.")

    cpu_name = os.getenv("PROCESSOR_IDENTIFIER")
    if not cpu_name:
        cpu_name = platform.processor() or "Unknown CPU"

    return HardwareInfo(
        os_name=platform.system() or "Unknown",
        os_version=platform.release(),
        architecture=platform.machine() or "Unknown",
        cpu_name=cpu_name,
        logical_cpus=os.cpu_count() or 1,
        system_ram_bytes=system_ram,
        free_disk_bytes=free_disk_bytes,
        disk_path=disk_path,
        gpus=gpus,
        detection_warnings=warnings,
    )
