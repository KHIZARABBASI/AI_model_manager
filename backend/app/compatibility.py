from pydantic import BaseModel, Field

from app.hardware import HardwareInfo


GIB = 1024 * 1024 * 1024


class CompatibilityRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=300)
    repository_size_bytes: int | None = Field(default=None, ge=0)


class CompatibilityResponse(BaseModel):
    model_id: str
    status: str
    reasons: list[str]
    assumptions: list[str]
    estimated_runtime_memory_bytes: int | None
    estimated_download_and_install_bytes: int | None
    available_memory_bytes: int | None
    available_disk_bytes: int | None


def estimate_compatibility(
    hardware: HardwareInfo,
    request: CompatibilityRequest,
) -> CompatibilityResponse:
    assumptions = [
        "Repository size is treated as an upper-bound weight size, not a measured quantized weight size.",
        "The estimate adds 25 percent overhead for model weights and a 2 GiB runtime/KV-cache allowance.",
        "Disk allowance includes the repository size plus 5 GiB for installation overhead.",
    ]

    repository_size = request.repository_size_bytes
    if repository_size is not None and repository_size == 0:
        repository_size = None

    nvidia_gpus = [
        gpu
        for gpu in hardware.gpus
        if gpu.vram_bytes is not None
        and gpu.vram_source == "nvidia-smi"
    ]
    nvidia_vram = max(
        (gpu.vram_bytes for gpu in nvidia_gpus if gpu.vram_bytes is not None),
        default=None,
    )
    available_memory = nvidia_vram or hardware.system_ram_bytes

    runtime_memory: int | None = None
    if repository_size is not None:
        runtime_memory = int(repository_size * 1.25) + (2 * GIB)

    disk_requirement: int | None = None
    if repository_size is not None:
        disk_requirement = repository_size + (5 * GIB)

    reasons: list[str] = []
    if repository_size is None:
        reasons.append(
            "Repository size is unavailable, so runtime and disk requirements cannot be estimated reliably."
        )
    if hardware.system_ram_bytes is None:
        reasons.append("System RAM could not be detected.")
    if hardware.free_disk_bytes is None:
        reasons.append("Free disk space could not be detected.")

    if (
        repository_size is None
        or hardware.system_ram_bytes is None
        or hardware.free_disk_bytes is None
    ):
        status = "Not enough information"
        if repository_size is not None:
            reasons.append(
                "This estimate will be more reliable once hardware detection provides RAM and disk data."
            )
    else:
        assert available_memory is not None
        assert runtime_memory is not None
        assert disk_requirement is not None

        disk_short = hardware.free_disk_bytes < disk_requirement
        memory_short = runtime_memory > available_memory
        comfortable = runtime_memory <= int(available_memory * 0.8)

        if disk_short:
            reasons.append(
                "Free disk space is below the estimated download and installation allowance."
            )
        if memory_short:
            reasons.append(
                "Estimated runtime memory exceeds detected available memory."
            )
        elif comfortable:
            reasons.append(
                "Estimated runtime memory leaves a 20 percent headroom margin."
            )
        else:
            reasons.append(
                "Estimated runtime memory is within detected memory but has limited headroom."
            )

        if nvidia_vram is not None:
            reasons.append("NVIDIA VRAM was used as the available memory estimate.")
        else:
            reasons.append("System RAM was used because verified NVIDIA VRAM was unavailable.")

        if memory_short:
            status = "Likely to run out of memory"
        elif comfortable and not disk_short:
            status = "Likely to run comfortably"
        else:
            status = "May run, but could be slow"

    return CompatibilityResponse(
        model_id=request.model_id,
        status=status,
        reasons=reasons,
        assumptions=assumptions,
        estimated_runtime_memory_bytes=runtime_memory,
        estimated_download_and_install_bytes=disk_requirement,
        available_memory_bytes=available_memory,
        available_disk_bytes=hardware.free_disk_bytes,
    )
