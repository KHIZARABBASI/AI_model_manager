from app.compatibility import CompatibilityRequest, estimate_compatibility
from app.hardware import GpuInfo, HardwareInfo


def hardware(
    *,
    ram_gib: int | None = 16,
    disk_gib: int = 200,
    gpus: list[GpuInfo] | None = None,
) -> HardwareInfo:
    gib = 1024 * 1024 * 1024
    return HardwareInfo(
        os_name="Windows",
        os_version="11",
        architecture="AMD64",
        cpu_name="Test CPU",
        logical_cpus=8,
        system_ram_bytes=ram_gib * gib if ram_gib is not None else None,
        free_disk_bytes=disk_gib * gib,
        disk_path="C:",
        gpus=gpus or [],
        detection_warnings=[],
    )


def test_missing_repository_size_is_not_enough_information() -> None:
    result = estimate_compatibility(
        hardware(),
        CompatibilityRequest(model_id="example/model"),
    )

    assert result.status == "Not enough information"
    assert result.estimated_runtime_memory_bytes is None


def test_large_model_is_likely_out_of_memory() -> None:
    result = estimate_compatibility(
        hardware(ram_gib=8),
        CompatibilityRequest(
            model_id="example/large-model",
            repository_size_bytes=8 * 1024 * 1024 * 1024,
        ),
    )

    assert result.status == "Likely to run out of memory"


def test_nvidia_vram_is_preferred_over_system_ram() -> None:
    result = estimate_compatibility(
        hardware(
            ram_gib=32,
            gpus=[
                GpuInfo(
                    name="Test GPU",
                    vram_bytes=8 * 1024 * 1024 * 1024,
                    vram_source="nvidia-smi",
                )
            ],
        ),
        CompatibilityRequest(
            model_id="example/small-model",
            repository_size_bytes=3 * 1024 * 1024 * 1024,
        ),
    )

    assert result.available_memory_bytes == 8 * 1024 * 1024 * 1024
    assert result.status == "Likely to run comfortably"


def test_insufficient_disk_makes_status_may_run() -> None:
    result = estimate_compatibility(
        hardware(ram_gib=32, disk_gib=6),
        CompatibilityRequest(
            model_id="example/model",
            repository_size_bytes=4 * 1024 * 1024 * 1024,
        ),
    )

    assert result.status == "May run, but could be slow"
    assert any("disk" in reason.lower() for reason in result.reasons)
