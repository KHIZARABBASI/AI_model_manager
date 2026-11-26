from app.hardware import GpuInfo, HardwareInfo, detect_hardware


def test_detect_hardware_returns_documented_shape() -> None:
    hardware = detect_hardware()

    assert isinstance(hardware, HardwareInfo)
    assert hardware.os_name
    assert hardware.cpu_name
    assert hardware.logical_cpus >= 1
    assert isinstance(hardware.gpus, list)


def test_gpu_model_accepts_missing_vram() -> None:
    gpu = GpuInfo(name="Example GPU")

    assert gpu.name == "Example GPU"
    assert gpu.vram_bytes is None
