import subprocess


def get_npu_usage() -> str:
    try:
        result = subprocess.run(
            ["sudo", "cat", "/sys/kernel/debug/rknpu/load"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            return result.stdout.strip().replace("Core", "").removeprefix("NPU load: ")
    except Exception:
        pass
    return "0: 0%"
