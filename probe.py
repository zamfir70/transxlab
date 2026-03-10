"""TransXLab environment probe. Outputs JSON to stdout.

Called by the Rust binary via subprocess. No arguments needed.
"""
import json
import sys


def probe():
    result = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "torch_available": False,
        "cuda_available": False,
        "gpus": [],
        "error": None,
    }

    try:
        import torch
        result["torch_available"] = True
        result["cuda_available"] = torch.cuda.is_available()

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                vram_bytes = props.total_memory
                result["gpus"].append({
                    "name": props.name,
                    "vram_bytes": vram_bytes,
                    "vram_gb": round(vram_bytes / (1024 ** 3), 1),
                    "cuda_version": torch.version.cuda or "unknown",
                    "torch_version": torch.__version__,
                    "bf16_supported": torch.cuda.is_bf16_supported(),
                    "cudnn_available": torch.backends.cudnn.is_available(),
                    "compute_capability": f"{props.major}.{props.minor}",
                })
    except ImportError:
        pass
    except Exception as e:
        result["error"] = str(e)

    print(json.dumps(result))


if __name__ == "__main__":
    probe()
