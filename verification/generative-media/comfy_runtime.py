"""Process-local INT8 compatibility for PyTorch 2.14 / ROCm 7.2 gfx1030."""


def int8_accumulate(torch, a, b):
    """Exact INT8 dot products without hipBLASLt or RDNA2-unsafe Triton."""
    if a.dtype != torch.int8 or b.dtype != torch.int8:
        raise ValueError("INT8 matrices required")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0] or a.device != b.device:
        raise ValueError("compatible matrix shapes and devices required")
    result = torch.zeros((a.shape[0], b.shape[1]), dtype=torch.int32, device=a.device)
    # Each partial has <=512 products of magnitude <=16384. Every intermediate
    # integer is exactly representable in FP32; combine partials in INT32, not
    # FP32. Fixed tiles bound temporary memory independently of model width.
    for m in range(0, a.shape[0], 256):
        for n in range(0, b.shape[1], 1024):
            tile = result[m:m + 256, n:n + 1024]
            for k in range(0, a.shape[1], 512):
                left = a[m:m + 256, k:k + 512].float()
                right = b[k:k + 512, n:n + 1024].float()
                tile.add_((left @ right).to(torch.int32))
    return result


def configure(torch, eager):
    if not torch.version.hip:
        return
    torch.backends.cuda.preferred_blas_library("cublas")
    original = eager._int8_matmul_accumulate

    def accumulate(a, b):
        if a.device.type == "cuda" and torch.cuda.get_device_properties(
            a.device
        ).gcnArchName.split(":", 1)[0] == "gfx1030":
            return int8_accumulate(torch, a, b)
        return original(a, b)

    # Both the public Python API and torch.ops tensor dispatch reach this
    # primitive. Keep upstream ConvRot, activation quantization and scaling.
    eager._int8_matmul_accumulate = accumulate


def main():
    import runpy
    import sys
    from pathlib import Path
    import torch
    from comfy_kitchen.backends.eager import quantization

    configure(torch, quantization)
    device = torch.cuda.get_device_properties(0)
    if torch.cuda.device_count() != 1 or "V620" not in device.name:
        raise RuntimeError(f"Expected only the headless V620, got {device}")
    print(f"Qualification device: {device}; INT8 GEMM: exact tiled FP32", flush=True)
    entry = sys.argv.pop(1)
    sys.argv[0] = entry
    sys.path.insert(0, str(Path(entry).parent))
    runpy.run_path(entry, run_name="__main__")


if __name__ == "__main__":
    main()
