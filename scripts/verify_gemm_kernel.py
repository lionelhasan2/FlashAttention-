from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = ROOT / "kernels" / "gemm.cu"


def build_kernel_harness(kernel_src: str, tmp_dir: Path) -> Path:
    harness = tmp_dir / "gemm_verify.cu"
    harness.write_text(f'''
#include <cuda_runtime.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

{kernel_src}

int main(int argc, char** argv) {{
    if (argc != 3) {{
        std::cerr << "Usage: gemm_verify <width> <output_path>\\n";
        return 1;
    }}

    int width = std::atoi(argv[1]);
    std::string output_path = argv[2];
    size_t total = static_cast<size_t>(width) * width;

    std::vector<float> hM(total), hN(total), hP(total);

    std::ifstream inM("/tmp/gemm_a.bin", std::ios::binary);
    std::ifstream inN("/tmp/gemm_b.bin", std::ios::binary);
    if (!inM || !inN) {{
        std::cerr << "Missing input binary files\\n";
        return 2;
    }}

    inM.read(reinterpret_cast<char*>(hM.data()), total * sizeof(float));
    inN.read(reinterpret_cast<char*>(hN.data()), total * sizeof(float));

    float* dM;
    float* dN;
    float* dP;
    cudaMalloc(&dM, total * sizeof(float));
    cudaMalloc(&dN, total * sizeof(float));
    cudaMalloc(&dP, total * sizeof(float));

    cudaMemcpy(dM, hM.data(), total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dN, hN.data(), total * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x,
                (width + threads.y - 1) / threads.y);

    NaiveMatrixMulKernel<<<blocks, threads>>>(dM, dN, dP, width);
    cudaDeviceSynchronize();

    cudaMemcpy(hP.data(), dP, total * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dM);
    cudaFree(dN);
    cudaFree(dP);

    std::ofstream out(output_path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(hP.data()), total * sizeof(float));
    return 0;
}}
''')
    return harness


def verify_case(width: int) -> tuple[bool, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this validation. Use a GPU-backed Colab runtime.")

    torch.manual_seed(0)
    a = torch.randn(width, width, dtype=torch.float32, device="cuda")
    b = torch.randn(width, width, dtype=torch.float32, device="cuda")
    expected = torch.matmul(a, b)

    a.cpu().numpy().astype(np.float32).tofile("/tmp/gemm_a.bin")
    b.cpu().numpy().astype(np.float32).tofile("/tmp/gemm_b.bin")

    with tempfile.TemporaryDirectory(prefix="gemm_verify_") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        kernel_src = KERNEL_PATH.read_text()
        harness = build_kernel_harness(kernel_src, tmp_dir)
        binary = tmp_dir / "gemm_verify"

        compile_result = subprocess.run(
            ["nvcc", "-O3", "--std=c++17", str(harness), "-o", str(binary)],
            capture_output=True,
            text=True,
        )
        if compile_result.returncode != 0:
            raise RuntimeError(f"nvcc compilation failed:\n{compile_result.stderr}")

        output_path = tmp_dir / "result.bin"
        run_result = subprocess.run(
            [str(binary), str(width), str(output_path)],
            capture_output=True,
            text=True,
        )
        if run_result.returncode != 0:
            raise RuntimeError(f"Kernel execution failed:\n{run_result.stderr}")

        actual = np.fromfile(output_path, dtype=np.float32).reshape(width, width)
        expected_np = expected.detach().cpu().numpy()
        max_abs_err = float(np.max(np.abs(actual - expected_np)))
        is_ok = bool(np.allclose(actual, expected_np, atol=1e-4, rtol=1e-4))

    return is_ok, max_abs_err


def run_suite() -> bool:
    print("Verifying naive GEMM kernel against torch.matmul...")
    sizes = [4, 8, 16]
    all_passed = True

    for size in sizes:
        ok, max_err = verify_case(size)
        print(f"size={size}x{size} | ok={ok} | max_abs_err={max_err:.6e}")
        if not ok:
            all_passed = False

    if all_passed:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")

    return all_passed


if __name__ == "__main__":
    raise SystemExit(0 if run_suite() else 1)
