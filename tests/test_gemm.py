from __future__ import annotations

from scripts.verify_gemm_kernel import run_suite


if __name__ == "__main__":
    raise SystemExit(0 if run_suite() else 1)
