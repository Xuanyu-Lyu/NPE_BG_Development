#!/usr/bin/env python3
"""Report whether this repository's Python and R dependencies are available."""

from __future__ import annotations

import importlib
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path


PYTHON_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "scikit-learn": "sklearn",
    "joblib": "joblib",
    "torch": "torch",
    "sbi": "sbi",
    "jupyterlab": "jupyterlab",
    "ipykernel": "ipykernel",
}


def check_python() -> bool:
    print(f"Python {platform.python_version()} ({platform.machine()})")
    ok = sys.version_info >= (3, 10)
    if not ok:
        print("  ERROR: sbi requires Python 3.10 or newer")

    for distribution, module in PYTHON_PACKAGES.items():
        try:
            importlib.import_module(module)
            version = metadata.version(distribution)
            print(f"  OK    {distribution} {version}")
        except Exception as exc:  # report binary-loader failures as well as absence
            ok = False
            print(f"  ERROR {distribution}: {exc}")

    if importlib.util.find_spec("torch") is not None:
        import torch

        print(f"  INFO  MPS available: {torch.backends.mps.is_available()}")
        print(f"  INFO  CUDA available: {torch.cuda.is_available()}")
    return ok


def check_r() -> bool:
    command = [
        "Rscript",
        "-e",
        (
            'for (p in c("OpenMx", "MASS")) '
            'cat(p, if (requireNamespace(p, quietly=TRUE)) '
            'as.character(packageVersion(p)) else "MISSING", "\\n")'
        ),
    ]
    print("R packages (needed only for ace_npe/04_fit_openmx_reference.R)")
    try:
        result = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=30
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"  WARNING: could not run R dependency check: {exc}")
        return False

    print("  " + result.stdout.strip().replace("\n", "\n  "))
    return result.returncode == 0 and "MISSING" not in result.stdout


def check_univariate_source() -> bool:
    module = (
        Path(__file__).resolve().parents[1]
        / "sempgs_npe"
        / "univariate"
        / "core_simulation.py"
    )
    print("Univariate SEM-PGS source")
    if module.exists():
        print(f"  OK    {module}")
        return True
    print("  WARNING: core_simulation.py is missing; this workflow cannot run yet")
    return False


def main() -> int:
    python_ok = check_python()
    print()
    check_r()
    print()
    check_univariate_source()
    return 0 if python_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
