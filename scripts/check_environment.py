#!/usr/bin/env python3
"""Environment check script — verifies all dependencies and GPU availability.

Usage:
    python scripts/check_environment.py

Prints a table of what's available, what's missing, and exact install commands.
"""

import sys
import os
import platform
import subprocess


def check_import(module_name: str, package_name: str = None, version_attr: str = "__version__"):
    """Try importing a module. Returns (ok, version_or_error)."""
    try:
        mod = __import__(module_name)
        version = getattr(mod, version_attr, "installed")
        return True, str(version)
    except ImportError as e:
        return False, str(e)


REQUIRED = [
    ("numpy", "numpy", "numpy"),
    ("torch", "torch", "torch"),
    ("torchvision", "torchvision", "torchvision"),
    ("transformers", "transformers", "transformers"),
    ("PIL", "Pillow", "PIL"),
    ("yaml", "pyyaml", "yaml"),
    ("tqdm", "tqdm", "tqdm"),
]

OPTIONAL = [
    ("rasterio", "rasterio", "rasterio"),
    ("cv2", "opencv-python", "cv2"),
    ("peft", "peft", "peft"),
    ("accelerate", "accelerate", "accelerate"),
    ("sklearn", "scikit-learn", "sklearn"),
    ("fastapi", "fastapi", "fastapi"),
    ("uvicorn", "uvicorn", "uvicorn"),
    ("streamlit", "streamlit", "streamlit"),
    ("pandas", "pandas", "pandas"),
]


def check_gpu():
    """Check GPU availability and VRAM."""
    try:
        import torch
        if not torch.cuda.is_available():
            return "No CUDA GPU detected (will use CPU)"
        n = torch.cuda.device_count()
        info = []
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / 1e9
            info.append(f"GPU {i}: {props.name} ({vram_gb:.1f} GB VRAM)")
        return "\n    ".join(info)
    except Exception as e:
        return f"Could not check GPU: {e}"


def check_disk():
    """Check available disk space."""
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        return f"{free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total"
    except Exception:
        return "unknown"


def check_ram():
    """Check available RAM."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return f"{mem.available / 1e9:.1f} GB free / {mem.total / 1e9:.1f} GB total"
    except ImportError:
        return "psutil not installed (run: pip install psutil)"


def main():
    print("\n" + "=" * 60)
    print("  SatQuery AI — Environment Check")
    print("=" * 60)
    print(f"  Python:   {sys.version.split()[0]}")
    print(f"  Platform: {platform.system()} {platform.release()}")
    print(f"  CWD:      {os.getcwd()}")
    print()

    print("  GPU:")
    print(f"    {check_gpu()}")
    print(f"  RAM: {check_ram()}")
    print(f"  Disk: {check_disk()}")
    print()

    print("  Required packages:")
    missing_required = []
    for module, package, display in REQUIRED:
        ok, info = check_import(module)
        status = f"✓ {info}" if ok else f"✗ MISSING"
        print(f"    {display:<20} {status}")
        if not ok:
            missing_required.append(package)

    print()
    print("  Optional packages:")
    for module, package, display in OPTIONAL:
        ok, info = check_import(module)
        status = f"✓ {info}" if ok else "  not installed"
        print(f"    {display:<20} {status}")

    print()
    if missing_required:
        print(f"  ⚠ Missing required packages: {', '.join(missing_required)}")
        print(f"  Install with:")
        print(f"    pip install {' '.join(missing_required)}")
        sys.exit(1)
    else:
        print("  ✓ All required packages present.")
    
    # Check satquery imports
    print()
    print("  SatQuery AI imports:")
    sys.path.insert(0, os.path.abspath("."))
    satquery_checks = [
        "src.common.schemas",
        "src.common.constants",
        "src.preprocessing.normalize",
        "src.preprocessing.geotiff_utils",
        "src.agent.controller",
        "src.models.vqa_model",
        "src.models.change_model",
        "src.models.fusion_model",
    ]
    all_ok = True
    for mod in satquery_checks:
        try:
            __import__(mod)
            print(f"    {mod:<40} ✓")
        except Exception as e:
            print(f"    {mod:<40} ✗ {e}")
            all_ok = False
    
    print()
    if all_ok:
        print("  ✓ SatQuery AI packages import successfully.")
        print("  Ready to run smoke test: python scripts/smoke_test.py")
    else:
        print("  ✗ Some SatQuery AI imports failed — check errors above.")
        sys.exit(1)

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
