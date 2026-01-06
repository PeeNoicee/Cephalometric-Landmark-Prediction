"""Build a fast-starting Windows executable for the Cephalometric AI app.

Usage (Windows PowerShell or CMD):
    python build_single_exe.py

The script will:
1. Invoke PyInstaller in onedir/GUI mode on web/dashboard.py
2. Bundle checkpoints, dataset, and src directories so the model + configs ship with the exe
3. Copy the generated folder into the release/ folder for plug-and-play distribution

Note: Uses --onedir for faster startup (no decompression overhead).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
ENTRY_SCRIPT = PROJECT_ROOT / "web" / "dashboard.py"
OUTPUT_NAME = "CephalometricAI"
RELEASE_DIR = PROJECT_ROOT / "release"
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build"

PYINSTALLER = [sys.executable, "-m", "PyInstaller"]

DATA_TARGETS = [
    (PROJECT_ROOT / "checkpoints", "checkpoints"),
    (PROJECT_ROOT / "dataset", "dataset"),
    (PROJECT_ROOT / "src", "src"),
]

def _format_add_data(source: Path, target: str) -> str:
    sep = ";" if os.name == "nt" else ":"
    return f"{source}{sep}{target}"

def build_single_exe() -> None:
    if not ENTRY_SCRIPT.exists():
        raise FileNotFoundError(f"Entry script not found: {ENTRY_SCRIPT}")

    add_data_args: list[str] = []
    for src, tgt in DATA_TARGETS:
        if src.exists():
            add_data_args.extend(["--add-data", _format_add_data(src, tgt)])

    cmd = (
        PYINSTALLER
        + [
            "--clean",
            "--onedir",  # Faster startup - no decompression
            "--windowed",
            "--name",
            OUTPUT_NAME,
            "--noconfirm",
        ]
        + add_data_args
        + [str(ENTRY_SCRIPT)]
    )

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)

    exe_dir = DIST_DIR / OUTPUT_NAME
    if not exe_dir.exists():
        raise FileNotFoundError(f"PyInstaller finished but {exe_dir} was not found.")

    RELEASE_DIR.mkdir(exist_ok=True)
    target_dir = RELEASE_DIR / OUTPUT_NAME
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(exe_dir, target_dir)
    print(f"Application copied to {target_dir}")
    print(f"Run: {target_dir / f'{OUTPUT_NAME}.exe'}")

if __name__ == "__main__":
    build_single_exe()
