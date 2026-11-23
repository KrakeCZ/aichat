#!/usr/bin/env python3
"""Beginner-friendly launcher: select a directory or paste a path."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def choose_project() -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
        window = tk.Tk()
        window.withdraw()
        window.attributes("-topmost", True)
        selected = filedialog.askdirectory(title="Universal Code Atlas — vyber složku projektu")
        window.destroy()
        if selected:
            return selected
    except Exception:
        pass
    return input("Zadej úplnou cestu ke složce projektu: ").strip().strip('"')


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else choose_project()
    if not path:
        print("Nebyl vybrán žádný projekt.")
        raise SystemExit(1)
    project = Path(path).expanduser()
    if not project.is_dir():
        print("Složka neexistuje:", project)
        raise SystemExit(1)
    try:
        raise SystemExit(subprocess.call([sys.executable, str(ROOT / "atlas.py"), "run", str(project)]))
    except KeyboardInterrupt:
        raise SystemExit(0)
