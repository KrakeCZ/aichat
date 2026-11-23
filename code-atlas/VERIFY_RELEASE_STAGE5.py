#!/usr/bin/env python3
"""Compatibility name: verify the CURRENT installed release, not the old Stage 5 ZIP."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name('VERIFY_RELEASE_STAGE6.py')),run_name='__main__')
