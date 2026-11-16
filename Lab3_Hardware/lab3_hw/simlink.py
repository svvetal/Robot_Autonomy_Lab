"""Locate the Lab3_Code simulation package (planning/validation side only)."""
import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIM_ROOT = Path(os.environ.get('LAB3_SIM_ROOT', ROOT.parent / 'Lab3_Code')).resolve()


def use_sim():
    if not (SIM_ROOT / 'lab3' / 'simulator.py').exists():
        raise SystemExit('Cannot find the Lab3_Code simulation package at {}. '
                         'Set LAB3_SIM_ROOT=/path/to/Lab3_Code'.format(SIM_ROOT))
    if str(SIM_ROOT) not in sys.path:
        sys.path.insert(0, str(SIM_ROOT))
    return SIM_ROOT


def sim_sources_sha256():
    """Hash of everything that determines the plan (code, layout, shelves)."""
    h = hashlib.sha256()
    files = sorted((SIM_ROOT / 'lab3').rglob('*.py')) + [SIM_ROOT / 'run_lab3.py',
                                                         SIM_ROOT / 'shelf_heights.json',
                                                         SIM_ROOT / 'table_layout.json']
    for f in files:
        if f.exists():
            h.update(str(f.relative_to(SIM_ROOT)).encode())
            h.update(f.read_bytes())
    return h.hexdigest()
