"""Shelf-height configuration for the hardware interface."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEIGHTS_PATH = ROOT / 'shelf_heights.json'
KEYS = ('left_bottom_mm', 'right_bottom_mm', 'right_middle_mm', 'right_top_mm')

def load_measured_heights(path=HEIGHTS_PATH):
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError('Cannot read measured shelf heights from {}: {}'.format(path, exc)) from exc
    if not isinstance(raw, dict) or set(raw) != set(KEYS):
        raise ValueError('Expected exactly four measured shelf heights: {}'.format(KEYS))
    data = {}
    for key in KEYS:
        v = raw[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError('{} must be a finite number of millimetres'.format(key))
        if not 60 <= v <= 900:
            raise ValueError('{}={} outside 60-900 mm; verify the base-frame datum'.format(key, v))
        data[key] = float(v)
    if not (data['right_middle_mm'] - data['right_bottom_mm'] > 55 and
            data['right_top_mm'] - data['right_middle_mm'] > 55):
        raise ValueError('Right shelf levels must be ordered and separated by >55 mm')
    return data

def heights_fingerprint(data):
    """Hash the measured height values in a stable order."""
    canonical = json.dumps({k: float(data[k]) for k in KEYS},
                           sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()

def assert_matching_heights(doc, path=HEIGHTS_PATH):
    current = load_measured_heights(path)
    if heights_fingerprint(current) != doc.get('shelf_heights_sha256'):
        raise ValueError('STALE HARDWARE PLAN: shelf_heights.json differs from plan; '
                         're-run hardware_main.py plan AND validate before robot execution')
    return current
