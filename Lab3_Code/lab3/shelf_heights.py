"""Measured shelf heights in the Panda base frame."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEIGHTS_FILE = ROOT / 'shelf_heights.json'

# Reference heights in the unmodified shelf XML. Edit shelf_heights.json instead.
ORIGINAL_TOP_M = {
    'LShelfBottom': 0.1475,
    'RShelfBottom': 0.1475,
    'RShelfMiddle': 0.3475,
    'RShelfTop': 0.5475,
}
JSON_TO_BODY = {
    'left_bottom_mm': 'LShelfBottom',
    'right_bottom_mm': 'RShelfBottom',
    'right_middle_mm': 'RShelfMiddle',
    'right_top_mm': 'RShelfTop',
}

def load_heights(path=HEIGHTS_FILE):
    """Load and validate the four shelf-top heights in metres."""
    path = Path(path)
    try:
        values = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError as exc:
        raise ValueError(f'Cannot find shelf heights: {path}') from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f'Invalid JSON in {path}: {exc}') from exc
    if not isinstance(values, dict) or set(values) != set(JSON_TO_BODY):
        raise ValueError(f'{path}: expected exactly these four keys: '
                         f'{", ".join(JSON_TO_BODY)}')
    result = {}
    for key, body in JSON_TO_BODY.items():
        val = values[key]
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
            raise ValueError(f'{path}: {key} must be a finite number in millimetres')
        top = val / 1000.0
        if not 0.06 <= top <= 0.90:
            raise ValueError(f'{path}: {key}={val} mm is outside plausible 60–900 mm; '
                             'check the reference frame and units')
        result[body] = top

    bottom, mid, upper = (result[k] for k in
                          ('RShelfBottom', 'RShelfMiddle', 'RShelfTop'))
    if not (mid - bottom > 0.055 and upper - mid > 0.055):
        raise ValueError(f'{path}: right shelf top heights must ascend with '
                         '>55 mm separation between levels')
    return result

def heights_as_mm(heights):
    """Return shelf heights using configuration keys and millimetres."""
    return {key: round(1000 * heights[body], 3) for key, body in JSON_TO_BODY.items()}
