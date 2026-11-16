"""Table layout: 3x3 grid, side storage slots, initial setup and target patterns.

Conventions (robot base frame, robot looking along +X):
  * Grid row 0 is the row FARTHEST from the robot (top row of the handout figure).
  * Grid col 0 is on the robot's LEFT (+Y), i.e. the left column of the figure.
  * Red storage column is on the robot's left (+Y), blue on the right (-Y).

Every location (grid cell or storage slot) is a symbolic "cell".  The neighbour
tables below drive the finger-collision reasoning in the symbolic domain.
"""
import json
from pathlib import Path
import numpy as np
from .config import BLOCK_HALF, TABLE_TOP

# Nominal (simulation) layout. On the real robot, measure the table and write
# table_layout.json next to run_lab3.py: it overrides these values for the
# simulator, the planner AND the hardware plan, so they always agree.
GRID_CENTER = np.array([0.50, 0.0])
GRID_PITCH = 0.055          # 4 cm cube + 1.5 cm gap -> open fingers hit neighbours
SIDE_Y = 0.17               # storage columns at y = +/-SIDE_Y
# 8 cm spacing: open fingers (finger mesh reaches ~5.3 cm from the cube centre
# at GRIP_OPEN) clear the neighbouring storage cube on BOTH axes. 7 cm did not.
SIDE_X = (0.62, 0.54, 0.46, 0.38)
TABLE_TOP_Z = TABLE_TOP

LAYOUT_FILE = Path(__file__).resolve().parents[1] / 'table_layout.json'


def _load_override(path=LAYOUT_FILE):
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding='utf-8'))
    allowed = {'grid_center_xy', 'grid_pitch', 'side_x', 'side_y', 'table_top_z', 'note'}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f'{path}: unknown keys {sorted(unknown)}')
    if 'grid_center_xy' in raw:
        x, y = map(float, raw['grid_center_xy'])
        if not (0.35 <= x <= 0.65 and abs(y) <= 0.15):
            raise ValueError(f'{path}: grid_center_xy {raw["grid_center_xy"]} implausible')
    if 'grid_pitch' in raw and not 0.050 <= float(raw['grid_pitch']) <= 0.065:
        raise ValueError(f'{path}: grid_pitch must be 0.050-0.065 m')
    if 'side_x' in raw:
        sx = [float(v) for v in raw['side_x']]
        if len(sx) != 4 or min(abs(a - b) for a, b in zip(sx, sx[1:])) < 0.08:
            raise ValueError(f'{path}: side_x needs 4 values >= 8 cm apart')
    if 'side_y' in raw and not 0.15 <= float(raw['side_y']) <= 0.25:
        raise ValueError(f'{path}: side_y must be 0.15-0.25 m')
    if 'table_top_z' in raw and not -0.03 <= float(raw['table_top_z']) <= 0.03:
        raise ValueError(f'{path}: table_top_z must be within +/-3 cm of the base')
    return raw


_OV = _load_override()
if 'grid_center_xy' in _OV:
    GRID_CENTER = np.array([float(v) for v in _OV['grid_center_xy']])
GRID_PITCH = float(_OV.get('grid_pitch', GRID_PITCH))
SIDE_X = tuple(float(v) for v in _OV.get('side_x', SIDE_X))
SIDE_Y = float(_OV.get('side_y', SIDE_Y))
TABLE_TOP_Z = float(_OV.get('table_top_z', TABLE_TOP_Z))
BLOCK_Z = TABLE_TOP_Z + BLOCK_HALF + 0.002      # resting cube-centre height


def layout_summary():
    """Everything that fixes cube positions; stored in hardware plans."""
    return {'grid_center_xy': [float(v) for v in GRID_CENTER], 'grid_pitch': GRID_PITCH,
            'side_x': list(SIDE_X), 'side_y': SIDE_Y, 'table_top_z': TABLE_TOP_Z,
            'block_z': BLOCK_Z, 'override_file': str(LAYOUT_FILE) if _OV else None}


OUTSIDE = 'out'             # pseudo-cell, permanently empty (grid border)

RED, BLUE = 'red', 'blue'
RGBA = {RED: (0.90, 0.12, 0.08, 1.0), BLUE: (0.0, 0.45, 0.75, 1.0)}


def grid_name(r: int, c: int) -> str:
    return f'g{r}{c}'


GRID_CELLS = [grid_name(r, c) for r in range(3) for c in range(3)]
RED_SLOTS = [f'sr{i}' for i in range(4)]
BLUE_SLOTS = [f'sb{i}' for i in range(4)]
ALL_CELLS = GRID_CELLS + RED_SLOTS + BLUE_SLOTS


def cell_xyz(cell: str) -> np.ndarray:
    """Cube-centre position for a cell."""
    if cell.startswith('g'):
        r, c = int(cell[1]), int(cell[2])
        x = GRID_CENTER[0] + (1 - r) * GRID_PITCH
        y = GRID_CENTER[1] + (1 - c) * GRID_PITCH
    elif cell.startswith('sr'):
        x, y = SIDE_X[int(cell[2])], +SIDE_Y
    elif cell.startswith('sb'):
        x, y = SIDE_X[int(cell[2])], -SIDE_Y
    else:
        raise ValueError(f'No position for cell {cell!r}')
    return np.array([x, y, BLOCK_Z])


STORAGE = {RED: RED_SLOTS, BLUE: BLUE_SLOTS}


def accepts(cell: str, colour: str) -> bool:
    """Grid cells take any colour; a storage column only takes its own colour."""
    if cell in GRID_CELLS:
        return True
    return cell in STORAGE[colour]


def finger_neighbours(cell: str) -> dict:
    """For each closing axis, the two cells the fingers sweep into.

    axis 'x': fingers on +X/-X sides of the cube  -> grid neighbours in rows r-1, r+1
    axis 'y': fingers on +Y/-Y sides of the cube  -> grid neighbours in cols c-1, c+1
    Storage slots are far apart, so both axes are always free.
    """
    if not cell.startswith('g'):
        return {'x': (OUTSIDE, OUTSIDE), 'y': (OUTSIDE, OUTSIDE)}
    r, c = int(cell[1]), int(cell[2])

    def nb(rr, cc):
        return grid_name(rr, cc) if 0 <= rr < 3 and 0 <= cc < 3 else OUTSIDE
    return {'x': (nb(r - 1, c), nb(r + 1, c)), 'y': (nb(r, c - 1), nb(r, c + 1))}


# ---- Initial setup (Figure 1): 4 red on the left, 4 blue on the right -------
INITIAL_BLOCKS = (
    [(f'red{i}', RED, RED_SLOTS[i]) for i in range(4)] +
    [(f'blue{i}', BLUE, BLUE_SLOTS[i]) for i in range(4)]
)   # (body name, colour, starting cell)


# ---- Target patterns (Figure 2), rows top->bottom, '.' = must be empty ------
def _parse(rows):
    m = {'B': BLUE, 'R': RED, '.': None}
    return {grid_name(r, c): m[ch] for r, row in enumerate(rows) for c, ch in enumerate(row)}


PATTERNS = {
    'pickaxe': _parse(['BBB', '.R.', '.R.']),
    'axe':     _parse(['BB.', 'BR.', '.R.']),
    'box':     _parse(['BRB', 'R.R', 'BRB']),
    'bow':     _parse(['BR.', 'B.R', 'BR.']),
}
PATTERN_ORDER = ['pickaxe', 'axe', 'box', 'bow']
