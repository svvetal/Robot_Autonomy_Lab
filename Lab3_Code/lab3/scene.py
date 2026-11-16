"""Generate the Lab 3 MuJoCo scene.

Starts from the Lab 2 shelf model (with measured shelf heights), removes the
single Lab 2 'Block', and adds:
  * 9 grey grid tiles (visual only, no collision) for the 3x3 placement grid
  * 8 free cubes (4 red, 4 blue) at their storage slots (Figure 1)
  * the 'lab3_tool' site at the fingertip centre (same offset as Lab 2)
"""
from pathlib import Path
import ast
import xml.etree.ElementTree as ET

from .shelf_heights import ORIGINAL_TOP_M, load_heights
from .config import BLOCK_HALF
from .config import TABLE_TOP
from .layout import GRID_CELLS, INITIAL_BLOCKS, RGBA, TABLE_TOP_Z, cell_xyz

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_MJCF = ROOT / 'franka_emika_panda' / 'panda_torque_table_shelves.xml'
GENERATED_MJCF = ROOT / 'franka_emika_panda' / 'lab3_runtime_scene.xml'
TOOL_SITE = 'lab3_tool'


def _fix_body_name(body):
    """The starter XML writes names as "{'Name'}"; normalise to Name."""
    name = body.get('name', '')
    if name.startswith("{'") and name.endswith("'}"):
        value = ast.literal_eval(name)
        if isinstance(value, set) and len(value) == 1:
            body.set('name', str(next(iter(value))))


def _apply_shelf_heights(worldbody, heights):
    for board, nominal_top in ORIGINAL_TOP_M.items():
        delta = heights[board] - nominal_top
        if abs(delta) < 1e-12:
            continue
        for name in [board] + [board + 'Supp' + k for k in ('1', '2', '3', 'B')]:
            body = next(b for b in worldbody.findall('body') if b.get('name') == name)
            xyz = body.get('pos').split()
            xyz[2] = repr(float(xyz[2]) + delta)
            body.set('pos', ' '.join(xyz))


def prepare_scene(output=GENERATED_MJCF, heights=None, blocks=INITIAL_BLOCKS):
    """Write the Lab 3 runtime scene. `blocks` = [(name, colour, cell), ...]."""
    heights = load_heights() if heights is None else heights
    tree = ET.parse(ORIGINAL_MJCF)
    root = tree.getroot()
    worldbody = root.find('worldbody')
    for body in root.iter('body'):
        _fix_body_name(body)
    _apply_shelf_heights(worldbody, heights)

    # Measured table height (table_layout.json) shifts the table plane.
    if abs(TABLE_TOP_Z - TABLE_TOP) > 1e-9:
        table = next(b for b in worldbody.findall('body') if b.get('name') == 'TablePlane')
        xyz = table.get('pos').split()
        xyz[2] = repr(float(xyz[2]) + TABLE_TOP_Z - TABLE_TOP)
        table.set('pos', ' '.join(xyz))

    # Remove the single Lab 2 cube.
    for body in list(worldbody.findall('body')):
        if body.get('name') == 'Block':
            worldbody.remove(body)

    # Grid tiles: thin, visual only (contype/conaffinity 0) so they never collide.
    for cell in GRID_CELLS:
        x, y, _ = cell_xyz(cell)
        ET.SubElement(worldbody, 'geom', {
            'name': f'tile_{cell}', 'type': 'box', 'pos': f'{x} {y} {TABLE_TOP_Z + 0.0001}',
            'size': f'{BLOCK_HALF + 0.004} {BLOCK_HALF + 0.004} 0.0002',
            'rgba': '0.75 0.75 0.75 1', 'contype': '0', 'conaffinity': '0'})

    # Free cubes.
    for name, colour, cell in blocks:
        x, y, z = cell_xyz(cell)
        body = ET.SubElement(worldbody, 'body', {'name': name, 'pos': f'{x} {y} {z}'})
        ET.SubElement(body, 'freejoint', {'name': f'{name}_free'})
        r, g, b, a = RGBA[colour]
        ET.SubElement(body, 'geom', {
            'name': f'{name}_geom', 'type': 'box', 'density': '20',
            'size': f'{BLOCK_HALF} {BLOCK_HALF} {BLOCK_HALF}',
            'rgba': f'{r} {g} {b} {a}', 'friction': '1.0 0.005 0.0001'})

    hand = next(b for b in root.iter('body') if b.get('name') == 'hand')
    if not any(s.get('name') == TOOL_SITE for s in hand.findall('site')):
        ET.SubElement(hand, 'site', {'name': TOOL_SITE, 'pos': '0 0 0.103',
                                     'size': '0.006', 'rgba': '1 0.55 0.2 0.45'})

    visual = root.find('visual')
    if visual is None:
        visual = ET.SubElement(root, 'visual')
    glob = visual.find('global')
    if glob is None:
        glob = ET.SubElement(visual, 'global')
    glob.set('offwidth', '960')
    glob.set('offheight', '720')

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding='utf-8', xml_declaration=True)
    return output
