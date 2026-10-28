"""Generate the MuJoCo scene using measured shelf heights."""
from pathlib import Path
import ast
import xml.etree.ElementTree as ET
from .shelf_heights import ORIGINAL_TOP_M, load_heights

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_MJCF = ROOT / 'franka_emika_panda' / 'panda_torque_table_shelves.xml'
GENERATED_MJCF = ROOT / 'franka_emika_panda' / 'lab2_runtime_shelves.xml'

def _fix_body_name(body):
    name = body.get('name', '')
    if name.startswith("{'" ) and name.endswith("'}"):
        value = ast.literal_eval(name)
        if isinstance(value, set) and len(value) == 1:
            body.set('name', str(next(iter(value))))

def prepare_scene(output=GENERATED_MJCF, heights=None):
    """Write the calibrated scene without changing the source XML."""
    heights = load_heights() if heights is None else heights
    if set(heights) != set(ORIGINAL_TOP_M):
        raise ValueError('Expected 4 shelf heights indexed by MuJoCo body name')
    tree = ET.parse(ORIGINAL_MJCF)
    root = tree.getroot()
    worldbody = root.find('worldbody')
    if worldbody is None:
        raise RuntimeError('Missing worldbody in shelf MJCF')
    bodies = list(root.iter('body'))
    for body in bodies:
        _fix_body_name(body)

    for board, nominal_top in ORIGINAL_TOP_M.items():
        related = [board] + [board + 'Supp' + k for k in ('1', '2', '3', 'B')]
        # Move the board and its local supports by the same Z offset.
        delta = heights[board] - nominal_top
        for name in related:
            matches = [b for b in worldbody.findall('body') if b.get('name') == name]
            if len(matches) != 1:
                raise RuntimeError(f'Expected exactly one shelf part {name}, found {len(matches)}')
            body = matches[0]
            xyz = [float(x) for x in body.get('pos').split()]
            if len(xyz) != 3:
                raise RuntimeError(f'Invalid pos for {name}')
            if name == board:
                geom = body.find('geom')
                if geom is None or geom.get('type') != 'box':
                    raise RuntimeError(f'Expected box geom on shelf board {name}')
                half_thickness = float(geom.get('size').split()[2])
                actual_orig_top = xyz[2] + half_thickness
                if abs(actual_orig_top - nominal_top) > 0.00001:
                    raise RuntimeError(
                        f'{board}: stock source model was modified or ORIGINAL_TOP_M '
                        f'was overwritten: baseline={nominal_top:.6f} m, '
                        f'original XML top={actual_orig_top:.6f} m. '
                        'Real measured heights belong ONLY in shelf_heights.json. '
                        'Restore the fixed ORIGINAL_TOP_M values in '
                        'lab2/shelf_heights.py to (0.1475, 0.1475, 0.3475, 0.5475) '
                        'and restore the uncalibrated '
                        'franka_emika_panda/panda_torque_table_shelves.xml if edited.'
                    )

            if abs(delta) > 1e-12:
                tokens = body.get('pos').split()
                tokens[2] = repr(float(tokens[2]) + delta)
                body.set('pos', ' '.join(tokens))

    hand = next((b for b in bodies if b.get('name') == 'hand'), None)
    if hand is None:
        raise RuntimeError('Expected Franka hand body missing in starter MJCF')
    if not any(site.get('name') == 'lab2_tool' for site in hand.findall('site')):
        ET.SubElement(hand, 'site', {'name': 'lab2_tool', 'pos': '0 0 0.103',
                                     'size': '0.006', 'rgba': '1 0.55 0.2 0.45'})
    visual = root.find('visual')
    if visual is None:
        visual = ET.SubElement(root, 'visual')
    global_render = visual.find('global')
    if global_render is None:
        global_render = ET.SubElement(visual, 'global')
    global_render.set('offwidth', '960')
    global_render.set('offheight', '720')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding='utf-8', xml_declaration=True)
    return output

def scene_shelf_geometry(xml=ORIGINAL_MJCF):
    """Read poses and half-sizes of shelf bodies."""
    tree = ET.parse(xml)
    result = {}
    for b in tree.getroot().find('worldbody').findall('body'):
        name = b.get('name', '').strip("{}'")
        if 'Shelf' in name or name == 'TablePlane':
            geom = b.find('geom')
            result[name] = (tuple(map(float, b.get('pos').split())),
                            tuple(map(float, geom.get('size').split())))
    return result
