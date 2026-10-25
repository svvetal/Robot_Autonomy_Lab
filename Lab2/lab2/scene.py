"""Open original, unmodified shelf MJCF; add tool site and clean malformed body names.

Original file has names like "{'Block'}" due to set formatting in starter RobotUtil.
We repair names only in a generated scene. All shelf geoms remain at identical poses/sizes.
"""
from pathlib import Path
import ast
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_MJCF = ROOT / 'franka_emika_panda' / 'panda_torque_table_shelves.xml'
GENERATED_MJCF = ROOT / 'franka_emika_panda' / 'lab2_runtime_shelves.xml'


def prepare_scene(output=GENERATED_MJCF):
    tree = ET.parse(ORIGINAL_MJCF)
    root = tree.getroot()
    for body in root.iter('body'):
        name = body.get('name', '')
        if name.startswith("{'" ) and name.endswith("'}"):
            value = ast.literal_eval(name)
            if isinstance(value, set) and len(value) == 1:
                body.set('name', str(next(iter(value))))
    hand = next((b for b in root.iter('body') if b.get('name') == 'hand'), None)
    if hand is None:
        raise RuntimeError('Expected Franka hand body missing in starter MJCF')
    if not any(site.get('name') == 'lab2_tool' for site in hand.findall('site')):
        ET.SubElement(hand, 'site', {'name': 'lab2_tool', 'pos': '0 0 0.103',
                                     'size': '0.006', 'rgba': '1 0.55 0.2 0.45'})
    # MuJoCo's default offscreen frame buffer may be only 640x480.
    # Enlarging the *render* buffer does not alter physics or shelf geometry.
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
    """Return deterministic shelf dimensions/poses, independent of MuJoCo."""
    tree = ET.parse(xml)
    result = {}
    for b in tree.getroot().find('worldbody').findall('body'):
        name = b.get('name', '').strip("{}'")
        if 'Shelf' in name or name == 'TablePlane':
            geom = b.find('geom')
            result[name] = (tuple(map(float, b.get('pos').split())),
                            tuple(map(float, geom.get('size').split())))
    return result
