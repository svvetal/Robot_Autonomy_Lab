from pathlib import Path
import ast
import xml.etree.ElementTree as ET

from .config import (
    BLOCK_HALF, BLOCK_CENTER_Z, BLOCK_NAMES, BLOCK_RGBA, GRID_CELLS,
    GRID_PITCH, TABLE_TOP, TOOL_SITE_OFFSET, grid_position, staging_position,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_MJCF = ROOT / 'franka_emika_panda' / 'panda_torque_table_shelves.xml'
GENERATED_MJCF = ROOT / 'franka_emika_panda' / 'lab3_runtime.xml'

def _float3(values):

    return ' '.join(f'{float(v):.6f}' for v in values)

def _normalize_legacy_body_names(root):

    for body in root.iter('body'):
        name = body.get('name', '')
        if name.startswith("{'") and name.endswith("'}"):
            try:
                value = ast.literal_eval(name)
            except (ValueError, SyntaxError):
                continue
            if isinstance(value, set) and len(value) == 1:
                body.set('name', str(next(iter(value))))

def _remove_legacy_block(worldbody):

    found = [body for body in worldbody.findall('body')
             if body.get('name') == 'Block']
    if len(found) != 1:
        raise RuntimeError(f'Expected one original Lab 2 Block, found {len(found)}')
    for body in found:
        worldbody.remove(body)

def _add_visual_locations(worldbody):

    marker_z = TABLE_TOP + 0.00030
    for r in range(3):
        for c in range(3):
            xyz = grid_position(r, c, z=marker_z)
            ET.SubElement(worldbody, 'site', {
                'name': f'cell_G{r}{c}',
                'type': 'box',
                'pos': _float3(xyz),
                'size': _float3([BLOCK_HALF, BLOCK_HALF, 0.00025]),
                'rgba': '0.72 0.72 0.72 0.65',
                'group': '1',
            })
    for name in BLOCK_NAMES:
        color = name.split('_')[0]
        xyz = staging_position(name, z=marker_z)
        rgba = ('0.88 0.30 0.27 0.45' if color == 'red'
                else '0.25 0.52 0.87 0.45')
        ET.SubElement(worldbody, 'site', {
            'name': f'stage_{name}',
            'type': 'box',
            'pos': _float3(xyz),
            'size': _float3([BLOCK_HALF, BLOCK_HALF, 0.00025]),
            'rgba': rgba,
            'group': '1',
        })

def _add_free_blocks(worldbody):

    for name in BLOCK_NAMES:
        color = name.split('_')[0]
        body = ET.SubElement(worldbody, 'body', {
            'name': name,
            'pos': _float3(staging_position(name)),
        })
        ET.SubElement(body, 'freejoint', {'name': f'{name}_free'})
        ET.SubElement(body, 'geom', {
            'name': f'{name}_geom',
            'type': 'box',
            'size': _float3([BLOCK_HALF] * 3),

            'density': '650',
            'friction': '0.9 0.005 0.0001',
            'rgba': BLOCK_RGBA[color],
        })

def _ensure_tool_site(root):
    hands = [body for body in root.iter('body') if body.get('name') == 'hand']
    if len(hands) != 1:
        raise RuntimeError('Expected exactly one Panda hand body')
    hand = hands[0]
    if not any(site.get('name') == 'lab3_tool' for site in hand.findall('site')):
        ET.SubElement(hand, 'site', {
            'name': 'lab3_tool',
            'pos': _float3([0, 0, TOOL_SITE_OFFSET]),
            'size': '0.006',
            'rgba': '1 0.55 0.2 0.45',
        })

def prepare_scene(output=GENERATED_MJCF):

    root = ET.parse(SOURCE_MJCF).getroot()
    worldbody = root.find('worldbody')
    if worldbody is None:
        raise RuntimeError('Source MJCF has no worldbody')
    _normalize_legacy_body_names(root)
    _remove_legacy_block(worldbody)
    _ensure_tool_site(root)
    _add_visual_locations(worldbody)
    _add_free_blocks(worldbody)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output, encoding='utf-8', xml_declaration=True)
    return output

def inspect_scene(xml=GENERATED_MJCF):

    root = ET.parse(xml).getroot()
    bodies = {body.get('name'): body for body in root.iter('body')}
    joints = {joint.get('name') for joint in root.iter('joint')}
    sites = {site.get('name') for site in root.iter('site')}

    asset_dir = ROOT / 'franka_emika_panda' / 'assets'
    meshes = [mesh.get('file') for mesh in root.findall('./asset/mesh')
              if mesh.get('file')]
    missing = [f for f in meshes if not (asset_dir / f).is_file()]
    if missing:
        raise RuntimeError(f'Missing Panda mesh assets: {missing[:10]}')
    if any(f'joint{k}' not in joints for k in range(1, 8)):
        raise RuntimeError('One or more Panda arm joints are missing')
    if any(name not in bodies for name in BLOCK_NAMES) or 'Block' in bodies:
        raise RuntimeError('Lab 3 must contain eight named cubes, no legacy Block')
    if any(f'cell_{name}' not in sites for name in GRID_CELLS):
        raise RuntimeError('Missing one or more of the nine grid sites')
    if any(f'stage_{name}' not in sites for name in BLOCK_NAMES):
        raise RuntimeError('Missing one or more staging sites')
    if 'lab3_tool' not in sites:
        raise RuntimeError('Missing lab3_tool site')

    positions = {}
    for name in BLOCK_NAMES:
        body = bodies[name]
        if len(body.findall('freejoint')) != 1:
            raise RuntimeError(f'{name} must have exactly one free joint')
        if len(body.findall('geom')) != 1:
            raise RuntimeError(f'{name} must have one physical geom')
        positions[name] = [float(x) for x in body.get('pos').split()]

    return {
        'scene_path': str(Path(xml).resolve()),
        'seven_arm_joints': True,
        'finger_joints': sum(name in joints for name in ('finger_joint1', 'finger_joint2')),
        'block_count': len(BLOCK_NAMES),
        'red_count': sum(name.startswith('red_') for name in BLOCK_NAMES),
        'blue_count': sum(name.startswith('blue_') for name in BLOCK_NAMES),
        'free_block_joints': len(BLOCK_NAMES),
        'grid_sites': sum(f'cell_{name}' in sites for name in GRID_CELLS),
        'staging_sites': sum(f'stage_{name}' in sites for name in BLOCK_NAMES),
        'shelf_bodies_retained': sum('Shelf' in (name or '') for name in bodies),
        'lab3_tool_site': 'lab3_tool',
        'grid_pitch_m': GRID_PITCH,
        'block_width_m': 2 * BLOCK_HALF,
        'block_center_z_m': BLOCK_CENTER_Z,
        'mesh_assets_found': len(meshes),
        'mesh_assets_missing': len(missing),
        'initial_block_xyz_m': positions,
    }