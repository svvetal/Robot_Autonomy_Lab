import argparse
import json

def main():
    parser = argparse.ArgumentParser(description='Lab 3 simulator setup / checks')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--check-scene', action='store_true',
                       help='Validate generated XML and mesh assets (no MuJoCo needed)')
    group.add_argument('--check-simulator', action='store_true',
                       help='Load MuJoCo and check eight block poses / collision query')
    group.add_argument('--view-scene', action='store_true',
                       help='Open MuJoCo interactive viewer without moving the arm')
    args = parser.parse_args()

    from lab3.scene import prepare_scene, inspect_scene
    xml = prepare_scene()
    if args.check_scene or not (args.check_simulator or args.view_scene):
        print(json.dumps(inspect_scene(xml), indent=2))
        return

    try:
        from lab3.config import Settings
        from lab3.simulator import Lab3Sim
        sim = Lab3Sim(Settings(), control='kinematic')
    except ImportError as exc:
        parser.error(str(exc))

    if args.check_simulator:
        import numpy as np
        from lab3.config import BLOCK_NAMES
        positions = sim.block_positions()
        print('MuJoCo model loaded successfully')
        print('Blocks:', len(positions), ', '.join(BLOCK_NAMES))
        for name, xyz in positions.items():
            print(f'  {name:8s} : {np.round(xyz, 4).tolist()}')
        print('Arm collision check at HOME:')
        issues = sim.collision_reasons(sim.arm_q())
        print('  No unintended contacts' if not issues else '\n'.join('  ' + x for x in issues))
        print('Scene/model check completed; no grasp or task execution performed.')
    else:
        try:
            import mujoco.viewer
        except ImportError as exc:
            parser.error(f'MuJoCo viewer unavailable: {exc}')
        print('Opening eight-block scene. There is NO Lab 2 playback or task sequence.')
        mujoco.viewer.launch(sim.model, sim.data)

if __name__ == '__main__':
    main()