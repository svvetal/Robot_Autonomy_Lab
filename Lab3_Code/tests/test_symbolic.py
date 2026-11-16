"""Symbolic layer check: chains the four patterns from the Figure 1 setup.
Run from Lab3_Code:  python3 -m tests.test_symbolic   (takes ~2 min)"""
import time
from lab3.layout import INITIAL_BLOCKS, ALL_CELLS, PATTERN_ORDER, accepts
from lab3.symbolic.domain import make_planner
from lab3.symbolic.problem import init_from_occupancy, goal_for_pattern, apply_plan

EXPECTED_FEASIBLE = {'pickaxe': True, 'axe': False, 'box': True, 'bow': True}


def main():
    from lab3.determinism import ensure_deterministic_hashing
    ensure_deterministic_hashing()
    occ = {c: None for c in ALL_CELLS}
    for _, col, cell in INITIAL_BLOCKS:
        occ[cell] = col
    planner = make_planner()
    for name in PATTERN_ORDER:
        t = time.time()
        plan = planner.plan(init_from_occupancy(occ), goal_for_pattern(name))
        print(f'{name}: {"SKIP (no plan)" if plan is None else f"{len(plan)} actions"} '
              f'[{time.time() - t:.1f}s]')
        assert (plan is not None) == EXPECTED_FEASIBLE[name], name
        if plan is not None:
            for a in plan:
                print('   ', a.label())
                cell, colour, _ = a.args
                if a.name == 'place':
                    assert accepts(cell, colour), f'{a.label()} puts a cube in the wrong column'
            occ = apply_plan(occ, plan)
    print('OK')


if __name__ == '__main__':
    main()
