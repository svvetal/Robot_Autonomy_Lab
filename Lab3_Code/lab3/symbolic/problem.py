"""Convert world state / target patterns into symbolic init and goal sets."""
from .domain import static_facts
from ..layout import ALL_CELLS, PATTERNS


def init_from_occupancy(occupancy: dict) -> set:
    """occupancy: {cell: colour or None} for every cell in ALL_CELLS."""
    facts = static_facts()
    facts.add(('handempty', ()))
    for c in ALL_CELLS:
        col = occupancy.get(c)
        facts.add(('has', (c, col)) if col else ('empty', (c,)))
    return facts


def goal_for_pattern(name: str) -> set:
    goal = {('handempty', ())}
    for cell, col in PATTERNS[name].items():
        goal.add(('has', (cell, col)) if col else ('empty', (cell,)))
    return goal


def apply_plan(occupancy: dict, plan) -> dict:
    """Symbolically roll the occupancy forward (used to chain patterns)."""
    occ = dict(occupancy)
    for act in plan:
        cell, col, _ax = act.args
        if act.name == 'pick':
            assert occ.get(cell) == col, f'{act.label()} inconsistent with state'
            occ[cell] = None
        else:
            assert occ.get(cell) is None, f'{act.label()} into occupied cell'
            occ[cell] = col
    return occ
