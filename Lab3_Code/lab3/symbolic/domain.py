"""STRIPS domain for the 3x3 block grid, solved with the HW3 A* planner.

Blocks of the same colour are interchangeable, so the symbolic state tracks
colours per cell rather than block identities. This removes the 4!*4! symmetry
and keeps A* well within the HW3 expansion budget.  The executor (world model)
maps "pick red from g01" back to the concrete MuJoCo body that sits there.

Predicates
  (has ?c ?col)        cell ?c holds a block of colour ?col
  (empty ?c)           cell ?c is free           ('out' is always empty)
  (accepts ?c ?col)    static: a ?col block may be put in ?c. Grid cells accept
                       both colours; the red storage column only red, the blue
                       column only blue (also between patterns). Excludes 'out'.
  (fn ?c ?axis ?a ?b)  static: closing along ?axis, the fingers sweep cells ?a, ?b
  (handempty) / (holding ?col)

Actions (the axis argument becomes the grasp yaw at execution time)
  pick(?c ?col ?ax)   needs both finger-neighbours on ?ax empty
  place(?c ?col ?ax)  same condition, so a block is never dropped somewhere it
                      can't be grasped (e.g. into the middle of a ring)
"""
from .SymPlanner import ActionSchema, Planner
from ..layout import ALL_CELLS, OUTSIDE, RED, BLUE, accepts, finger_neighbours

AXES = ('x', 'y')

PICK = ActionSchema(
    name='pick',
    parameters=('?c', '?col', '?ax'),
    preconds=(
        ('handempty', ()),
        ('has', ('?c', '?col')),
        ('fn', ('?c', '?ax', '?a', '?b')),
        ('empty', ('?a',)),
        ('empty', ('?b',)),
    ),
    add_effects=(('holding', ('?col',)), ('empty', ('?c',))),
    del_effects=(('handempty', ()), ('has', ('?c', '?col'))),
)

PLACE = ActionSchema(
    name='place',
    parameters=('?c', '?col', '?ax'),
    preconds=(
        ('holding', ('?col',)),
        ('accepts', ('?c', '?col')),
        ('empty', ('?c',)),
        ('fn', ('?c', '?ax', '?a', '?b')),
        ('empty', ('?a',)),
        ('empty', ('?b',)),
    ),
    add_effects=(('handempty', ()), ('has', ('?c', '?col'))),
    del_effects=(('holding', ('?col',)), ('empty', ('?c',))),
)


def make_domain():
    return [PICK, PLACE]


def static_facts():
    facts = {('empty', (OUTSIDE,))}
    for c in ALL_CELLS:
        for col in (RED, BLUE):
            if accepts(c, col):
                facts.add(('accepts', (c, col)))
        for ax, (a, b) in finger_neighbours(c).items():
            facts.add(('fn', (c, ax, a, b)))
    return facts


def make_planner():
    return Planner(make_domain())
