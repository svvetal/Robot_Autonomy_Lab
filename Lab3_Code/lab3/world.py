"""World model: which concrete cube sits in which symbolic cell."""
import numpy as np
from .layout import ALL_CELLS, INITIAL_BLOCKS, PATTERNS, cell_xyz


class WorldModel:
    def __init__(self, blocks=INITIAL_BLOCKS):
        self.colour = {name: col for name, col, _ in blocks}
        self.at = {c: None for c in ALL_CELLS}         # cell -> block name
        for name, _col, cell in blocks:
            self.at[cell] = name
        self.held = None

    def occupancy(self):
        """{cell: colour or None} - the symbolic planner's view."""
        return {c: (self.colour[b] if b else None) for c, b in self.at.items()}

    def pick(self, cell, colour):
        name = self.at[cell]
        if name is None or self.colour[name] != colour:
            raise RuntimeError(f'World model: no {colour} cube in {cell} (have {name})')
        if self.held is not None:
            raise RuntimeError(f'World model: already holding {self.held}')
        self.at[cell], self.held = None, name
        return name

    def place(self, cell):
        if self.at[cell] is not None:
            raise RuntimeError(f'World model: {cell} already holds {self.at[cell]}')
        name, self.held = self.held, None
        self.at[cell] = name
        return name

    def check_against_sim(self, sim, tol=0.012):
        """Return [(block, cell, error_m)] for cubes that drifted out of their cell."""
        bad = []
        for cell, name in self.at.items():
            if name is not None:
                err = float(np.linalg.norm(sim.block_pose(name)[:2, 3] - cell_xyz(cell)[:2]))
                if err > tol:
                    bad.append((name, cell, err))
        return bad

    def matches(self, pattern):
        occ = self.occupancy()
        return all(occ[c] == col for c, col in PATTERNS[pattern].items())

    def ascii(self):
        sym = {None: '.', 'red': 'R', 'blue': 'B'}
        occ = self.occupancy()
        return '\n'.join(''.join(sym[occ[f'g{r}{c}']] for c in range(3)) for r in range(3))
