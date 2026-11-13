"""Regression checks for close-spaced top grasps and recoverable goal layouts."""
import unittest

from lab3.access import (
    blocked_core, can_access, occupancy_mask, peel_order,
    top_access_options, top_access_options_xyz,
)
from lab3.config import GRID_CELLS, grid_position


class TopAccessTests(unittest.TestCase):
    def test_each_adjacent_side_uses_perpendicular_jaws(self):
        expected = {"G01": {0., 180.}, "G21": {0., 180.},
                    "G10": {90., 270.}, "G12": {90., 270.}}
        for neighbor, yaws in expected.items():
            for phase in ("pick", "place"):
                options = top_access_options("G11", ("G11", neighbor), phase=phase)
                self.assertEqual({option.yaw_deg for option in options}, yaws)

    def test_locked_square_and_recoverable_ring(self):
        square = ("G00", "G01", "G10", "G11", "G21")
        self.assertIsNone(peel_order(square))
        self.assertEqual(blocked_core(square), ("G00", "G01", "G10", "G11"))
        ring = tuple(cell for cell in GRID_CELLS if cell != "G11")
        self.assertFalse(can_access("G00", ring))
        self.assertTrue(can_access("G01", ring))
        self.assertIsNotNone(peel_order(ring))
        self.assertEqual(blocked_core(ring), ())
        self.assertFalse(can_access("G11", ring, phase="place"))

    def test_exhaustive_grid_masks_have_monotone_access_and_valid_certificates(self):
        # Every one of the 512 occupancy arrangements of the assignment grid.
        for mask in range(1 << len(GRID_CELLS)):
            occupied = [cell for i, cell in enumerate(GRID_CELLS) if mask & (1 << i)]
            for cell in occupied:
                if can_access(cell, mask):
                    for neighbor in occupied:
                        reduced = occupancy_mask(x for x in occupied if x != neighbor)
                        self.assertTrue(can_access(cell, reduced))
            order = peel_order(mask)
            if order is not None:
                remaining = mask
                for cell in order:
                    self.assertTrue(can_access(cell, remaining))
                    remaining &= ~(1 << GRID_CELLS.index(cell))
                self.assertEqual(remaining, 0)
            else:
                core = blocked_core(mask)
                self.assertTrue(core)
                self.assertTrue(all(not can_access(cell, core) for cell in core))

    def test_measured_pose_and_symbolic_checks_agree(self):
        occupied = ("G00", "G01", "G02", "G11", "G21")
        for cell in occupied:
            center = grid_position(int(cell[1]), int(cell[2]))
            obstacles = [grid_position(int(loc[1]), int(loc[2]))
                         for loc in occupied if loc != cell]
            self.assertEqual(top_access_options(cell, occupied),
                             top_access_options_xyz(center, obstacles))


if __name__ == "__main__":
    unittest.main()
