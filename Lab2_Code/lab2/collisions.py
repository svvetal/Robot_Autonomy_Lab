"""Contact handling for grasping a cube from a shelf."""
import numpy as np

SUPPORT_BODIES = frozenset({
    'LShelfBottom', 'RShelfBottom', 'RShelfMiddle', 'RShelfTop',
})

def supported_pickup_contact(*, shelf_body, shelf_center, shelf_half_size,
                             block_center, block_half_size, contact_distance,
                             tolerance=0.0015):
    """Recognize shallow cube contact with its supporting shelf."""
    if shelf_body not in SUPPORT_BODIES:
        return False
    c = np.asarray(shelf_center, float)
    s = np.asarray(shelf_half_size, float)
    p = np.asarray(block_center, float)
    h = np.asarray(block_half_size, float)
    if not (c.shape == (3,) and s.shape == (3,) and p.shape == (3,) and h.shape == (3,)):
        return False
    if not all(np.isfinite(v).all() for v in (c, s, p, h)):
        return False
    if np.any(s <= 0) or np.any(h <= 0):
        return False
    shelf_top = c[2] + s[2]
    cube_bottom = p[2] - h[2]
    return bool(
        -tolerance <= contact_distance <= tolerance
        and abs(cube_bottom - shelf_top) <= tolerance
        and abs(p[0]-c[0]) <= s[0]+h[0]+tolerance
        and abs(p[1]-c[1]) <= s[1]+h[1]+tolerance
    )
