"""Scene geometry that more than one backend has to agree on.

Deliberately free of simulator imports: the ManiSkill env, the Isaac rig job
builder and the protocol validator all read the same definition here, so a
vessel cannot be one shape on the replay gate and another on the rig.
"""
from __future__ import annotations

import math


def cylinder_container_parts(inner_diameter: float, height: float,
                             wall: float = 0.003, sections: int = 20):
    """An open-top ROUND vessel: a ring of `sections` wall slabs.

    A pot, a tin, a jar — the receptacle an episode drops something INTO.
    The square `container` prop is the wrong model for one, and not
    cosmetically: its diagonal is 41 % longer than its width, so a long object
    that a round vessel forces to stand up on end lies down across the corners
    instead. The final pose the episode is scored on is then a different pose.

    Returns ``[(offset_xyz, half_size_xyz, yaw_rad), ...]`` in the vessel's own
    frame, whose origin is the INNER FLOOR CENTRE — the same convention as the
    square `container` prop, so ``pos.z = support_z`` rests it on the surface.

    There is no floor slab. The surface the vessel stands on is its floor: a
    pot's own base is a millimetre of steel, and a square slab under a round
    pot would put corners outside the wall ring that objects can rest on.
    Callers that need the contents to sit higher should raise ``pos.z``.
    """
    r_in = float(inner_diameter) / 2.0
    h, t = float(height), float(wall)
    n = int(sections)
    if r_in <= 0 or h <= 0 or t <= 0:
        raise ValueError("cylinder_container needs positive inner_diameter, "
                         "height and wall")
    if n < 6:
        raise ValueError("cylinder_container needs at least 6 sections")
    # Tangential half-length so neighbouring slabs meet at the inner radius,
    # with a little overlap: a ring that does not close leaks contacts.
    half_t = r_in * math.tan(math.pi / n) * 1.08
    r_mid = r_in + t / 2.0
    parts = []
    for i in range(n):
        th = 2.0 * math.pi * i / n
        parts.append((
            (r_mid * math.cos(th), r_mid * math.sin(th), h / 2.0),
            (t / 2.0, half_t, h / 2.0),
            th,
        ))
    return parts


def polygon_container_parts(inner_across_flats: float, height: float,
                            wall: float = 0.004, sections: int = 6,
                            floor: float = 0.004):
    """An open-top REGULAR-POLYGON vessel, with a floor: a hexagonal pen cup.

    The round `cylinder_container` is static — it is the pot an episode drops
    something into, and the table is its floor. A vessel the robot PICKS UP is
    a different object: it needs its own bottom (there is no table under it
    once it is in the air) and it has to be ONE rigid body, so the walls travel
    with it. The square dynamic `container` is that, for a rectangular bin. A
    hexagonal desk tidy is neither: squaring it off moves the outer faces the
    pads close on by up to 15 % of the width and squares off the cavity the
    contents settle in.

    `inner_across_flats` is the width of the cavity between two opposite walls.
    `sections` must be even so that opposite walls exist to measure that
    across.

    Returns ``[(offset_xyz, half_size_xyz, yaw_rad), ...]`` in the vessel's own
    frame, whose origin is the INNER FLOOR CENTRE — the same convention as the
    other containers, so ``pos.z = support_z + floor`` rests it on a surface.

    The floor is `sections / 2` rectangles rather than one square: the corners
    of a square big enough to cover a hexagon stick out past the walls, and an
    object can then come to rest on a ledge outside the vessel it is supposed
    to be in. Each rectangle spans one pair of opposite edges — the apothem
    across, that edge's own length along — and its four corners land exactly on
    four of the polygon's vertices, so rotating it through the ``sections / 2``
    distinct edge directions tiles the polygon exactly.
    """
    a_in = float(inner_across_flats) / 2.0        # inner apothem
    h, t, f = float(height), float(wall), float(floor)
    n = int(sections)
    if a_in <= 0 or h <= 0 or t <= 0 or f <= 0:
        raise ValueError("polygon_container needs positive inner_across_flats, "
                         "height, wall and floor")
    if n < 4 or n % 2:
        raise ValueError(f"polygon_container needs an even sections >= 4, got {n}")
    r_in = a_in / math.cos(math.pi / n)           # inner circumradius
    side = 2.0 * r_in * math.sin(math.pi / n)     # inner edge length
    parts = []
    # floor: n/2 rectangles, one per pair of opposite edges
    for i in range(n // 2):
        th = 2.0 * math.pi * i / n
        parts.append(((0.0, 0.0, -f / 2.0), (a_in, side / 2.0, f / 2.0), th))
    # walls: one slab outside each edge, overlapping at the corners (a ring
    # that does not close leaks contacts)
    for i in range(n):
        th = 2.0 * math.pi * i / n
        r_mid = a_in + t / 2.0
        parts.append((
            (r_mid * math.cos(th), r_mid * math.sin(th), h / 2.0),
            (t / 2.0, side / 2.0 + t, h / 2.0),
            th,
        ))
    return parts


def arch_block_parts(length: float, depth: float, height: float,
                     arch_radius: float, leg_height: float = 0.0,
                     slabs: int = 24):
    """A toy BRIDGE block: a slab with an arch cut through its short axis.

    The classic wooden-block arch. Length runs along local +x, depth along
    +y, height along +z; the opening is cut all the way through the depth, so
    both length-by-height faces show the arch. `leg_height` is the straight
    rise under the semicircular head, so the opening is
    ``leg_height + arch_radius`` tall and ``2 * arch_radius`` wide.

    Modelled as vertical slabs because that is a decomposition into boxes that
    is exact except for the staircase on the arc, and every consumer here
    (SAPIEN collisions, the Isaac rig, the mesh exporter) wants convex parts.
    A single convex hull of this shape is a brick: the opening is what the
    fingers reach through and what makes a row of these a viaduct rather than
    a wall.

    Returns ``[(offset_xyz, half_size_xyz), ...]`` in the block's own frame,
    whose origin is the centre of the BOUNDING BOX (so ``pos.z = support_z +
    height / 2`` rests it on a surface).
    """
    L, D, H = float(length), float(depth), float(height)
    r, a = float(arch_radius), float(leg_height)
    n = int(slabs)
    if L <= 0 or D <= 0 or H <= 0:
        raise ValueError("arch_block needs positive length, depth and height")
    if r < 0 or a < 0:
        raise ValueError("arch_block needs a non-negative radius and leg height")
    if 2 * r > L:
        raise ValueError("arch_block opening is wider than the block")
    if a + r >= H:
        raise ValueError("arch_block opening is taller than the block")
    if n < 4:
        raise ValueError("arch_block needs at least 4 slabs")
    edges = [-L / 2 + L * i / n for i in range(n + 1)]
    parts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mid = 0.5 * (lo + hi)
        # the arc is sampled at the slab's OUTER edge (the one nearer the
        # crown) so the staircase never eats into the solid material: the
        # modelled block is a subset of the real one, never a superset.
        u = min(abs(lo), abs(hi))
        floor = a + math.sqrt(max(r * r - u * u, 0.0)) if u < r else 0.0
        if floor >= H:
            continue
        parts.append((
            (mid, 0.0, (floor + H) / 2.0 - H / 2.0),
            ((hi - lo) / 2.0, D / 2.0, (H - floor) / 2.0),
        ))
    return parts
