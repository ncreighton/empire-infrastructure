"""
Generate 3 binary STL test models for the ForgeFiles pipeline.

Models:
  1. dragon_guardian.stl   — Dragon-like creature from combined primitives (~500-1000 tri)
  2. geometric_vase.stl    — Parametric vase via sine-wave revolution (~400-800 tri)
  3. articulated_gear.stl  — Gear/cog with mathematical teeth (~300-600 tri)

Uses only stdlib (struct, math). Writes proper binary STL format.
"""

import struct
import math
import os

# ---------------------------------------------------------------------------
# STL helpers
# ---------------------------------------------------------------------------

def cross(a, b):
    """Cross product of two 3-vectors."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def normalize(v):
    """Return unit vector (or zero vector if degenerate)."""
    mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if mag < 1e-12:
        return (0.0, 0.0, 0.0)
    return (v[0] / mag, v[1] / mag, v[2] / mag)


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def scale(v, s):
    return (v[0] * s, v[1] * s, v[2] * s)


def compute_normal(v1, v2, v3):
    """Compute face normal from three vertices (CCW winding)."""
    return normalize(cross(sub(v2, v1), sub(v3, v1)))


def write_stl(filepath, triangles):
    """
    Write binary STL.
    triangles: list of (v1, v2, v3) where each v is (x, y, z) floats.
    """
    with open(filepath, "wb") as f:
        # 80-byte header (zeros)
        f.write(b"\x00" * 80)
        # triangle count
        f.write(struct.pack("<I", len(triangles)))
        for v1, v2, v3 in triangles:
            n = compute_normal(v1, v2, v3)
            f.write(struct.pack("<3f", *n))
            f.write(struct.pack("<3f", *v1))
            f.write(struct.pack("<3f", *v2))
            f.write(struct.pack("<3f", *v3))
            f.write(struct.pack("<H", 0))  # attribute byte count


def read_stl_info(filepath):
    """Read back and report triangle count and file size."""
    size = os.path.getsize(filepath)
    with open(filepath, "rb") as f:
        f.read(80)  # header
        count = struct.unpack("<I", f.read(4))[0]
    expected = 80 + 4 + count * 50
    return count, size, expected


# ---------------------------------------------------------------------------
# Primitive mesh generators
# ---------------------------------------------------------------------------

def sphere_triangles(cx, cy, cz, r, n_lat=12, n_lon=16):
    """Generate triangles for a UV sphere."""
    tris = []

    def pt(lat_i, lon_j):
        theta = math.pi * lat_i / n_lat
        phi = 2 * math.pi * lon_j / n_lon
        x = cx + r * math.sin(theta) * math.cos(phi)
        y = cy + r * math.sin(theta) * math.sin(phi)
        z = cz + r * math.cos(theta)
        return (x, y, z)

    for i in range(n_lat):
        for j in range(n_lon):
            p00 = pt(i, j)
            p10 = pt(i + 1, j)
            p01 = pt(i, j + 1)
            p11 = pt(i + 1, j + 1)
            if i > 0:
                tris.append((p00, p10, p01))
            if i < n_lat - 1:
                tris.append((p01, p10, p11))
    return tris


def cone_triangles(base_center, tip, radius, n_seg=16):
    """Cone from base_center to tip with given radius at the base."""
    tris = []
    bx, by, bz = base_center
    # Build a local coordinate frame perpendicular to the axis
    axis = sub(tip, base_center)
    axis_n = normalize(axis)
    # Find a perpendicular vector
    if abs(axis_n[0]) < 0.9:
        perp = normalize(cross(axis_n, (1, 0, 0)))
    else:
        perp = normalize(cross(axis_n, (0, 1, 0)))
    perp2 = normalize(cross(axis_n, perp))

    def base_pt(j):
        angle = 2 * math.pi * j / n_seg
        dx = radius * (math.cos(angle) * perp[0] + math.sin(angle) * perp2[0])
        dy = radius * (math.cos(angle) * perp[1] + math.sin(angle) * perp2[1])
        dz = radius * (math.cos(angle) * perp[2] + math.sin(angle) * perp2[2])
        return (bx + dx, by + dy, bz + dz)

    for j in range(n_seg):
        p0 = base_pt(j)
        p1 = base_pt(j + 1)
        # Side face
        tris.append((p0, p1, tip))
        # Base face
        tris.append((p1, p0, base_center))
    return tris


def cylinder_triangles(bottom_center, top_center, radius, n_seg=12):
    """Cylinder between two points."""
    tris = []
    bx, by, bz = bottom_center
    tx, ty, tz = top_center
    axis = sub(top_center, bottom_center)
    axis_n = normalize(axis)
    if abs(axis_n[0]) < 0.9:
        perp = normalize(cross(axis_n, (1, 0, 0)))
    else:
        perp = normalize(cross(axis_n, (0, 1, 0)))
    perp2 = normalize(cross(axis_n, perp))

    def ring_pt(center, j):
        cx, cy, cz = center
        angle = 2 * math.pi * j / n_seg
        dx = radius * (math.cos(angle) * perp[0] + math.sin(angle) * perp2[0])
        dy = radius * (math.cos(angle) * perp[1] + math.sin(angle) * perp2[1])
        dz = radius * (math.cos(angle) * perp[2] + math.sin(angle) * perp2[2])
        return (cx + dx, cy + dy, cz + dz)

    for j in range(n_seg):
        b0 = ring_pt(bottom_center, j)
        b1 = ring_pt(bottom_center, j + 1)
        t0 = ring_pt(top_center, j)
        t1 = ring_pt(top_center, j + 1)
        # Side quads (2 tris each)
        tris.append((b0, b1, t0))
        tris.append((t0, b1, t1))
        # Caps
        tris.append((b1, b0, bottom_center))
        tris.append((t0, t1, top_center))
    return tris


def flat_triangle(p1, p2, p3):
    """Single triangle."""
    return [(p1, p2, p3)]


def quad_triangles(p1, p2, p3, p4):
    """Quad as two triangles."""
    return [(p1, p2, p3), (p1, p3, p4)]


# ---------------------------------------------------------------------------
# Model 1: Dragon Guardian (~500-1000 triangles, ~80mm tall)
# ---------------------------------------------------------------------------

def generate_dragon():
    """
    Dragon assembled from primitive shapes:
    - Spherical body (torso)
    - Smaller sphere for head
    - Cone for snout
    - 4 cylindrical legs
    - 2 triangular wings (flat quads)
    - Cone for tail
    - Small sphere for tail tip
    - Cone horns on head
    """
    tris = []

    # --- Body (large sphere, centered at origin, radius 15mm) ---
    # Slightly elongated by using an ellipsoid via sphere + scaling trick:
    # We generate at origin and the body sits at z=40 (center)
    body_cx, body_cy, body_cz = 0, 0, 40
    body_r = 15
    tris.extend(sphere_triangles(body_cx, body_cy, body_cz, body_r, n_lat=14, n_lon=18))

    # --- Belly (slightly smaller sphere overlapping lower body) ---
    tris.extend(sphere_triangles(0, 2, 35, 10, n_lat=10, n_lon=14))

    # --- Neck (cylinder from body top to head position) ---
    neck_bottom = (0, -5, 55)
    neck_top = (0, -10, 68)
    tris.extend(cylinder_triangles(neck_bottom, neck_top, 5, n_seg=12))

    # --- Head (sphere) ---
    head_cx, head_cy, head_cz = 0, -12, 72
    tris.extend(sphere_triangles(head_cx, head_cy, head_cz, 7, n_lat=12, n_lon=14))

    # --- Snout (cone extending forward from head) ---
    snout_base = (0, -18, 72)
    snout_tip = (0, -28, 70)
    tris.extend(cone_triangles(snout_base, snout_tip, 4, n_seg=12))

    # --- Horns (two small cones on top of head) ---
    tris.extend(cone_triangles((3, -10, 78), (4, -8, 86), 1.5, n_seg=8))
    tris.extend(cone_triangles((-3, -10, 78), (-4, -8, 86), 1.5, n_seg=8))

    # --- Eyes (two tiny spheres) ---
    tris.extend(sphere_triangles(3.5, -17, 74, 1.2, n_lat=6, n_lon=8))
    tris.extend(sphere_triangles(-3.5, -17, 74, 1.2, n_lat=6, n_lon=8))

    # --- Front legs (cylinders) ---
    # Front-right
    tris.extend(cylinder_triangles((8, -3, 30), (12, -3, 5), 3, n_seg=10))
    # Front-left
    tris.extend(cylinder_triangles((-8, -3, 30), (-12, -3, 5), 3, n_seg=10))
    # Feet (small spheres)
    tris.extend(sphere_triangles(12, -3, 5, 3.5, n_lat=8, n_lon=10))
    tris.extend(sphere_triangles(-12, -3, 5, 3.5, n_lat=8, n_lon=10))

    # --- Back legs (cylinders, slightly thicker) ---
    # Back-right
    tris.extend(cylinder_triangles((9, 5, 28), (14, 8, 5), 3.5, n_seg=10))
    # Back-left
    tris.extend(cylinder_triangles((-9, 5, 28), (-14, 8, 5), 3.5, n_seg=10))
    # Feet
    tris.extend(sphere_triangles(14, 8, 5, 4, n_lat=8, n_lon=10))
    tris.extend(sphere_triangles(-14, 8, 5, 4, n_lat=8, n_lon=10))

    # --- Tail (cone from back of body, curving down) ---
    tail_base = (0, 10, 35)
    tail_mid = (0, 25, 25)
    tail_tip = (0, 35, 20)
    tris.extend(cylinder_triangles(tail_base, tail_mid, 5, n_seg=10))
    tris.extend(cone_triangles(tail_mid, tail_tip, 4, n_seg=10))
    # Tail spike
    tris.extend(cone_triangles(tail_tip, (0, 40, 22), 2, n_seg=8))

    # --- Wings (flat triangular surfaces, multiple quads for each wing) ---
    # Right wing
    wing_root_r = [(10, -2, 50), (12, 0, 45)]
    wing_tip_r = [(35, -5, 65), (30, -3, 55)]
    wing_far_r = [(28, -8, 75)]

    # Build wing as fan of triangles from shoulder
    shoulder_r = (10, -2, 50)
    wing_pts_r = [
        (15, -4, 55),
        (22, -6, 62),
        (30, -8, 70),
        (35, -7, 65),
        (32, -5, 58),
        (25, -3, 50),
        (18, -1, 45),
        (12, 0, 43),
    ]
    for k in range(len(wing_pts_r) - 1):
        tris.append((shoulder_r, wing_pts_r[k], wing_pts_r[k + 1]))
        # Back face
        tris.append((shoulder_r, wing_pts_r[k + 1],
                      (wing_pts_r[k][0], wing_pts_r[k][1] + 0.5, wing_pts_r[k][2])))

    # Left wing (mirror)
    shoulder_l = (-10, -2, 50)
    wing_pts_l = [(-p[0], p[1], p[2]) for p in wing_pts_r]
    for k in range(len(wing_pts_l) - 1):
        tris.append((shoulder_l, wing_pts_l[k + 1], wing_pts_l[k]))
        tris.append((shoulder_l,
                      (-wing_pts_l[k][0] * -1, wing_pts_l[k][1] + 0.5, wing_pts_l[k][2]),
                      wing_pts_l[k + 1]))

    # --- Chest ridge (small cones down the front) ---
    for i in range(4):
        z = 52 - i * 5
        tris.extend(cone_triangles((0, -8 - i, z), (0, -10 - i, z + 3), 1.0, n_seg=6))

    return tris


# ---------------------------------------------------------------------------
# Model 2: Geometric Vase (~400-800 triangles, 120mm tall, 60mm wide)
# ---------------------------------------------------------------------------

def generate_vase():
    """
    Revolution surface: rotate a sine-wave profile around the Z axis.
    Profile: r(z) = 20 + 10*sin(z * 2*pi / height * 2.5) with pinch at bottom.
    """
    tris = []
    height = 120.0
    n_z = 40  # vertical slices
    n_theta = 24  # angular slices
    dz = height / n_z

    def profile_radius(z):
        """Compute radius at height z."""
        # Pinch at very bottom
        t = z / height  # 0..1
        # Base pinch
        base_pinch = min(1.0, t * 5)  # ramps from 0 to 1 in bottom 20%
        # Sine wave body
        wave = 20 + 10 * math.sin(t * 2.5 * 2 * math.pi)
        # Top flare
        top_flare = 1.0 + 0.3 * max(0, t - 0.85) / 0.15
        # Overall
        r = wave * base_pinch * top_flare
        return max(r, 1.0)  # minimum radius to avoid degenerate faces

    def surface_pt(zi, tj):
        z = zi * dz
        theta = 2 * math.pi * tj / n_theta
        r = profile_radius(z)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        return (x, y, z)

    # Build the surface
    for i in range(n_z):
        for j in range(n_theta):
            p00 = surface_pt(i, j)
            p10 = surface_pt(i + 1, j)
            p01 = surface_pt(i, j + 1)
            p11 = surface_pt(i + 1, j + 1)
            tris.append((p00, p10, p01))
            tris.append((p01, p10, p11))

    # Bottom cap (close the base)
    for j in range(n_theta):
        p0 = surface_pt(0, j)
        p1 = surface_pt(0, j + 1)
        center = (0, 0, 0)
        tris.append((center, p1, p0))

    # Top rim — add a slight lip by adding a ring of inward-facing triangles
    rim_inset = 0.7  # how much the inner rim is smaller
    for j in range(n_theta):
        p_outer0 = surface_pt(n_z, j)
        p_outer1 = surface_pt(n_z, j + 1)
        z_top = height
        theta0 = 2 * math.pi * j / n_theta
        theta1 = 2 * math.pi * (j + 1) / n_theta
        r_inner = profile_radius(z_top) * rim_inset
        p_inner0 = (r_inner * math.cos(theta0), r_inner * math.sin(theta0), z_top + 2)
        p_inner1 = (r_inner * math.cos(theta1), r_inner * math.sin(theta1), z_top + 2)
        # Rim strip
        tris.append((p_outer0, p_outer1, p_inner0))
        tris.append((p_inner0, p_outer1, p_inner1))

    # Inner top cap
    top_center = (0, 0, height + 2)
    for j in range(n_theta):
        theta0 = 2 * math.pi * j / n_theta
        theta1 = 2 * math.pi * (j + 1) / n_theta
        r_inner = profile_radius(height) * rim_inset
        p0 = (r_inner * math.cos(theta0), r_inner * math.sin(theta0), height + 2)
        p1 = (r_inner * math.cos(theta1), r_inner * math.sin(theta1), height + 2)
        tris.append((top_center, p0, p1))

    return tris


# ---------------------------------------------------------------------------
# Model 3: Articulated Gear (~300-600 triangles, 40mm diameter)
# ---------------------------------------------------------------------------

def generate_gear():
    """
    Gear/cog with involute-style teeth generated mathematically.
    Flat gear (like a spur gear cross-section) with thickness.
    """
    tris = []

    n_teeth = 16
    outer_r = 20.0  # 40mm diameter
    inner_r = 14.0  # root radius
    hub_r = 6.0  # central hub hole (solid hub)
    thickness = 6.0  # gear thickness in Z

    tooth_half_angle = math.pi / n_teeth * 0.4  # tooth takes ~40% of the pitch

    def gear_profile():
        """Return list of (x,y) points forming the gear outline."""
        pts = []
        for i in range(n_teeth):
            base_angle = 2 * math.pi * i / n_teeth

            # Leading edge of tooth gap (on inner radius)
            a0 = base_angle - math.pi / n_teeth
            pts.append((inner_r * math.cos(a0), inner_r * math.sin(a0)))

            # Rise to outer (tooth leading edge)
            a1 = base_angle - tooth_half_angle
            pts.append((outer_r * math.cos(a1), outer_r * math.sin(a1)))

            # Tooth tip leading
            a2 = base_angle - tooth_half_angle * 0.6
            pts.append(((outer_r + 1.5) * math.cos(a2), (outer_r + 1.5) * math.sin(a2)))

            # Tooth tip trailing
            a3 = base_angle + tooth_half_angle * 0.6
            pts.append(((outer_r + 1.5) * math.cos(a3), (outer_r + 1.5) * math.sin(a3)))

            # Tooth trailing edge
            a4 = base_angle + tooth_half_angle
            pts.append((outer_r * math.cos(a4), outer_r * math.sin(a4)))

            # Trailing edge of tooth (back to inner)
            a5 = base_angle + math.pi / n_teeth
            pts.append((inner_r * math.cos(a5), inner_r * math.sin(a5)))

        return pts

    profile = gear_profile()
    n_pts = len(profile)

    z_bot = 0.0
    z_top = thickness

    # --- Top and bottom faces (triangulated as fan from center) ---
    center_top = (0, 0, z_top)
    center_bot = (0, 0, z_bot)

    for i in range(n_pts):
        j = (i + 1) % n_pts
        p0 = profile[i]
        p1 = profile[j]

        # Top face
        tris.append((center_top, (p0[0], p0[1], z_top), (p1[0], p1[1], z_top)))
        # Bottom face (reversed winding)
        tris.append((center_bot, (p1[0], p1[1], z_bot), (p0[0], p0[1], z_bot)))

    # --- Side walls (connect top and bottom outlines) ---
    for i in range(n_pts):
        j = (i + 1) % n_pts
        p0b = (profile[i][0], profile[i][1], z_bot)
        p1b = (profile[j][0], profile[j][1], z_bot)
        p0t = (profile[i][0], profile[i][1], z_top)
        p1t = (profile[j][0], profile[j][1], z_top)
        tris.append((p0b, p1b, p0t))
        tris.append((p0t, p1b, p1t))

    # --- Hub (central raised cylinder) ---
    hub_height = thickness + 3
    n_hub = 20
    tris.extend(cylinder_triangles((0, 0, z_bot), (0, 0, hub_height), hub_r, n_seg=n_hub))

    # --- Spoke cutouts (decorative: add small raised ridges as spokes) ---
    n_spokes = 4
    for s in range(n_spokes):
        angle = 2 * math.pi * s / n_spokes
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        # Each spoke is a thin box from hub to inner_r
        spoke_w = 1.5  # half-width
        spoke_h = thickness + 1  # slightly raised

        # Perpendicular direction
        px, py = -sin_a, cos_a

        # Four corners at hub_r
        r0 = hub_r + 1
        r1 = inner_r - 1
        p_inner = [
            (r0 * cos_a + spoke_w * px, r0 * sin_a + spoke_w * py, z_bot),
            (r0 * cos_a - spoke_w * px, r0 * sin_a - spoke_w * py, z_bot),
            (r0 * cos_a - spoke_w * px, r0 * sin_a - spoke_w * py, spoke_h),
            (r0 * cos_a + spoke_w * px, r0 * sin_a + spoke_w * py, spoke_h),
        ]
        p_outer = [
            (r1 * cos_a + spoke_w * px, r1 * sin_a + spoke_w * py, z_bot),
            (r1 * cos_a - spoke_w * px, r1 * sin_a - spoke_w * py, z_bot),
            (r1 * cos_a - spoke_w * px, r1 * sin_a - spoke_w * py, spoke_h),
            (r1 * cos_a + spoke_w * px, r1 * sin_a + spoke_w * py, spoke_h),
        ]

        # 6 faces of the spoke box
        # Top
        tris.extend(quad_triangles(p_inner[3], p_inner[2], p_outer[2], p_outer[3]))
        # Bottom
        tris.extend(quad_triangles(p_inner[0], p_outer[0], p_outer[1], p_inner[1]))
        # Front
        tris.extend(quad_triangles(p_inner[0], p_inner[3], p_outer[3], p_outer[0]))
        # Back
        tris.extend(quad_triangles(p_inner[1], p_outer[1], p_outer[2], p_inner[2]))
        # Inner end
        tris.extend(quad_triangles(p_inner[0], p_inner[1], p_inner[2], p_inner[3]))
        # Outer end
        tris.extend(quad_triangles(p_outer[0], p_outer[3], p_outer[2], p_outer[1]))

    # --- Chamfer ring on top face (decorative ring at inner_r) ---
    chamfer_r_inner = inner_r - 2
    chamfer_r_outer = inner_r
    chamfer_z = z_top + 0.8
    n_chamfer = 24
    for i in range(n_chamfer):
        a0 = 2 * math.pi * i / n_chamfer
        a1 = 2 * math.pi * (i + 1) / n_chamfer
        pi0 = (chamfer_r_inner * math.cos(a0), chamfer_r_inner * math.sin(a0), chamfer_z)
        pi1 = (chamfer_r_inner * math.cos(a1), chamfer_r_inner * math.sin(a1), chamfer_z)
        po0 = (chamfer_r_outer * math.cos(a0), chamfer_r_outer * math.sin(a0), z_top)
        po1 = (chamfer_r_outer * math.cos(a1), chamfer_r_outer * math.sin(a1), z_top)
        tris.append((pi0, pi1, po0))
        tris.append((po0, pi1, po1))

    return tris


# ---------------------------------------------------------------------------
# Model 4: Phone Stand (~200-400 triangles, ~80mm tall, L-shaped)
# ---------------------------------------------------------------------------

def generate_phone_stand():
    """
    L-shaped bracket with a slot for holding a phone.
    Functional print, asymmetric, thin walls.
    """
    tris = []

    # Base plate: 60mm wide x 40mm deep x 5mm thick
    base_w, base_d, base_h = 60, 40, 5

    def box_tris(cx, cy, cz, w, d, h):
        """Generate triangles for an axis-aligned box centered at (cx, cy, cz)."""
        hw, hd, hh = w / 2, d / 2, h / 2
        corners = [
            (cx - hw, cy - hd, cz - hh),  # 0: bottom-left-front
            (cx + hw, cy - hd, cz - hh),  # 1: bottom-right-front
            (cx + hw, cy + hd, cz - hh),  # 2: bottom-right-back
            (cx - hw, cy + hd, cz - hh),  # 3: bottom-left-back
            (cx - hw, cy - hd, cz + hh),  # 4: top-left-front
            (cx + hw, cy - hd, cz + hh),  # 5: top-right-front
            (cx + hw, cy + hd, cz + hh),  # 6: top-right-back
            (cx - hw, cy + hd, cz + hh),  # 7: top-left-back
        ]
        c = corners
        box = []
        # Bottom
        box.extend(quad_triangles(c[0], c[3], c[2], c[1]))
        # Top
        box.extend(quad_triangles(c[4], c[5], c[6], c[7]))
        # Front
        box.extend(quad_triangles(c[0], c[1], c[5], c[4]))
        # Back
        box.extend(quad_triangles(c[2], c[3], c[7], c[6]))
        # Left
        box.extend(quad_triangles(c[0], c[4], c[7], c[3]))
        # Right
        box.extend(quad_triangles(c[1], c[2], c[6], c[5]))
        return box

    # Base plate (sits on ground, centered at x=0)
    tris.extend(box_tris(0, 0, base_h / 2, base_w, base_d, base_h))

    # Back wall (vertical support, 60mm wide x 5mm thick x 75mm tall)
    wall_h = 75
    wall_t = 5
    tris.extend(box_tris(0, base_d / 2 - wall_t / 2, base_h + wall_h / 2,
                         base_w, wall_t, wall_h))

    # Phone lip (small ridge at front of base to hold phone bottom)
    lip_h = 10
    lip_t = 4
    lip_w = base_w
    tris.extend(box_tris(0, -base_d / 2 + lip_t / 2, base_h + lip_h / 2,
                         lip_w, lip_t, lip_h))

    # Support ribs (two triangular buttresses connecting wall to base)
    rib_t = 3  # thickness of rib
    rib_h = 35  # height up the wall
    rib_d = 20  # depth along base

    for sign in [-1, 1]:
        x_center = sign * (base_w / 2 - 8)
        # Right triangle: base on bottom plate, hypotenuse to wall
        # Build as a triangular prism (6 faces)
        p0 = (x_center - rib_t / 2, base_d / 2 - wall_t, base_h)  # bottom-back
        p1 = (x_center - rib_t / 2, base_d / 2 - wall_t - rib_d, base_h)  # bottom-front
        p2 = (x_center - rib_t / 2, base_d / 2 - wall_t, base_h + rib_h)  # top-back
        p3 = (x_center + rib_t / 2, base_d / 2 - wall_t, base_h)
        p4 = (x_center + rib_t / 2, base_d / 2 - wall_t - rib_d, base_h)
        p5 = (x_center + rib_t / 2, base_d / 2 - wall_t, base_h + rib_h)
        # Two triangular faces
        tris.append((p0, p1, p2))
        tris.append((p3, p5, p4))
        # Three quad faces
        tris.extend(quad_triangles(p0, p3, p4, p1))  # bottom
        tris.extend(quad_triangles(p0, p2, p5, p3))  # back
        tris.extend(quad_triangles(p1, p4, p5, p2))  # hypotenuse

    return tris


# ---------------------------------------------------------------------------
# Model 5: Mini Planter (~400-600 triangles, cylinder with drainage holes)
# ---------------------------------------------------------------------------

def generate_mini_planter():
    """
    Cylindrical planter with drainage holes in the bottom.
    Hollow interior, lip at top.
    """
    tris = []
    outer_r = 25.0
    inner_r = 22.0
    height = 50.0
    lip_height = 4.0
    lip_r = 27.0
    n_seg = 20
    base_thickness = 4.0

    def ring(center_z, radius, segments):
        pts = []
        for j in range(segments):
            angle = 2 * math.pi * j / segments
            pts.append((radius * math.cos(angle), radius * math.sin(angle), center_z))
        return pts

    # Outer wall
    bottom_outer = ring(0, outer_r, n_seg)
    top_outer = ring(height, outer_r, n_seg)
    for j in range(n_seg):
        j1 = (j + 1) % n_seg
        tris.append((bottom_outer[j], bottom_outer[j1], top_outer[j]))
        tris.append((top_outer[j], bottom_outer[j1], top_outer[j1]))

    # Inner wall (reversed normals for inside)
    bottom_inner = ring(base_thickness, inner_r, n_seg)
    top_inner = ring(height, inner_r, n_seg)
    for j in range(n_seg):
        j1 = (j + 1) % n_seg
        tris.append((bottom_inner[j], top_inner[j], bottom_inner[j1]))
        tris.append((top_inner[j], top_inner[j1], bottom_inner[j1]))

    # Top lip ring (outer_r -> lip_r, then lip_r down to inner connection)
    top_lip = ring(height, lip_r, n_seg)
    top_lip_inner = ring(height + lip_height, lip_r, n_seg)
    # Lip top surface: outer to lip
    for j in range(n_seg):
        j1 = (j + 1) % n_seg
        tris.append((top_outer[j], top_lip[j], top_outer[j1]))
        tris.append((top_outer[j1], top_lip[j], top_lip[j1]))
    # Lip outer wall
    for j in range(n_seg):
        j1 = (j + 1) % n_seg
        tris.append((top_lip[j], top_lip_inner[j], top_lip[j1]))
        tris.append((top_lip[j1], top_lip_inner[j], top_lip_inner[j1]))
    # Lip top face (ring from lip_r to inner_r)
    top_inner_lip = ring(height + lip_height, inner_r, n_seg)
    for j in range(n_seg):
        j1 = (j + 1) % n_seg
        tris.append((top_lip_inner[j], top_inner_lip[j], top_lip_inner[j1]))
        tris.append((top_lip_inner[j1], top_inner_lip[j], top_inner_lip[j1]))
    # Inner lip wall back down
    for j in range(n_seg):
        j1 = (j + 1) % n_seg
        tris.append((top_inner_lip[j], top_inner[j], top_inner_lip[j1]))
        tris.append((top_inner_lip[j1], top_inner[j], top_inner[j1]))

    # Bottom plate (annular ring: outer_r to inner_r at z=0 and z=base_thickness)
    bottom_plate_outer = ring(0, outer_r, n_seg)
    bottom_plate_inner = ring(0, inner_r * 0.3, n_seg)  # small center hole
    # Bottom face (outer ring)
    for j in range(n_seg):
        j1 = (j + 1) % n_seg
        tris.append((bottom_plate_outer[j1], bottom_plate_outer[j], bottom_plate_inner[j]))
        tris.append((bottom_plate_outer[j1], bottom_plate_inner[j], bottom_plate_inner[j1]))
    # Inner floor
    floor_inner = ring(base_thickness, inner_r, n_seg)
    floor_center = ring(base_thickness, inner_r * 0.3, n_seg)
    for j in range(n_seg):
        j1 = (j + 1) % n_seg
        tris.append((floor_inner[j], floor_inner[j1], floor_center[j]))
        tris.append((floor_center[j], floor_inner[j1], floor_center[j1]))

    # Drainage hole walls (connect bottom_plate_inner to floor_center)
    for j in range(n_seg):
        j1 = (j + 1) % n_seg
        tris.append((bottom_plate_inner[j], floor_center[j], bottom_plate_inner[j1]))
        tris.append((bottom_plate_inner[j1], floor_center[j], floor_center[j1]))

    return tris


# ---------------------------------------------------------------------------
# Model 6: Chess Piece — Queen (~400-700 triangles)
# ---------------------------------------------------------------------------

def generate_chess_piece():
    """
    Chess queen: circular base/pedestal, tapered body, crowned top.
    Tests base/pedestal detection (actual pedestal).
    """
    tris = []
    n_seg = 20

    # Build as stacked cylinder segments (lathe profile)
    # Profile: (radius, z_bottom, z_top)
    profile = [
        # Wide base
        (16.0, 0.0, 3.0),
        (14.0, 3.0, 5.0),
        # Pedestal taper
        (8.0, 5.0, 12.0),
        # Body bulge
        (10.0, 12.0, 20.0),
        (11.0, 20.0, 35.0),
        (10.0, 35.0, 45.0),
        # Neck taper
        (6.0, 45.0, 55.0),
        # Crown base
        (9.0, 55.0, 58.0),
        # Crown taper
        (7.0, 58.0, 65.0),
        # Tip
        (3.0, 65.0, 70.0),
    ]

    def ring_at(z, r):
        return [(r * math.cos(2 * math.pi * j / n_seg),
                 r * math.sin(2 * math.pi * j / n_seg), z) for j in range(n_seg)]

    # Bottom cap
    center_bottom = (0, 0, 0)
    bottom_ring = ring_at(0, profile[0][0])
    for j in range(n_seg):
        j1 = (j + 1) % n_seg
        tris.append((center_bottom, bottom_ring[j1], bottom_ring[j]))

    # Stacked segments
    prev_ring = bottom_ring
    for r, z_bot, z_top in profile:
        top_ring = ring_at(z_top, r)
        for j in range(n_seg):
            j1 = (j + 1) % n_seg
            tris.append((prev_ring[j], prev_ring[j1], top_ring[j]))
            tris.append((top_ring[j], prev_ring[j1], top_ring[j1]))
        prev_ring = top_ring

    # Connect segments with transition rings where radius changes
    for i in range(len(profile) - 1):
        r1 = profile[i][0]
        r2 = profile[i + 1][0]
        z = profile[i][2]  # transition height
        if abs(r1 - r2) > 0.5:
            ring1 = ring_at(z, r1)
            ring2 = ring_at(z, r2)
            for j in range(n_seg):
                j1 = (j + 1) % n_seg
                tris.append((ring1[j], ring1[j1], ring2[j]))
                tris.append((ring2[j], ring1[j1], ring2[j1]))

    # Top cap
    top_r = profile[-1][0]
    top_z = profile[-1][2]
    top_ring = ring_at(top_z, top_r)
    center_top = (0, 0, top_z)
    for j in range(n_seg):
        j1 = (j + 1) % n_seg
        tris.append((center_top, top_ring[j], top_ring[j1]))

    # Crown sphere on top
    tris.extend(sphere_triangles(0, 0, top_z + 3, 4, n_lat=8, n_lon=12))

    # Crown spikes (4 small cones around the crown base)
    for k in range(4):
        angle = 2 * math.pi * k / 4 + math.pi / 4
        spike_x = 7 * math.cos(angle)
        spike_y = 7 * math.sin(angle)
        spike_base = (spike_x, spike_y, 56)
        spike_tip = (spike_x * 1.2, spike_y * 1.2, 63)
        tris.extend(cone_triangles(spike_base, spike_tip, 1.5, n_seg=6))

    return tris


# ---------------------------------------------------------------------------
# Model 7: Gear Tower (~500-800 triangles, stacked interlocking gears)
# ---------------------------------------------------------------------------

def generate_gear_tower():
    """
    Three stacked gears of different sizes on a central shaft.
    Articulated/mechanical look, complex silhouette.
    """
    tris = []

    def gear_disc(center_z, outer_r, inner_r, thickness, n_teeth, hub_r=None):
        """Generate a gear disc at a given Z height."""
        disc_tris = []
        tooth_half_angle = math.pi / n_teeth * 0.4
        z_bot = center_z
        z_top = center_z + thickness

        def gear_profile():
            pts = []
            for i in range(n_teeth):
                base_angle = 2 * math.pi * i / n_teeth
                a0 = base_angle - math.pi / n_teeth
                pts.append((inner_r * math.cos(a0), inner_r * math.sin(a0)))
                a1 = base_angle - tooth_half_angle
                pts.append((outer_r * math.cos(a1), outer_r * math.sin(a1)))
                a2 = base_angle + tooth_half_angle
                pts.append((outer_r * math.cos(a2), outer_r * math.sin(a2)))
                a3 = base_angle + math.pi / n_teeth
                pts.append((inner_r * math.cos(a3), inner_r * math.sin(a3)))
            return pts

        profile = gear_profile()
        n_pts = len(profile)
        center_top = (0, 0, z_top)
        center_bot = (0, 0, z_bot)

        for i in range(n_pts):
            j = (i + 1) % n_pts
            p0, p1 = profile[i], profile[j]
            # Top face
            disc_tris.append((center_top, (p0[0], p0[1], z_top), (p1[0], p1[1], z_top)))
            # Bottom face
            disc_tris.append((center_bot, (p1[0], p1[1], z_bot), (p0[0], p0[1], z_bot)))

        # Side walls
        for i in range(n_pts):
            j = (i + 1) % n_pts
            p0b = (profile[i][0], profile[i][1], z_bot)
            p1b = (profile[j][0], profile[j][1], z_bot)
            p0t = (profile[i][0], profile[i][1], z_top)
            p1t = (profile[j][0], profile[j][1], z_top)
            disc_tris.append((p0b, p1b, p0t))
            disc_tris.append((p0t, p1b, p1t))

        return disc_tris

    # Central shaft (cylinder)
    shaft_r = 3.0
    shaft_h = 55.0
    tris.extend(cylinder_triangles((0, 0, 0), (0, 0, shaft_h), shaft_r, n_seg=12))

    # Bottom gear: large, 20 teeth
    tris.extend(gear_disc(2, outer_r=22, inner_r=16, thickness=6, n_teeth=20))

    # Middle gear: medium, 14 teeth, offset
    tris.extend(gear_disc(18, outer_r=16, inner_r=11, thickness=5, n_teeth=14))

    # Top gear: small, 10 teeth
    tris.extend(gear_disc(34, outer_r=12, inner_r=8, thickness=4, n_teeth=10))

    # Top cap (small sphere)
    tris.extend(sphere_triangles(0, 0, shaft_h, 4, n_lat=8, n_lon=10))

    return tris


# ---------------------------------------------------------------------------
# Model 8: Organic Blob (~400-600 triangles, Perlin-noise-like deformed sphere)
# ---------------------------------------------------------------------------

def generate_organic_blob():
    """
    Sphere deformed by pseudo-random noise.
    Organic shape, no flat faces, tests centroid calculation.
    Uses sin-based pseudo-noise (no external deps).
    """
    tris = []
    base_r = 20.0
    n_lat = 16
    n_lon = 20

    def pseudo_noise(x, y, z):
        """Simple deterministic pseudo-noise from sin combinations."""
        return (math.sin(x * 1.7 + y * 2.3 + 0.5) * 0.3 +
                math.sin(y * 2.1 + z * 1.9 + 1.2) * 0.25 +
                math.sin(z * 1.3 + x * 2.7 + 2.8) * 0.2 +
                math.sin(x * 3.1 + z * 1.1) * 0.15 +
                math.sin(y * 2.9 + x * 0.7 + z * 1.5) * 0.1)

    def blob_point(lat_i, lon_j):
        theta = math.pi * lat_i / n_lat
        phi = 2 * math.pi * lon_j / n_lon
        # Base sphere coordinates
        nx = math.sin(theta) * math.cos(phi)
        ny = math.sin(theta) * math.sin(phi)
        nz = math.cos(theta)
        # Deform radius by noise
        noise_val = pseudo_noise(nx * 3, ny * 3, nz * 3)
        r = base_r * (1.0 + noise_val * 0.35)
        x = r * nx
        y = r * ny
        z = r * nz + 25  # lift above ground
        return (x, y, z)

    for i in range(n_lat):
        for j in range(n_lon):
            p00 = blob_point(i, j)
            p10 = blob_point(i + 1, j)
            p01 = blob_point(i, j + 1)
            p11 = blob_point(i + 1, j + 1)
            if i > 0:
                tris.append((p00, p10, p01))
            if i < n_lat - 1:
                tris.append((p01, p10, p11))

    return tris


# ---------------------------------------------------------------------------
# Main: generate all models
# ---------------------------------------------------------------------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    models = [
        ("dragon_guardian.stl", generate_dragon, "Dragon Guardian"),
        ("geometric_vase.stl", generate_vase, "Geometric Vase"),
        ("articulated_gear.stl", generate_gear, "Articulated Gear"),
        ("phone_stand.stl", generate_phone_stand, "Phone Stand"),
        ("mini_planter.stl", generate_mini_planter, "Mini Planter"),
        ("chess_piece.stl", generate_chess_piece, "Chess Piece"),
        ("gear_tower.stl", generate_gear_tower, "Gear Tower"),
        ("organic_blob.stl", generate_organic_blob, "Organic Blob"),
    ]

    print("=" * 60)
    print("ForgeFiles Pipeline - Test Model Generator")
    print("=" * 60)

    for filename, gen_func, label in models:
        filepath = os.path.join(script_dir, filename)
        print(f"\nGenerating {label}...")
        triangles = gen_func()
        write_stl(filepath, triangles)

        # Verify
        tri_count, file_size, expected_size = read_stl_info(filepath)
        valid = file_size == expected_size

        print(f"  File:      {filepath}")
        print(f"  Triangles: {tri_count}")
        print(f"  File size: {file_size:,} bytes")
        print(f"  Expected:  {expected_size:,} bytes")
        print(f"  Valid:     {'YES' if valid else 'NO - SIZE MISMATCH'}")

    print("\n" + "=" * 60)
    print("All models generated successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
