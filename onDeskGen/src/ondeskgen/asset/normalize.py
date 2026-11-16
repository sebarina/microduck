"""Normalize a generated mesh: Z up, meters, origin at bottom center, front +X.

Order of operations (all rigid except the final uniform scale):
  1. up-axis conversion (glTF is Y-up)
  2. principal alignment in XY: the minimum-area bounding rectangle of the
     footprint is aligned with the axes, longer side along Y (= width)
  3. front: either an explicit yaw override, or a category rule comparing
     where the upper part of the object sits along X (see ``front_rule``)
  4. uniform scale so the ``scale_by`` extent matches the spec
  5. translate so the XY bbox center is at the origin and min z = 0
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh

from ondeskgen.paths import AssetSpec

UP_AXIS_TO_Z = {
    "y": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float),  # +Y -> +Z
    "z": np.eye(3),
}


@dataclass
class NormalizedMesh:
    mesh: trimesh.Trimesh
    face_colors: np.ndarray            # (F, 3) float in [0, 1], per face of ``mesh``
    transform: np.ndarray              # 4x4 applied to the raw (concatenated) mesh
    scale: float
    report: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- loading
def _sample_texture(image, uv: np.ndarray) -> np.ndarray:
    img = np.asarray(image.convert("RGB"), dtype=float) / 255.0
    h, w = img.shape[:2]
    u = np.mod(uv[:, 0], 1.0)
    v = np.mod(uv[:, 1], 1.0)
    px = np.clip((u * (w - 1)).round().astype(int), 0, w - 1)
    py = np.clip(((1.0 - v) * (h - 1)).round().astype(int), 0, h - 1)
    return img[py, px]


def mesh_face_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    """Best-effort base color per face (texture at face-centroid UV, else factors)."""
    n = len(mesh.faces)
    default = np.full((n, 3), 0.7)
    vis = mesh.visual
    try:
        if isinstance(vis, trimesh.visual.TextureVisuals):
            mat = vis.material
            factor = None
            image = None
            if isinstance(mat, trimesh.visual.material.PBRMaterial):
                image = mat.baseColorTexture
                if mat.baseColorFactor is not None:
                    factor = np.asarray(mat.baseColorFactor[:3], dtype=float) / 255.0
            elif isinstance(mat, trimesh.visual.material.SimpleMaterial):
                image = mat.image
                if mat.diffuse is not None:
                    factor = np.asarray(mat.diffuse[:3], dtype=float) / 255.0
            if image is not None and vis.uv is not None and len(vis.uv) == len(mesh.vertices):
                uv = vis.uv[mesh.faces].mean(axis=1)
                colors = _sample_texture(image, uv)
                return colors * factor if factor is not None else colors
            if factor is not None:
                return np.tile(factor, (n, 1))
        elif isinstance(vis, trimesh.visual.ColorVisuals) and vis.kind is not None:
            return np.asarray(vis.face_colors[:, :3], dtype=float) / 255.0
    except Exception:  # noqa: BLE001 -- color is cosmetic; never fail the pipeline on it
        pass
    return default


def load_mesh_with_colors(path) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Load a GLB/OBJ/... as one mesh (scene transforms applied) + per-face colors."""
    scene = trimesh.load(path, force="scene")
    meshes, colors = [], []
    for node in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node]
        geom = scene.geometry[geom_name]
        if not isinstance(geom, trimesh.Trimesh) or len(geom.faces) == 0:
            continue
        m = geom.copy()
        m.apply_transform(transform)
        meshes.append(m)
        colors.append(mesh_face_colors(geom))
    if not meshes:
        raise ValueError(f"{path}: no triangle geometry found")
    merged = trimesh.util.concatenate([trimesh.Trimesh(m.vertices, m.faces, process=False)
                                       for m in meshes])
    return merged, np.concatenate(colors, axis=0)


# --------------------------------------------------------------------------- geometry
def min_area_rect_angle(xy: np.ndarray) -> tuple[float, np.ndarray]:
    """Angle (rad) that rotates the min-area bounding rectangle onto the axes.

    Returns (theta, extents) where rotating points by ``theta`` makes the
    rectangle axis-aligned, and extents are its (dx, dy) after rotation.
    """
    from scipy.spatial import ConvexHull

    hull = xy[ConvexHull(xy).vertices]
    edges = np.diff(np.vstack([hull, hull[:1]]), axis=0)
    angles = np.unique(np.mod(np.arctan2(edges[:, 1], edges[:, 0]), np.pi / 2))
    best = (np.inf, 0.0, None)
    for a in angles:
        c, s = np.cos(-a), np.sin(-a)
        r = hull @ np.array([[c, s], [-s, c]])
        ext = r.max(axis=0) - r.min(axis=0)
        area = ext[0] * ext[1]
        if area < best[0] - 1e-12:
            best = (area, -a, ext)
    return best[1], best[2]


def rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def upper_part_offset(mesh: trimesh.Trimesh, frac: float = 0.6, n: int = 40000,
                      seed: int = 0) -> np.ndarray:
    """XY offset of the upper part's centroid from the footprint center,
    normalized by the footprint half-extents (so it is scale-free)."""
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    z0, z1 = pts[:, 2].min(), pts[:, 2].max()
    upper = pts[pts[:, 2] > z0 + frac * (z1 - z0)]
    lo, hi = pts[:, :2].min(axis=0), pts[:, :2].max(axis=0)
    if not len(upper):
        return np.zeros(2)
    return (upper[:, :2].mean(axis=0) - (lo + hi) / 2) / np.maximum((hi - lo) / 2, 1e-9)


def lower_part_offset(mesh: trimesh.Trimesh, frac: float = 0.8, n: int = 40000, seed: int = 0) -> np.ndarray:
    """XY offset of the centroid of the part below ``frac`` of the height (under a desk top),
    normalized by the footprint half-extents."""
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    z0, z1 = pts[:, 2].min(), pts[:, 2].max()
    low = pts[pts[:, 2] < z0 + frac * (z1 - z0)]
    lo, hi = pts[:, :2].min(axis=0), pts[:, :2].max(axis=0)
    if not len(low):
        return np.zeros(2)
    return (low[:, :2].mean(axis=0) - (lo + hi) / 2) / np.maximum((hi - lo) / 2, 1e-9)


# --------------------------------------------------------------------------- main
def normalize(mesh: trimesh.Trimesh, face_colors: np.ndarray, spec: AssetSpec,
              up_axis: str = "y") -> NormalizedMesh:
    m = mesh.copy()
    T = np.eye(4)

    def apply(R3: np.ndarray):
        nonlocal T
        H = np.eye(4)
        H[:3, :3] = R3
        m.apply_transform(H)
        T = H @ T

    apply(UP_AXIS_TO_Z[up_axis])
    raw_extents = m.extents.copy()

    report: dict = {"up_axis": up_axis}
    if spec.category != "ball":
        theta, ext = min_area_rect_angle(m.vertices[:, :2])
        apply(rot_z(theta))
        if m.extents[0] > m.extents[1]:          # longer footprint side -> Y (width)
            apply(rot_z(np.pi / 2))
        report["align_deg"] = float(np.degrees(theta))
        fv = (spec.extra or {}).get("front_vector_raw")
        if spec.yaw_deg is not None:
            apply(rot_z(np.radians(spec.yaw_deg)))
            report["front"] = f"yaw override {spec.yaw_deg} deg"
        elif fv is not None:
            # e.g. the pull-out direction of Lux3D-articulated drawers: that side is the front (+X);
            # prefer the two yaws that keep the long side on Y
            v = T[:3, :3] @ np.asarray(fv, float)
            cands = [0, 2] if abs(v[0]) > 0.5 * np.linalg.norm(v[:2]) else [0, 1, 2, 3]
            k = max(cands, key=lambda k: (rot_z(k * np.pi / 2) @ v)[0])
            if k:
                apply(rot_z(k * np.pi / 2))
            report["front"] = f"front vector (articulation pull direction) -> rotated {90 * k} deg"
        elif spec.front_rule in ("low_side", "high_side"):
            # The tall part (sofa/chair backrest) is the back, the goal's frame
            # is the front. Choose among the four 90-degree yaws the one that
            # puts it at -X (low_side) / +X (high_side). This overrides the
            # "longer side = width" convention, which fails e.g. for a chair
            # whose reclined back makes it deeper than wide.
            u = upper_part_offset(m)
            sign = 1.0 if spec.front_rule == "high_side" else -1.0
            k = int(np.argmax([sign * (rot_z(k * np.pi / 2)[:2, :2] @ u)[0] for k in range(4)]))
            if k:
                apply(rot_z(k * np.pi / 2))
            report["front"] = (f"{spec.front_rule}: upper-part offset (x,y)=({u[0]:+.3f},{u[1]:+.3f}) "
                               f"of half-extent -> rotated {90 * k} deg")
        elif spec.front_rule == "legroom":
            # Desk: the side you put your legs in is the front. Below the top, the
            # back is closed (modesty panel, pedestal backs) more than the front, so
            # the lower part's centroid sits toward the back: put it at -X. Only the
            # two yaws that keep the long side on Y are considered.
            u = lower_part_offset(m)
            if abs(u[0]) < 0.08:
                report["front"] = (f"legroom: ambiguous (lower-part offset x={u[0]:+.3f}); kept, check the "
                                   "render or pass yaw_deg")
            else:
                if u[0] > 0:
                    apply(rot_z(np.pi))
                report["front"] = (f"legroom: lower-part offset x={u[0]:+.3f} of half-depth -> "
                                   f"rotated {180 if u[0] > 0 else 0} deg")
        else:
            report["front"] = "none (symmetric category)"

    scale_by = spec.scale_by
    if scale_by == "lux3d":
        # Lux3D's ai_predict_size gives metric meshes; keep them when plausible for the category.
        rng = (spec.extra or {}).get("size_range") or {}
        ext0 = {"depth": float(m.extents[0]), "width": float(m.extents[1]), "height": float(m.extents[2])}
        bad = {k: round(v, 3) for k, v in ext0.items() if k in rng and not rng[k][0] <= v <= rng[k][1]}
        if bad:
            scale_by = (spec.extra or {}).get("fallback_scale_by", "width")
            report["lux3d_size"] = {"kept": False, "extents": ext0, "out_of_range": bad, "fallback": scale_by}
        else:
            report["lux3d_size"] = {"kept": True, "extents": ext0}
    if scale_by == "lux3d":
        scale = 1.0
    else:
        axis_index = {"depth": 0, "width": 1, "height": 2}[scale_by]
        measured = float(m.extents[axis_index])
        if measured <= 0:
            raise ValueError(f"{spec.name}: zero extent along {scale_by}")
        scale = spec.target(scale_by) / measured
    S = np.eye(4)
    S[:3, :3] *= scale
    m.apply_transform(S)
    T = S @ T

    lo, hi = m.bounds
    offset = -np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    Tr = np.eye(4)
    Tr[:3, 3] = offset
    m.apply_transform(Tr)
    T = Tr @ T

    ext = m.extents
    report.update({
        "raw_extents_after_up": [float(x) for x in raw_extents],
        "scale": scale,
        "scale_by": scale_by,
        "extents_xyz": [float(x) for x in ext],
        "target": {"depth": spec.depth, "width": spec.width, "height": spec.height},
        "rel_error": {
            "depth": float(ext[0] / spec.depth - 1),
            "width": float(ext[1] / spec.width - 1),
            "height": float(ext[2] / spec.height - 1),
        },
    })
    return NormalizedMesh(mesh=m, face_colors=face_colors, transform=T, scale=scale, report=report)
