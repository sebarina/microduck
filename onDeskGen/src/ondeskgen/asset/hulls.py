"""Convex-decomposition collision (CoACD) and mass properties for movable furniture.

Method:
  1. CoACD (Wei et al., SIGGRAPH 2022; https://github.com/SarahWeiii/CoACD)
     splits the normalized full-resolution mesh into convex parts. Its
     "auto" preprocessing remeshes non-watertight input into a manifold
     first, which inflates the surface by a few millimeters (measured 4-11 mm
     at the bottom on Lux3D furniture).
  2. Every part is clipped at the floor (z >= 0) and re-hulled, so the asset
     still rests at z = 0 and does not pop up when placed in a scene.
  3. A hull that stands on the floor with a wide base (> ``max_base_extent``)
     is cut by vertical planes into columns. MuJoCo returns about 3 contacts
     per convex mesh against a plane; a cabinet resting on a single 0.5 x 1.6 m
     hull then rocks on a changing triangle of points and creeps (measured
     3.9 mm in 5 s). The columns' union is exactly the original hull.
  4. Each hull is written as an OBJ and becomes one MuJoCo mesh geom. MuJoCo
     collides a mesh geom as its convex hull, which is exact here.

Mass properties: the hulls are treated as one uniform-density solid (overlaps
counted twice, CoACD parts barely overlap), then scaled to a per-category
reference mass. Real furniture concentrates mass in its frame; a uniform
solid is an approximation, recorded as such in asset.json.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import trimesh
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection, QhullError

from .proxies import Proxy

# Reference masses (kg) and furniture-floor friction come from the engine
# configuration (configs/ondeskgen.json): estimates from typical retail
# furniture, not measurements. Friction is applied through explicit <pair>
# contacts in the scene so the floor keeps mu = 1.0 for the robot's feet.
from ondeskgen.config import engine_config as _engine_config  # noqa: E402

CATEGORY_MASS = {c: float(v["mass_kg"]) for c, v in _engine_config()["categories"].items()}
FURNITURE_FLOOR_FRICTION = float(_engine_config()["physics"]["floor_friction"])


@dataclass
class HullParams:
    threshold: float = 0.05          # CoACD concavity threshold (lower = more, tighter hulls)
    max_hulls: int = 64              # -1 = unlimited
    max_hull_vertices: int = 64
    preprocess_resolution: int = 50
    resolution: int = 2000
    mcts_iterations: int = 150
    max_base_extent: float = 0.3     # m: floor-touching hulls wider than this are cut into columns
    seed: int = 0


@dataclass
class MassProps:
    mass: float
    com: list[float]                  # asset frame
    fullinertia: list[float]          # about the CoM: ixx iyy izz ixy ixz iyz
    volume_m3: float
    source: str


LEVEL_TOL = 0.010   # m: a floor-standing hull whose bottom is this close to the floor gets "levelling feet"


def _level(hv: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Stretch a hull's bottom down to the floor when it floats by less than LEVEL_TOL (Lux3D legs of
    slightly different length: a 4-legged table then rests on 3 and wobbles). Returns (vertices, gap)."""
    zmin = float(hv[:, 2].min())
    if not 0.0 < zmin < LEVEL_TOL:
        return None
    hv = hv.copy()
    hv[hv[:, 2] < zmin + 0.002, 2] = 0.0
    return hv, zmin


def _clip_to_floor(v: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    v = v.copy()
    v[:, 2] = np.maximum(v[:, 2], 0.0)
    return _hull(v)


def _hull(v: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        h = ConvexHull(v)
    except QhullError:
        return None                    # degenerate (flat) part
    if h.volume < 1e-9:
        return None
    hv = v[h.vertices]
    remap = -np.ones(len(v), int)
    remap[h.vertices] = np.arange(len(h.vertices))
    faces = remap[h.simplices]
    # orient faces outward
    c = hv.mean(0)
    for i, f in enumerate(faces):
        n = np.cross(hv[f[1]] - hv[f[0]], hv[f[2]] - hv[f[0]])
        if np.dot(n, hv[f[0]] - c) < 0:
            faces[i] = f[[0, 2, 1]]
    return hv, faces


def _split_columns(hv: np.ndarray, max_extent: float) -> list[np.ndarray]:
    """Cut a convex hull with a wide floor contact into vertical columns (vertex sets)."""
    base = hv[hv[:, 2] < 0.002, :2]
    if len(base) < 3 or np.ptp(base, axis=0).max() <= max_extent:
        return [hv]
    lo, hi = hv[:, :2].min(0), hv[:, :2].max(0)
    n = np.maximum(1, np.ceil((hi - lo) / max_extent).astype(int))
    eq = ConvexHull(hv).equations                       # A x + b <= 0
    xs, ys = np.linspace(lo[0], hi[0], n[0] + 1), np.linspace(lo[1], hi[1], n[1] + 1)
    out = []
    for i in range(n[0]):
        for j in range(n[1]):
            cut = np.array([[-1, 0, 0, xs[i]], [1, 0, 0, -xs[i + 1]], [0, -1, 0, ys[j]], [0, 1, 0, -ys[j + 1]]], float)
            hs = np.vstack([eq, cut])
            norm = np.linalg.norm(hs[:, :3], axis=1)
            # Chebyshev center: interior point required by the halfspace intersection
            r = linprog([0, 0, 0, -1], A_ub=np.hstack([hs[:, :3], norm[:, None]]), b_ub=-hs[:, 3],
                        bounds=[(None, None)] * 3 + [(0, None)], method="highs")
            if not r.success or r.x[3] < 1e-4:
                continue
            pts = HalfspaceIntersection(hs, r.x[:3]).intersections
            out.append(pts)
    return out or [hv]


def fit_convex_hulls(mesh: trimesh.Trimesh, out_dir: Path, params: HullParams | None = None,
                     prefix: str = "hull", stats: dict | None = None, clear: bool = True) -> list[Proxy]:
    import coacd

    p = params or HullParams()
    coacd.set_log_level("error")
    parts = coacd.run_coacd(coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces)),
                            threshold=p.threshold, max_convex_hull=p.max_hulls,
                            preprocess_mode="auto", preprocess_resolution=p.preprocess_resolution,
                            resolution=p.resolution, mcts_iterations=p.mcts_iterations,
                            # CoACD applies max_ch_vertex only with decimate=True (without it the limit
                            # was silently ignored: sofa hulls had up to 716 vertices; found 2026-11-11)
                            max_ch_vertex=p.max_hull_vertices, decimate=p.max_hull_vertices > 0,
                            merge=True, seed=p.seed)
    col_dir = Path(out_dir) / "collision"
    col_dir.mkdir(parents=True, exist_ok=True)
    if clear:
        for old in col_dir.glob("*.obj"):
            old.unlink()
    proxies, zmin_raw, dropped, n_split, levelled = [], 0.0, 0, 0, []
    for v, f in parts:
        v = np.asarray(v, float)
        zmin_raw = min(zmin_raw, float(v[:, 2].min()))
        clipped = _clip_to_floor(v)
        if clipped is None:
            dropped += 1
            continue
        lv = _level(clipped[0]) if prefix == "hull" else None   # base only: never pull a drawer onto the floor
        if lv is not None and (h := _hull(lv[0])) is not None:
            clipped = h
            levelled.append(round(lv[1] * 1000, 1))
        cols = _split_columns(clipped[0], p.max_base_extent)
        n_split += len(cols) > 1
        for cv in cols:
            piece = clipped if len(cols) == 1 else _hull(cv)
            if piece is None:
                continue
            hv, hf = piece
            i = len(proxies)
            fname = f"collision/{prefix}_{i}.obj"
            trimesh.Trimesh(hv, hf, process=False).export(Path(out_dir) / fname)
            lo, hi = hv.min(0), hv.max(0)
            proxies.append(Proxy(f"{prefix}_{i}", "mesh", ((lo + hi) / 2).tolist(), ((hi - lo) / 2).tolist(),
                                 file=fname))
    if stats is not None:
        stats.update({"method": "coacd", "params": asdict(p), "hulls_raw": len(parts),
                      "hulls_dropped_degenerate": dropped, "hulls_split_into_columns": n_split,
                      "hulls": len(proxies), "floor_clip_mm": round(-zmin_raw * 1000, 2),
                      "levelled_hulls_mm": levelled})
    return proxies


def hull_meshes(proxies: list[Proxy], asset_dir: Path) -> list[trimesh.Trimesh]:
    return [trimesh.load(Path(asset_dir) / pr.file, process=False) for pr in proxies if pr.type == "mesh"]


def mass_properties(solids: list[trimesh.Trimesh], mass: float) -> MassProps:
    """Uniform solid made of the given closed convex parts, scaled to ``mass``."""
    vols = np.array([s.volume for s in solids])
    coms = np.array([s.center_mass for s in solids])
    V = float(vols.sum())
    com = (vols[:, None] * coms).sum(0) / V
    rho = mass / V
    I = np.zeros((3, 3))
    for s, vol, c in zip(solids, vols, coms):
        Ii = np.asarray(s.moment_inertia) * rho       # trimesh: unit density, about the part CoM
        d = c - com
        I += Ii + rho * vol * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return MassProps(mass=float(mass), com=com.tolist(),
                     fullinertia=[I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]],
                     volume_m3=V, source="uniform solid over collision hulls, scaled to category mass")


def box_solids(proxies: list[Proxy]) -> list[trimesh.Trimesh]:
    """Box/cylinder proxies as solids (mass properties for the layered/voxel methods)."""
    out = []
    for pr in proxies:
        lo, hi = pr.bounds()
        out.append(trimesh.creation.box(extents=hi - lo, transform=trimesh.transformations.translation_matrix(
            (lo + hi) / 2)))
    return out
