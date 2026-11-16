"""Visual mesh for MuJoCo: decimated, colors baked into a few rgba materials.

Quadric decimation drops UVs, so textures cannot survive it. Instead the base
color of every original face is sampled (normalize.mesh_face_colors), carried
to the decimated faces by nearest face centroid, and quantized with k-means
into a handful of flat materials. Each color cluster is exported as one OBJ.
The original GLB stays in assets/raw for offline rendering (Unity replay).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


MIN_PART_FACES = 8


@dataclass
class VisualPart:
    file: str             # relative to the asset directory
    rgba: tuple[float, float, float, float]
    faces: int


def decimate(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    if len(mesh.faces) <= target_faces:
        return mesh.copy()
    # Generated meshes duplicate vertices along UV seams; the decimator treats
    # those as open boundaries and stalls (goal: 284k -> 26k faces instead of
    # 4k). Colors are carried per face, so merging by position is safe.
    welded = mesh.copy()
    welded.merge_vertices(merge_tex=True, merge_norm=True)
    out = welded.simplify_quadric_decimation(face_count=target_faces)
    out.remove_unreferenced_vertices()
    return out


def kmeans(x: np.ndarray, k: int, iters: int = 30, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Plain k-means (k-means++ init). Returns (labels, centers)."""
    rng = np.random.default_rng(seed)
    k = min(k, len(np.unique(x.round(3), axis=0)))
    centers = [x[rng.integers(len(x))]]
    for _ in range(1, k):
        d2 = np.min(((x[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
        if d2.sum() == 0:
            break
        centers.append(x[rng.choice(len(x), p=d2 / d2.sum())])
    centers = np.array(centers)
    for _ in range(iters):
        labels = np.argmin(((x[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
        new = np.array([x[labels == i].mean(0) if np.any(labels == i) else centers[i]
                        for i in range(len(centers))])
        if np.allclose(new, centers, atol=1e-5):
            break
        centers = new
    return labels, centers


def build_visual(mesh: trimesh.Trimesh, face_colors: np.ndarray, out_dir: Path,
                 target_faces: int = 4000, n_colors: int = 6,
                 subdir: str = "visual") -> list[VisualPart]:
    dec = decimate(mesh, target_faces)
    tree = cKDTree(mesh.triangles_center)
    _, idx = tree.query(dec.triangles_center)
    colors = face_colors[idx]
    # weight clustering by face area so tiny slivers do not get their own color
    area = dec.area_faces
    rep = np.repeat(np.arange(len(colors)), np.clip((area / area.mean() * 2).astype(int), 1, 20))
    _, centers = kmeans(colors[rep], n_colors)
    labels = np.argmin(((colors[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
    # A cluster of a few slivers can have < 4 vertices, which MuJoCo rejects as a mesh
    # (seen on a generated chair): fold such clusters into the nearest kept color.
    counts = np.bincount(labels, minlength=len(centers))
    keep = counts >= MIN_PART_FACES
    if keep.any() and not keep.all():
        d = ((centers[:, None, :] - centers[None]) ** 2).sum(-1)
        d[:, ~keep] = np.inf
        labels = np.argmin(d, axis=1)[labels]

    vis_dir = out_dir / subdir
    vis_dir.mkdir(parents=True, exist_ok=True)
    for f in vis_dir.glob("part_*.obj"):
        f.unlink()
    parts = []
    for i, c in enumerate(centers):
        sel = np.nonzero(labels == i)[0]
        if len(sel) == 0:
            continue
        sub = dec.submesh([sel], append=True, repair=False)
        name = f"{subdir}/part_{len(parts)}.obj"
        sub.export(out_dir / name, include_normals=True, include_color=False, include_texture=False)
        parts.append(VisualPart(name, (float(c[0]), float(c[1]), float(c[2]), 1.0), int(len(sel))))
    return parts
