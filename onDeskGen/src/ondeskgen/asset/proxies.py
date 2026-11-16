"""Collision proxies measured from the normalized mesh.

Method (works on non-watertight, multi-part generated meshes):
  1. Sample the surface densely.
  2. Cut the object into horizontal bands (``band`` m thick). For each band,
     rasterize the points onto an XY grid (``cell`` m), close sampling gaps
     with a 1-cell dilation, and fill enclosed holes (a leg or a sofa base is
     a hollow ring of surface samples; filling makes it solid). Holes larger
     than ``max_hole_area`` stay open (a goal's top frame must not become a
     roof; a U-shaped sofa back+arms is not closed anyway).
  3. Split each connected footprint into a few axis-aligned rectangles
     (greedy largest-rectangle cover), so a U-shape becomes back + 2 arms
     instead of one box that seals the seat.
  4. Stack rectangles across consecutive bands when their footprints match
     (IoU >= ``merge_iou``): legs become single columns, a table top one slab.
  5. Clean up: drop specks, drop boxes contained in others, snap near-ground
     bottoms to z=0, mark near-circular columns as cylinders.

Because every box comes from occupied cells, gaps between legs, under a table
top, or inside a goal mouth stay open as long as the mesh has them.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, asdict

import numpy as np
import trimesh
from scipy import ndimage


@dataclass
class ProxyParams:
    cell: float = 0.01
    band: float = 0.02
    points_per_m2: float = 60000.0
    max_points: int = 1_500_000
    max_hole_area: float | None = None   # m^2; None = fill all enclosed holes
    solid_fill: float = 0.8              # footprint/bbox ratio above which no decomposition
    rect_coverage: float = 0.95
    max_rects_per_component: int = 6
    min_rect_cells: int = 4
    merge_iou: float = 0.7
    min_box_volume: float = 2e-6         # m^3
    ground_snap: float = 0.03            # bottoms below this are snapped to 0
    round_fill_range: tuple[float, float] = (0.6, 0.9)
    max_boxes: int = 128                 # static boxes are cheap; truncation is recorded
    seed: int = 0


@dataclass
class Proxy:
    name: str
    type: str              # "box" | "cylinder" | "sphere" | "mesh" (convex hull, see hulls.py)
    pos: list[float]       # center, asset frame (mesh: AABB center)
    size: list[float]      # MuJoCo size: box half-extents, cylinder [r, half_h], sphere [r];
                           # mesh: AABB half-extents (bookkeeping only, MuJoCo reads the OBJ)
    file: str | None = None  # mesh only: OBJ path relative to the asset dir, asset-frame vertices

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        p = np.asarray(self.pos)
        if self.type in ("box", "mesh"):
            h = np.asarray(self.size)
        elif self.type == "cylinder":
            h = np.array([self.size[0], self.size[0], self.size[1]])
        else:
            h = np.full(3, self.size[0])
        return p - h, p + h

    def to_dict(self) -> dict:
        d = asdict(self)
        if d["file"] is None:          # primitives keep the original record format
            del d["file"]
        return d


# --------------------------------------------------------------------------- 2D helpers
def largest_rectangle(mask: np.ndarray) -> tuple[int, int, int, int, int]:
    """Largest all-True axis-aligned rectangle. Returns (area, r0, r1, c0, c1) half-open."""
    rows, cols = mask.shape
    heights = np.zeros(cols, dtype=int)
    best = (0, 0, 0, 0, 0)
    for r in range(rows):
        heights = np.where(mask[r], heights + 1, 0)
        stack: list[int] = []
        for c in range(cols + 1):
            h = heights[c] if c < cols else 0
            while stack and heights[stack[-1]] >= h:
                top = stack.pop()
                height = heights[top]
                left = stack[-1] + 1 if stack else 0
                area = height * (c - left)
                if area > best[0]:
                    best = (area, r - height + 1, r + 1, left, c)
            stack.append(c)
    return best


def rectangle_cover(component: np.ndarray, coverage: float, max_rects: int,
                    min_cells: int) -> list[tuple[int, int, int, int]]:
    remaining = component.copy()
    total = int(component.sum())
    rects = []
    while remaining.sum() > (1 - coverage) * total and len(rects) < max_rects:
        area, r0, r1, c0, c1 = largest_rectangle(remaining)
        if area < min_cells:
            break
        rects.append((r0, r1, c0, c1))
        remaining[r0:r1, c0:c1] = False
    return rects


def fill_holes_limited(mask: np.ndarray, max_hole_cells: int | None) -> np.ndarray:
    filled = ndimage.binary_fill_holes(mask)
    if max_hole_cells is None:
        return filled
    holes = filled & ~mask
    lab, n = ndimage.label(holes)
    if n == 0:
        return mask
    sizes = ndimage.sum(holes, lab, index=np.arange(1, n + 1))
    keep = np.isin(lab, 1 + np.nonzero(sizes <= max_hole_cells)[0])
    return mask | keep


def _iou(a, b) -> float:
    ix = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[2], b[2]))
    inter = ix * iy
    ua = (a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - inter
    return inter / ua if ua > 0 else 0.0


# --------------------------------------------------------------------------- main
def fit_boxes(mesh: trimesh.Trimesh, params: ProxyParams | None = None,
              prefix: str = "col", stats: dict | None = None) -> list[Proxy]:
    p = params or ProxyParams()
    n_pts = int(min(p.max_points, max(20000, mesh.area * p.points_per_m2)))
    pts, _ = trimesh.sample.sample_surface(mesh, n_pts, seed=p.seed)
    pts = np.vstack([pts, mesh.vertices])

    lo = pts.min(axis=0) - p.cell
    hi = pts.max(axis=0) + p.cell
    nx = int(np.ceil((hi[0] - lo[0]) / p.cell)) + 1
    ny = int(np.ceil((hi[1] - lo[1]) / p.cell)) + 1
    nb = int(np.ceil((hi[2] - max(lo[2], 0.0)) / p.band))
    z_base = max(lo[2], 0.0)
    ix = np.clip(((pts[:, 0] - lo[0]) / p.cell).astype(int), 0, nx - 1)
    iy = np.clip(((pts[:, 1] - lo[1]) / p.cell).astype(int), 0, ny - 1)
    ib = np.clip(((pts[:, 2] - z_base) / p.band).astype(int), 0, nb - 1)
    max_hole_cells = None if p.max_hole_area is None else int(p.max_hole_area / p.cell ** 2)
    struct = np.ones((3, 3), dtype=bool)

    # open stacks: list of [x0, x1, y0, y1, b0, b1, round_votes, n]
    open_boxes: list[list] = []
    closed: list[list] = []
    for b in range(nb):
        sel = ib == b
        rects: list[tuple] = []
        if np.any(sel):
            grid = np.zeros((nx, ny), dtype=bool)
            grid[ix[sel], iy[sel]] = True
            grid = ndimage.binary_dilation(grid, structure=struct)
            grid = fill_holes_limited(grid, max_hole_cells)
            lab, ncomp = ndimage.label(grid, structure=struct)
            for k in range(1, ncomp + 1):
                comp = lab == k
                xs, ys = np.nonzero(comp)
                x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
                sub = comp[x0:x1, y0:y1]
                fill = sub.mean()
                aspect = max(sub.shape) / max(1, min(sub.shape))
                is_round = p.round_fill_range[0] <= fill <= p.round_fill_range[1] and aspect < 1.3 \
                    and max(sub.shape) <= 12
                if is_round or fill >= p.solid_fill:
                    # compact footprint: one rectangle (jagged sampling edges
                    # would otherwise be peeled off as thin slivers)
                    rects.append((x0, x1, y0, y1, is_round))
                    continue
                for (r0, r1, c0, c1) in rectangle_cover(sub, p.rect_coverage,
                                                        p.max_rects_per_component, p.min_rect_cells):
                    rects.append((x0 + r0, x0 + r1, y0 + c0, y0 + c1, False))
        still_open = []
        used = set()
        for ob in open_boxes:
            match = None
            for i, r in enumerate(rects):
                if i not in used and _iou(ob[:4], r[:4]) >= p.merge_iou:
                    match = i
                    break
            if match is None:
                closed.append(ob)
                continue
            used.add(match)
            r = rects[match]
            ob[0], ob[1] = min(ob[0], r[0]), max(ob[1], r[1])
            ob[2], ob[3] = min(ob[2], r[2]), max(ob[3], r[3])
            ob[5] = b + 1
            ob[6] += int(r[4])
            ob[7] += 1
            still_open.append(ob)
        for i, r in enumerate(rects):
            if i not in used:
                still_open.append([r[0], r[1], r[2], r[3], b, b + 1, int(r[4]), 1])
        open_boxes = still_open
    closed.extend(open_boxes)

    # dilation grew every footprint by one cell on each side: shrink back.
    boxes = []
    for x0, x1, y0, y1, b0, b1, votes, n in closed:
        bx0, bx1 = lo[0] + (x0 + 1) * p.cell, lo[0] + (x1 - 1) * p.cell
        by0, by1 = lo[1] + (y0 + 1) * p.cell, lo[1] + (y1 - 1) * p.cell
        if bx1 - bx0 < p.cell:
            c = 0.5 * (bx0 + bx1)
            bx0, bx1 = c - p.cell / 2, c + p.cell / 2
        if by1 - by0 < p.cell:
            c = 0.5 * (by0 + by1)
            by0, by1 = c - p.cell / 2, c + p.cell / 2
        bz0 = z_base + b0 * p.band
        bz1 = min(z_base + b1 * p.band, pts[:, 2].max())
        if bz0 < p.ground_snap:
            bz0 = 0.0
        boxes.append([bx0, bx1, by0, by1, bz0, bz1, votes / n > 0.5])

    boxes = [bx for bx in boxes if (bx[1] - bx[0]) * (bx[3] - bx[2]) * (bx[5] - bx[4]) >= p.min_box_volume]
    boxes = _drop_contained(boxes)
    boxes.sort(key=lambda bx: -(bx[1] - bx[0]) * (bx[3] - bx[2]) * (bx[5] - bx[4]))
    vol = lambda bx: (bx[1] - bx[0]) * (bx[3] - bx[2]) * (bx[5] - bx[4])  # noqa: E731
    if stats is not None:
        dropped = boxes[p.max_boxes:]
        stats.update({"boxes_found": len(boxes), "boxes_dropped_by_cap": len(dropped),
                      "dropped_volume_m3": float(sum(vol(b) for b in dropped)),
                      "total_volume_m3": float(sum(vol(b) for b in boxes))})
    if len(boxes) > p.max_boxes:
        boxes = boxes[:p.max_boxes]

    proxies = []
    for i, (x0, x1, y0, y1, z0, z1, rnd) in enumerate(boxes):
        c = [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2]
        h = [(x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2]
        if rnd and h[2] > 2 * max(h[0], h[1]):
            proxies.append(Proxy(f"{prefix}_{i}", "cylinder", c, [max(h[0], h[1]), h[2]]))
        else:
            proxies.append(Proxy(f"{prefix}_{i}", "box", c, h))
    return proxies


def _drop_contained(boxes: list[list], tol: float = 0.005) -> list[list]:
    keep = []
    for i, a in enumerate(boxes):
        contained = False
        for j, b in enumerate(boxes):
            if i == j:
                continue
            if (a[0] >= b[0] - tol and a[1] <= b[1] + tol and a[2] >= b[2] - tol and
                    a[3] <= b[3] + tol and a[4] >= b[4] - tol and a[5] <= b[5] + tol):
                # identical boxes: keep the first one only
                if not (abs(np.subtract(a[:6], b[:6])).max() < tol and i < j):
                    contained = True
                    break
        if not contained:
            keep.append(a)
    return keep


# --------------------------------------------------------------------------- voxel variant
@dataclass
class VoxelParams:
    """Alternative: solid voxel grid first, then a 3D greedy box cover.

    Experimental (compared against fit_boxes in scripts/compare_collision.py);
    not the default."""
    pitch: float = 0.01
    points_per_m2: float = 60000.0
    max_points: int = 1_500_000
    close_iters: int = 1                 # 3D closing (dilate+erode) seals sampling gaps
    fill: bool = True                    # fill cavities enclosed in 3D (sofa/cabinet shells)
    merge_min_fill: float = 0.9          # merged box must stay >= this fraction solid; 1.0 = no merge
    floor_cap: bool = True               # the floor closes shells from below (open-bottom cabinets)
    min_box_volume: float = 2e-6         # m^3
    ground_snap: float = 0.03
    max_boxes: int = 128
    seed: int = 0


def voxelize_solid(mesh: trimesh.Trimesh, p: VoxelParams) -> tuple[np.ndarray, np.ndarray]:
    """Occupancy grid (x, y, z) and the world position of voxel (0, 0, 0)'s lower corner.

    Works on non-watertight, multi-part meshes: voxels hit by dense surface
    samples form a shell; closing seals gaps between samples; 3D hole filling
    makes enclosed shells solid while anything open to the outside (the space
    under a table, a goal mouth) stays empty."""
    n_pts = int(min(p.max_points, max(20000, mesh.area * p.points_per_m2)))
    pts, _ = trimesh.sample.sample_surface(mesh, n_pts, seed=p.seed)
    pts = np.vstack([pts, mesh.vertices])
    pad = p.close_iters + 2
    lo = pts.min(axis=0) - pad * p.pitch
    idx = np.floor((pts - lo) / p.pitch).astype(int)
    shape = idx.max(axis=0) + pad + 1
    occ = np.zeros(shape, dtype=bool)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    if p.close_iters:
        occ = ndimage.binary_closing(occ, structure=np.ones((3, 3, 3), bool), iterations=p.close_iters)
    if p.fill:
        if p.floor_cap:
            # a solid plane just below z=0 seals shells that sit open on the floor
            kf = int(np.floor((0.0 - lo[2]) / p.pitch)) - 1
            if 0 <= kf < occ.shape[2]:
                cap = occ[:, :, kf].copy()
                occ[:, :, kf] = True
                occ = ndimage.binary_fill_holes(occ)
                occ[:, :, kf] = cap
            else:
                occ = ndimage.binary_fill_holes(occ)
        else:
            occ = ndimage.binary_fill_holes(occ)
    return occ, lo


def greedy_boxes(occ: np.ndarray) -> list[tuple[int, int, int, int, int, int]]:
    """Cover all occupied voxels with disjoint boxes (half-open voxel index ranges).

    Grows each seed voxel along z first (legs/walls become single columns),
    then y, then x, as far as the whole face stays occupied."""
    rem = occ.copy()
    nx, ny, nz = rem.shape
    boxes = []
    for x, y, z in np.argwhere(rem):
        if not rem[x, y, z]:
            continue
        z1 = z + 1
        while z1 < nz and rem[x, y, z1]:
            z1 += 1
        y1 = y + 1
        while y1 < ny and rem[x, y1, z:z1].all():
            y1 += 1
        x1 = x + 1
        while x1 < nx and rem[x1, y:y1, z:z1].all():
            x1 += 1
        rem[x:x1, y:y1, z:z1] = False
        boxes.append((x, x1, y, y1, z, z1))
    return boxes


def merge_boxes(boxes: list[tuple], occ: np.ndarray, min_fill: float) -> list[tuple]:
    """Bottom-up merge: repeatedly replace the two touching boxes whose bounding
    union contains the fewest empty voxels by that union, as long as the union
    stays >= ``min_fill`` occupied. Exact greedy covers of curved surfaces give
    hundreds of slivers; this trades a bounded amount of filled-in empty space
    for far fewer boxes. Gaps wider than the fill budget (under a seat, a goal
    mouth) cannot be bridged because the union would fall below ``min_fill``."""
    if len(boxes) < 2:
        return list(boxes)
    S = np.zeros(np.array(occ.shape) + 1, dtype=np.int64)
    S[1:, 1:, 1:] = occ.cumsum(0).cumsum(1).cumsum(2)

    def count(B):  # B: (n, 6) half-open voxel boxes
        x0, x1, y0, y1, z0, z1 = B.T
        return (S[x1, y1, z1] - S[x0, y1, z1] - S[x1, y0, z1] - S[x1, y1, z0]
                + S[x0, y0, z1] + S[x0, y1, z0] + S[x1, y0, z0] - S[x0, y0, z0])

    B = np.array(boxes, dtype=np.int64)
    alive = np.ones(len(B), dtype=bool)

    def candidates(i):
        """Merge candidates of box i among alive boxes: (cost, j, union)."""
        a = B[i]
        lo = np.minimum(a[[0, 2, 4]], B[:, [0, 2, 4]])
        hi = np.maximum(a[[1, 3, 5]], B[:, [1, 3, 5]])
        # touching or overlapping (gap <= 1 voxel on every axis)
        near = np.all(np.maximum(a[[0, 2, 4]], B[:, [0, 2, 4]])
                      <= np.minimum(a[[1, 3, 5]], B[:, [1, 3, 5]]) + 1, axis=1)
        near &= alive
        near[i] = False
        js = np.nonzero(near)[0]
        if len(js) == 0:
            return []
        U = np.stack([lo[js, 0], hi[js, 0], lo[js, 1], hi[js, 1], lo[js, 2], hi[js, 2]], axis=1)
        vol = np.prod(U[:, 1::2] - U[:, ::2], axis=1)
        filled = count(U)
        ok = filled >= min_fill * vol
        return [(int(v - f), int(j), U[k]) for k, (j, v, f) in enumerate(zip(js, vol, filled)) if ok[k]]

    heap = []
    for i in range(len(B)):
        for cost, j, _ in candidates(i):
            if j > i:
                heapq.heappush(heap, (cost, i, j))
    while heap:
        cost, i, j = heapq.heappop(heap)
        if not (alive[i] and alive[j]):
            continue
        a, b = B[i], B[j]
        u = np.array([min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]),
                      max(a[3], b[3]), min(a[4], b[4]), max(a[5], b[5])])
        alive[i] = alive[j] = False
        B = np.vstack([B, u])
        alive = np.append(alive, True)
        k = len(B) - 1
        # boxes swallowed by the union disappear
        inside = alive & np.all(B[:, [0, 2, 4]] >= u[[0, 2, 4]], axis=1) & np.all(B[:, [1, 3, 5]] <= u[[1, 3, 5]], axis=1)
        inside[k] = False
        alive[inside] = False
        for c, j2, _ in candidates(k):
            heapq.heappush(heap, (c, min(k, j2), max(k, j2)))
    return [tuple(int(v) for v in b) for b in B[alive]]


def fit_voxel_boxes(mesh: trimesh.Trimesh, params: VoxelParams | None = None,
                    prefix: str = "col", stats: dict | None = None) -> list[Proxy]:
    p = params or VoxelParams()
    occ, lo = voxelize_solid(mesh, p)
    raw = greedy_boxes(occ)
    n_greedy = len(raw)
    if p.merge_min_fill < 1.0:
        raw = merge_boxes(raw, occ, p.merge_min_fill)
    if stats is not None:
        stats["boxes_greedy"] = n_greedy
    boxes = []
    for x0, x1, y0, y1, z0, z1 in raw:
        b = [lo[0] + x0 * p.pitch, lo[0] + x1 * p.pitch, lo[1] + y0 * p.pitch, lo[1] + y1 * p.pitch,
             lo[2] + z0 * p.pitch, lo[2] + z1 * p.pitch]
        if b[4] < p.ground_snap:
            b[4] = 0.0
        boxes.append(b)
    vol = lambda bx: (bx[1] - bx[0]) * (bx[3] - bx[2]) * (bx[5] - bx[4])  # noqa: E731
    small = [b for b in boxes if vol(b) < p.min_box_volume]
    boxes = sorted((b for b in boxes if vol(b) >= p.min_box_volume), key=lambda b: -vol(b))
    dropped = boxes[p.max_boxes:]
    if stats is not None:
        stats.update({"voxels_solid": int(occ.sum()), "grid": list(occ.shape),
                      "boxes_found": len(boxes) + len(small), "boxes_dropped_small": len(small),
                      "boxes_dropped_by_cap": len(dropped),
                      "dropped_volume_m3": float(sum(vol(b) for b in dropped + small)),
                      "total_volume_m3": float(sum(vol(b) for b in boxes + small))})
    boxes = boxes[:p.max_boxes]
    return [Proxy(f"{prefix}_{i}", "box",
                  [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2],
                  [(b[1] - b[0]) / 2, (b[3] - b[2]) / 2, (b[5] - b[4]) / 2])
            for i, b in enumerate(boxes)]


def ball_proxy(radius: float) -> list[Proxy]:
    return [Proxy("col_0", "sphere", [0.0, 0.0, radius], [radius])]


CATEGORY_PARAMS = {
    "sofa": ProxyParams(),
    "table": ProxyParams(),
    "desk": ProxyParams(),
    "chair": ProxyParams(),
    "cabinet": ProxyParams(),
    # goal: keep the frame open from above; fine cells for thin tubes.
    "goal": ProxyParams(cell=0.008, band=0.02, max_hole_area=0.002),
}
