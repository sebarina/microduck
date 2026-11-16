"""Lux3D articulation output (URDF + per-link GLB) -> links and joints in the model's raw frame.

Lux3D ``articulation_animation`` (input: a part-split GLB + a prompt such as
"抽屉可以拉出") returns a zip with ``urdf/robot.urdf`` and ``urdf/meshes/<link>.glb``.
Observed on the probe desk (task 6354581):
  - links: base + one link per moving part; joints prismatic (drawers) with an
    axis in the model's glTF frame (Y up) and limits [0, 0.4] m;
  - masses/inertias are placeholders (1 kg / 0.001): ignored, OnDeskGen computes
    its own from the collision hulls;
  - the URDF root frame is the model frame shifted so the bottom is at y = 0;
  - the closed drawers sat ~4 cm in front of the model's front face.
So: links are aligned to the generated model by matching the minimum corners
of their union and the whole mesh, and a moving part that protrudes beyond the
whole model along its joint axis by more than ``tol`` is moved back to flush
(the correction is recorded).
"""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

from ondeskgen.asset.normalize import mesh_face_colors


@dataclass
class Link:
    name: str
    mesh: trimesh.Trimesh          # in the model's raw frame (closed configuration)
    face_colors: np.ndarray


@dataclass
class Joint:
    name: str
    type: str                      # prismatic | revolute | continuous | fixed
    parent: str
    child: str
    origin: np.ndarray             # child frame origin, raw model frame
    axis: np.ndarray               # unit vector, raw model frame
    lower: float
    upper: float


@dataclass
class Articulation:
    links: list[Link]
    joints: list[Joint]
    corrections: list[dict] = field(default_factory=list)
    source: dict = field(default_factory=dict)

    @property
    def root(self) -> str:
        children = {j.child for j in self.joints}
        return next(lk.name for lk in self.links if lk.name not in children)

    def front_vector(self) -> np.ndarray | None:
        """Mean pull-out direction of prismatic joints (raw frame): the furniture's front."""
        ax = [j.axis for j in self.joints if j.type == "prismatic"]
        if not ax:
            return None
        v = np.mean(ax, axis=0)
        return v / np.linalg.norm(v) if np.linalg.norm(v) > 1e-9 else None


def _rpy_matrix(rpy) -> np.ndarray:
    return trimesh.transformations.euler_matrix(*rpy, axes="sxyz")[:3, :3]


def _load_glb_bytes(data: bytes) -> tuple[trimesh.Trimesh, np.ndarray]:
    scene = trimesh.load(io.BytesIO(data), file_type="glb", force="scene")
    meshes, colors = [], []
    for node in scene.graph.nodes_geometry:
        T, g = scene.graph[node]
        geom = scene.geometry[g]
        if not isinstance(geom, trimesh.Trimesh) or not len(geom.faces):
            continue
        m = geom.copy()
        m.apply_transform(T)
        meshes.append(trimesh.Trimesh(m.vertices, m.faces, process=False))
        colors.append(mesh_face_colors(geom))
    return trimesh.util.concatenate(meshes), np.concatenate(colors)


def load_articulation(zip_path: Path, whole: trimesh.Trimesh, tol: float = 0.01) -> Articulation:
    """Parse the Lux3D articulation zip and express everything in ``whole``'s frame."""
    z = zipfile.ZipFile(zip_path)
    urdf_name = next(n for n in z.namelist() if n.endswith(".urdf"))
    base_dir = str(Path(urdf_name).parent)
    root = ET.fromstring(z.read(urdf_name))
    raw_links: dict[str, tuple[trimesh.Trimesh, np.ndarray]] = {}
    for lk in root.findall("link"):
        mesh_el = lk.find("./visual/geometry/mesh")
        if mesh_el is None:
            continue
        rel = mesh_el.get("filename")
        raw_links[lk.get("name")] = _load_glb_bytes(z.read(f"{base_dir}/{rel}"))
    joints = []
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = np.array([float(x) for x in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
        rpy = np.array([float(x) for x in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()])
        ax_el = j.find("axis")
        axis = np.array([float(x) for x in (ax_el.get("xyz") if ax_el is not None else "1 0 0").split()])
        lim = j.find("limit")
        joints.append(dict(name=j.get("name"), type=j.get("type"), parent=j.find("parent").get("link"),
                           child=j.find("child").get("link"), xyz=xyz, R=_rpy_matrix(rpy), axis=axis,
                           lower=float(lim.get("lower", 0)) if lim is not None else 0.0,
                           upper=float(lim.get("upper", 0)) if lim is not None else 0.0))
    # world (URDF root) transforms at q = 0, parents first
    world_T = {n: np.eye(4) for n in raw_links}
    done = {n for n in raw_links if n not in {j["child"] for j in joints}}
    pending = list(joints)
    while pending:
        progressed = False
        for j in list(pending):
            if j["parent"] in done:
                T = np.eye(4)
                T[:3, :3] = j["R"]
                T[:3, 3] = j["xyz"]
                world_T[j["child"]] = world_T[j["parent"]] @ T
                done.add(j["child"])
                pending.remove(j)
                progressed = True
        if not progressed:
            raise ValueError("URDF joint tree has a cycle or a missing parent")
    meshes = {}
    for n, (m, c) in raw_links.items():
        mm = m.copy()
        mm.apply_transform(world_T[n])
        meshes[n] = (mm, c)
    union = trimesh.util.concatenate([m for m, _ in meshes.values()])
    shift = whole.bounds[0] - union.bounds[0]                  # URDF root -> model frame
    links = []
    for n, (m, c) in meshes.items():
        m.apply_translation(shift)
        links.append(Link(n, m, c))
    out_joints, corrections = [], []
    by_name = {lk.name: lk for lk in links}
    for j in joints:
        axis_w = world_T[j["parent"]][:3, :3] @ j["R"] @ j["axis"]
        axis_w = axis_w / max(np.linalg.norm(axis_w), 1e-12)
        origin = world_T[j["child"]][:3, 3] + shift
        if j["type"] == "prismatic":
            child = by_name[j["child"]].mesh
            proj_child = (child.vertices @ axis_w).max()
            proj_whole = (whole.vertices @ axis_w).max()
            excess = float(proj_child - proj_whole)
            if excess > tol:
                child.apply_translation(-excess * axis_w)
                origin = origin - excess * axis_w
                corrections.append({"joint": j["name"], "moved_back_m": round(excess, 4),
                                    "why": "closed part protruded beyond the generated model's front"})
        out_joints.append(Joint(j["name"], j["type"], j["parent"], j["child"], origin, axis_w, j["lower"], j["upper"]))
    return Articulation(links, out_joints, corrections,
                        {"urdf": urdf_name, "frame_shift_m": [round(float(x), 4) for x in shift],
                         "urdf_masses_ignored": True})
