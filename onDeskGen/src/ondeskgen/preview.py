"""Episode -> web replay files.

replay/
  scene.glb        every body of the compiled scene as a node "body~<body name>"
                   (MuJoCo Z-up coordinates; the viewer rotates the root to Y-up), with
                   child nodes "body~<body>|vis" (visual geoms) and "body~<body>|col" (collision geoms:
                   furniture hulls/boxes, robot collision meshes, ball, floor and walls)
  trajectory.bin   float32 [frames, bodies, 7] = world pos xyz + quat wxyz, 25 Hz
  trajectory.json  body names, times, frame count, plus a 25 Hz HUD track (FSM state,
                   command, active policy, ball position)

The robot's visual meshes are decimated (they are ~490k vertices in total) so the
page stays light; everything else is exported as simulated.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
import trimesh

ROBOT_ROOT = "trunk_base"
ROBOT_MAX_FACES = 1500
GEOM = mujoco.mjtGeom


def _quat_mat(q) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, float))
    return m.reshape(3, 3)


def _tf(pos, quat) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _quat_mat(quat)
    T[:3, 3] = pos
    return T


@lru_cache(maxsize=512)
def _decimated(key: bytes, verts: bytes, faces: bytes, nv: int, nf: int, target: int):
    v = np.frombuffer(verts, np.float32).reshape(nv, 3)
    f = np.frombuffer(faces, np.int32).reshape(nf, 3)
    m = trimesh.Trimesh(v, f, process=True)
    if len(m.faces) > target:
        try:
            m = m.simplify_quadric_decimation(face_count=target)
        except Exception:  # noqa: BLE001 -- keep the full mesh rather than fail the export
            pass
    return np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int32)


def _geom_mesh(model: mujoco.MjModel, g: int, decimate: int | None) -> trimesh.Trimesh | None:
    t = model.geom_type[g]
    s = model.geom_size[g]
    if t == GEOM.mjGEOM_MESH:
        mid = model.geom_dataid[g]
        va, nv = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        fa, nf = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        v = model.mesh_vert[va:va + nv].astype(np.float32)
        f = model.mesh_face[fa:fa + nf].astype(np.int32)
        if decimate and nf > decimate:
            v, f = _decimated(bytes([mid % 256]) + str(mid).encode(), v.tobytes(), f.tobytes(), nv, nf, decimate)
        return trimesh.Trimesh(v, f, process=False)
    if t == GEOM.mjGEOM_BOX:
        return trimesh.creation.box(extents=2 * s)
    if t == GEOM.mjGEOM_SPHERE:
        return trimesh.creation.icosphere(subdivisions=2, radius=s[0])
    if t == GEOM.mjGEOM_CYLINDER:
        return trimesh.creation.cylinder(radius=s[0], height=2 * s[1], sections=24)
    if t == GEOM.mjGEOM_CAPSULE:
        return trimesh.creation.capsule(radius=s[0], height=2 * s[1], count=[12, 12])
    if t == GEOM.mjGEOM_ELLIPSOID:
        m = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        m.vertices *= s
        return m
    if t == GEOM.mjGEOM_PLANE:
        sx, sy = (s[0] or 5.0), (s[1] or 5.0)
        m = trimesh.creation.box(extents=[2 * sx, 2 * sy, 0.002])
        m.apply_translation([0, 0, -0.001])
        return m
    return None


def _rgba(model, g) -> np.ndarray:
    mat = model.geom_matid[g]
    rgba = model.mat_rgba[mat] if mat >= 0 else model.geom_rgba[g]
    return np.clip(np.asarray(rgba, float), 0, 1)


def export_scene_glb(model: mujoco.MjModel, out: Path) -> dict:
    """Write scene.glb (body nodes at their qpos0 world pose). Returns counts."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    robot = model.body(ROBOT_ROOT).id if ROBOT_ROOT in [model.body(i).name for i in range(model.nbody)] else -1
    scene = trimesh.Scene(base_frame="__scene_root__")   # not "world": that is MuJoCo body 0
    # node names "body~<MuJoCo body name>": never collide with the glTF scene root ("world")
    names = ["body~" + (model.body(i).name or f"body_{i}") for i in range(model.nbody)]
    n_vis = n_col = 0
    for b in range(model.nbody):
        scene.graph.update(frame_from=scene.graph.base_frame, frame_to=names[b],
                           matrix=_tf(data.xpos[b], data.xquat[b]))
        for layer in ("vis", "col"):
            scene.graph.update(frame_from=names[b], frame_to=f"{names[b]}|{layer}", matrix=np.eye(4))
    for g in range(model.ngeom):
        b = model.geom_bodyid[g]
        is_robot = robot >= 0 and model.body_rootid[b] == robot
        collides = bool(model.geom_contype[g] or model.geom_conaffinity[g])
        layer = "col" if collides else "vis"
        mesh = _geom_mesh(model, g, ROBOT_MAX_FACES if (is_robot and not collides) else None)
        if mesh is None or len(mesh.faces) == 0:
            continue
        rgba = _rgba(model, g)
        if layer == "col" and not is_robot and model.geom_type[g] != GEOM.mjGEOM_PLANE \
                and model.geom_group[g] == 3:
            rgba = np.array([0.55, 0.25, 0.85, 0.45])       # collision proxies: same purple as the check images
        mesh.visual = trimesh.visual.ColorVisuals(mesh, face_colors=(rgba * 255).astype(np.uint8))
        gname = model.geom(g).name or f"geom_{g}"
        scene.add_geometry(mesh, node_name=f"{gname}#{g}", geom_name=f"g{g}",
                           parent_node_name=f"{names[b]}|{layer}",
                           transform=_tf(model.geom_pos[g], model.geom_quat[g]))
        n_vis += layer == "vis"
        n_col += layer == "col"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(scene.export(file_type="glb"))
    return {"bodies": int(model.nbody), "visual_geoms": n_vis, "collision_geoms": n_col,
            "bytes": out.stat().st_size}


def export_trajectory(traj_npz: Path, out_dir: Path, body_names: list[str]) -> dict:
    d = np.load(traj_npz, allow_pickle=False)
    names = [str(x) for x in d["body_names"]]
    if names != body_names:
        raise ValueError("trajectory body order differs from the scene model")
    pos = d["body_pos"].astype(np.float32)
    quat = d["body_quat"].astype(np.float32)
    frames = np.concatenate([pos, quat], axis=2)              # [F, B, 7]
    (out_dir / "trajectory.bin").write_bytes(np.ascontiguousarray(frames).tobytes())
    # HUD track at the body sample times (50 Hz control log -> every 2nd step)
    t = d["t"]
    idx = np.searchsorted(t, d["body_t"]).clip(0, len(t) - 1)
    states = [str(x) for x in d["state_names"]]
    hud = {"state": [states[int(i)] for i in d["state"][idx]],
           "cmd": np.round(d["cmd"][idx], 3).tolist(),
           "policy": [str(x) for x in d["policy"][idx]],
           "ball": np.round(d["ball"][idx], 4).tolist(),
           "trunk": np.round(d["trunk"][idx], 4).tolist()}
    meta = {"format": "float32 [frames][bodies][px,py,pz,qw,qx,qy,qz], MuJoCo world frame (Z up)",
            "frames": int(frames.shape[0]), "bodies": body_names, "dt": float(np.median(np.diff(d["body_t"])))
            if len(d["body_t"]) > 1 else 0.04, "t": np.round(d["body_t"], 3).tolist(), "hud": hud}
    (out_dir / "trajectory.json").write_text(json.dumps(meta, ensure_ascii=False))
    return {"frames": meta["frames"], "duration_s": round(float(d["body_t"][-1]) if len(d["body_t"]) else 0.0, 2)}


def export_replay(scene_xml: Path, traj_npz: Path, out_dir: Path) -> dict:
    """Scene + trajectory of one episode -> out_dir/{scene.glb, trajectory.bin, trajectory.json}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    names = [model.body(i).name for i in range(model.nbody)]
    info = {"scene_glb": out_dir / "scene.glb", "trajectory": out_dir / "trajectory.json"}
    info.update(export_scene_glb(model, out_dir / "scene.glb"))
    info.update(export_trajectory(traj_npz, out_dir, names))
    return info
