"""Per-asset MJCF fragments.

Fragment layout (assets/processed/<name>/<name>.xml):
  - one body named <name>; static assets have no joint (welded to world when
    attached) and no mass; dynamic assets (ball, and furniture) get
    a freejoint + explicit inertial (the scene builder may remove the
    freejoint to weld furniture again: build_scene(furniture="static"))
  - visual geoms: group 2, contype/conaffinity 0, density 0 (pure graphics)
  - collision geoms: group 3 (hidden by default in the viewer, toggle with
    key '3'), explicit friction/condim/solref/solimp; box/cylinder primitives
    or convex-hull mesh geoms (collision/hull_*.obj, CoACD)
Mesh files are referenced relative to the fragment; ``load_fragment_spec``
rewrites them to absolute paths so the fragment can be attached into any
scene with MjSpec.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import mujoco

from .proxies import Proxy
from .visual import VisualPart

COLLISION_RGBA = "0.55 0.25 0.85 0.45"


@dataclass
class ContactParams:
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
    condim: int = 3
    solref: tuple[float, float] = (0.02, 1.0)
    solimp: tuple[float, float, float, float, float] = (0.9, 0.95, 0.001, 0.5, 2.0)

    def attrs(self) -> dict:
        return {
            "friction": _fmt(self.friction),
            "condim": str(self.condim),
            "solref": _fmt(self.solref),
            "solimp": _fmt(self.solimp),
        }


@dataclass
class DynamicParams:
    mass: float
    diaginertia: tuple[float, float, float] | None = None
    com: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fullinertia: tuple[float, ...] | None = None    # ixx iyy izz ixy ixz iyz (about the CoM)


@dataclass
class ChildBody:
    """A moving part (drawer, door) attached to the root body by one joint (OnDeskGen).
    Positions are in the asset frame; ``pos`` is the joint frame origin in the closed state."""
    name: str
    pos: tuple[float, float, float]
    joint_type: str                          # "slide" | "hinge"
    axis: tuple[float, float, float]
    range: tuple[float, float]
    visual_parts: list[VisualPart]
    proxies: list[Proxy]
    dynamic: DynamicParams                   # com in the asset frame
    damping: float = 2.0
    frictionloss: float = 0.5
    armature: float = 0.01


@dataclass
class FragmentSpec:
    name: str
    visual_parts: list[VisualPart]
    proxies: list[Proxy]
    contact: ContactParams = field(default_factory=ContactParams)
    dynamic: DynamicParams | None = None
    children: list[ChildBody] = field(default_factory=list)


def _fmt(v) -> str:
    return " ".join(f"{x:.6g}" for x in v)


def _inertial(parent: ET.Element, dyn: DynamicParams, pos) -> None:
    inertia = ({"fullinertia": _fmt(dyn.fullinertia)} if dyn.fullinertia is not None
               else {"diaginertia": _fmt(dyn.diaginertia)})
    ET.SubElement(parent, "inertial", pos=_fmt(pos), mass=f"{dyn.mass:.6g}", **inertia)


def _geoms(body: ET.Element, prefix: str, vis_names: list[str], visual_parts, proxies, shift, contact) -> None:
    for vn, part in zip(vis_names, visual_parts):
        ET.SubElement(body, "geom", name=vn, type="mesh", mesh=vn, pos=_fmt(shift), material=vn.replace("_vis_", "_mat_"),
                      contype="0", conaffinity="0", group="2", density="0")
    for pr in proxies:
        # primitives are placed at their center; hull OBJ vertices are already in the asset frame
        pos = list(shift) if pr.type == "mesh" else [a + b for a, b in zip(pr.pos, shift)]
        attrs = {"name": f"{prefix}_{pr.name}", "type": pr.type, "pos": _fmt(pos),
                 "group": "3", "rgba": COLLISION_RGBA, "density": "0"}
        if pr.type == "mesh":
            attrs["mesh"] = f"{prefix}_{pr.name}"
        else:
            attrs["size"] = _fmt(pr.size)
        attrs.update(contact.attrs())
        ET.SubElement(body, "geom", attrs)


def write_fragment(frag: FragmentSpec, out_path: Path) -> Path:
    n = frag.name
    root = ET.Element("mujoco", model=n)
    ET.SubElement(root, "compiler", angle="radian")
    asset = ET.SubElement(root, "asset")
    groups = [(n, frag.visual_parts, frag.proxies)] + [(f"{n}_{c.name}", c.visual_parts, c.proxies)
                                                       for c in frag.children]
    vis_names: dict[str, list[str]] = {}
    for prefix, vparts, proxies in groups:
        vis_names[prefix] = []
        for i, part in enumerate(vparts):
            vn = f"{prefix}_vis_{i}"
            vis_names[prefix].append(vn)
            ET.SubElement(asset, "mesh", name=vn, file=part.file, inertia="shell")
            ET.SubElement(asset, "material", name=vn.replace("_vis_", "_mat_"), rgba=_fmt(part.rgba),
                          specular="0.2", shininess="0.2")
        for pr in proxies:
            if pr.type == "mesh":
                ET.SubElement(asset, "mesh", name=f"{prefix}_{pr.name}", file=pr.file)
    world = ET.SubElement(root, "worldbody")
    # The fragment frame is the asset frame (bottom center). A dynamic body's
    # own frame sits at its center of mass, so freejoint velocities refer to
    # the CoM (e.g. v = w x r rolling for the ball) like the official ball.xml.
    origin = [0.0, 0.0, 0.0] if frag.dynamic is None else list(frag.dynamic.com)
    shift = [-c for c in origin]
    body = ET.SubElement(world, "body", name=n, pos=_fmt(origin))
    if frag.dynamic is not None:
        ET.SubElement(body, "freejoint", name=f"{n}_free")
        _inertial(body, frag.dynamic, (0.0, 0.0, 0.0))
    _geoms(body, n, vis_names[n], frag.visual_parts, frag.proxies, shift, frag.contact)
    for c in frag.children:
        cn = f"{n}_{c.name}"
        cb = ET.SubElement(body, "body", name=cn, pos=_fmt([a - b for a, b in zip(c.pos, origin)]))
        ET.SubElement(cb, "joint", name=f"{cn}_joint", type=c.joint_type, axis=_fmt(c.axis),
                      range=_fmt(c.range), limited="true", damping=f"{c.damping:.6g}",
                      frictionloss=f"{c.frictionloss:.6g}", armature=f"{c.armature:.6g}")
        _inertial(cb, c.dynamic, [a - b for a, b in zip(c.dynamic.com, c.pos)])
        _geoms(cb, cn, vis_names[cn], c.visual_parts, c.proxies, [-x for x in c.pos], frag.contact)
    if frag.children:
        # moving parts sit inside the root's hull-filled cavities: they move on their joints only
        contact = ET.SubElement(root, "contact")
        names = [n] + [f"{n}_{c.name}" for c in frag.children]
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                ET.SubElement(contact, "exclude", body1=names[a], body2=names[b])
    ET.indent(root, space="  ")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(out_path, encoding="unicode")
    return out_path


def body_geoms(body) -> list:
    """Geoms of an MjsBody and all its descendants (a root body with drawers attached)."""
    out = list(body.geoms)
    for child in body.bodies:
        out += body_geoms(child)
    return out


def load_fragment_spec(fragment_xml: Path) -> mujoco.MjSpec:
    """Load a fragment as an MjSpec with absolute mesh paths (safe to attach)."""
    fragment_xml = Path(fragment_xml).resolve()
    spec = mujoco.MjSpec.from_file(str(fragment_xml))
    for mesh in spec.meshes:
        p = Path(mesh.file)
        if not p.is_absolute():
            mesh.file = str(fragment_xml.parent / p)
    spec.meshdir = ""
    return spec
