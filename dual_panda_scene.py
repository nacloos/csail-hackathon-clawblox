from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent
PANDA_XML = ROOT / "models" / "franka_emika_panda" / "panda.xml"
DUAL_SCENE = ROOT / "models" / "panda_dual" / "generated_scene.xml"

ARM_POSITIONS = {
    "left": "0 0.45 0",
    "right": "0 -0.45 0",
}

REFERENCE_ATTRS = {
    "body1",
    "body2",
    "joint",
    "joint1",
    "joint2",
    "tendon",
}


def ensure_dual_panda_scene(path: Path = DUAL_SCENE) -> Path:
    xml = build_dual_panda_scene_xml()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text(encoding="utf-8") != xml:
        path.write_text(xml, encoding="utf-8")
    return path


def build_dual_panda_scene_xml() -> str:
    panda = ET.parse(PANDA_XML).getroot()
    root = ET.Element("mujoco", {"model": "dual panda cube scene"})
    ET.SubElement(
        root,
        "compiler",
        {
            "angle": "radian",
            "meshdir": "../franka_emika_panda",
            "autolimits": "true",
        },
    )
    ET.SubElement(root, "option", {"cone": "elliptic", "impratio": "20", "iterations": "100", "tolerance": "1e-10"})

    default = panda.find("default")
    if default is not None:
        root.append(deepcopy(default))

    asset = deepcopy(panda.find("asset"))
    if asset is not None:
        for mesh in asset.findall("mesh"):
            file_name = mesh.get("file")
            if file_name and not file_name.startswith("assets/"):
                mesh.set("file", f"assets/{file_name}")
        _add_scene_assets(asset)
        root.append(asset)

    statistic = ET.SubElement(root, "statistic", {"center": "0.45 0 0.25", "extent": "1.6"})
    statistic.tail = "\n\n  "
    _add_visual(root)
    worldbody = ET.SubElement(root, "worldbody")
    _add_world(worldbody)

    panda_worldbody = panda.find("worldbody")
    if panda_worldbody is None:
        raise ValueError(f"missing worldbody in {PANDA_XML}")
    base_body = panda_worldbody.find("body")
    if base_body is None:
        raise ValueError(f"missing Panda base body in {PANDA_XML}")

    for arm_name, position in ARM_POSITIONS.items():
        body = _prefixed(deepcopy(base_body), f"{arm_name}_")
        body.set("pos", position)
        worldbody.append(body)

    for section_name in ("tendon", "equality", "actuator", "contact"):
        section = panda.find(section_name)
        if section is None:
            continue
        merged = ET.SubElement(root, section_name)
        for arm_name in ARM_POSITIONS:
            prefix = f"{arm_name}_"
            for child in section:
                merged.append(_prefixed(deepcopy(child), prefix))

    _indent(root)
    return ET.tostring(root, encoding="unicode") + "\n"


def _prefixed(node: ET.Element, prefix: str) -> ET.Element:
    name = node.get("name")
    if name:
        node.set("name", prefix + name)
    for attr in REFERENCE_ATTRS:
        value = node.get(attr)
        if value:
            node.set(attr, prefix + value)
    for child in node:
        _prefixed(child, prefix)
    return node


def _add_scene_assets(asset: ET.Element) -> None:
    ET.SubElement(asset, "texture", {"type": "skybox", "builtin": "gradient", "rgb1": "0.35 0.45 0.55", "rgb2": "0.02 0.03 0.04", "width": "512", "height": "3072"})
    ET.SubElement(asset, "texture", {"type": "2d", "name": "ground_grid", "builtin": "checker", "mark": "edge", "rgb1": "0.25 0.28 0.30", "rgb2": "0.16 0.18 0.20", "markrgb": "0.8 0.8 0.8", "width": "300", "height": "300"})
    ET.SubElement(asset, "material", {"name": "ground_grid", "texture": "ground_grid", "texuniform": "true", "texrepeat": "5 5", "reflectance": "0.02", "specular": "0", "shininess": "0"})
    ET.SubElement(asset, "material", {"name": "block_red", "rgba": "0.95 0.18 0.08 1"})
    ET.SubElement(asset, "material", {"name": "block_blue", "rgba": "0.10 0.36 0.95 1"})
    ET.SubElement(asset, "material", {"name": "block_green", "rgba": "0.12 0.70 0.30 1"})
    ET.SubElement(asset, "material", {"name": "wood", "rgba": "0.58 0.36 0.18 1"})
    ET.SubElement(asset, "material", {"name": "pillar_mat", "rgba": "0.70 0.72 0.70 1"})


def _add_visual(root: ET.Element) -> None:
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "headlight", {"diffuse": "0.6 0.6 0.6", "ambient": "0.3 0.3 0.3", "specular": "0 0 0"})
    ET.SubElement(visual, "rgba", {"haze": "0.12 0.16 0.20 1"})
    ET.SubElement(visual, "global", {"azimuth": "135", "elevation": "-25"})


def _add_world(worldbody: ET.Element) -> None:
    ET.SubElement(worldbody, "light", {"name": "key", "pos": "0.2 -0.8 1.6", "dir": "-0.2 0.7 -1", "directional": "true"})
    ET.SubElement(worldbody, "light", {"name": "fill", "pos": "-0.6 0.4 1.2", "dir": "0.5 -0.2 -1", "directional": "true", "diffuse": "0.25 0.25 0.25"})
    ET.SubElement(worldbody, "geom", {"name": "floor", "type": "plane", "size": "2 2 0.05", "material": "ground_grid", "friction": "2.0 0.5 0.02"})
    for name, pos, material in (
        ("block_red", "0.55 0 0.04", "block_red"),
        ("block_blue", "0.38 0.22 0.04", "block_blue"),
        ("block_green", "0.38 -0.22 0.04", "block_green"),
    ):
        body = ET.SubElement(worldbody, "body", {"name": name, "pos": pos})
        ET.SubElement(body, "freejoint", {"name": f"{name}_freejoint"})
        ET.SubElement(body, "geom", {"name": f"{name}_geom", "type": "box", "size": "0.035 0.035 0.035", "mass": "0.08", "material": material, "friction": "5.0 2.0 0.1", "condim": "6", "solref": "0.002 1", "solimp": "0.95 0.99 0.001"})
    ET.SubElement(worldbody, "camera", {"name": "overview", "pos": "1.25 -1.20 0.85", "xyaxes": "0.674 0.739 0 -0.36 0.33 0.87"})


def _indent(element: ET.Element, level: int = 0) -> None:
    indent = "\n" + level * "  "
    child_indent = "\n" + (level + 1) * "  "
    if len(element):
        if not element.text or not element.text.strip():
            element.text = child_indent
        for child in element:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    if level and (not element.tail or not element.tail.strip()):
        element.tail = indent
