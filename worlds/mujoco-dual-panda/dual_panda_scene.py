from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent
PANDA_XML = ROOT / "models" / "franka_emika_panda" / "panda.xml"
DUAL_SCENE = ROOT / "models" / "generated_scene.xml"

ARM_POSITIONS = {
    "left": "0 0.45 0",
    "right": "0 -0.45 0",
}

PANDA_HOME_QPOS = ("0", "0", "0", "-1.57079", "0", "1.57079", "-0.7853", "0.04", "0.04")
PANDA_HOME_CTRL = ("0", "0", "0", "-1.57079", "0", "1.57079", "-0.7853", "255")

OBJECTS = (
    # Two blocks reachable by the left arm.
    ("block_left_red", "0.48 0.55 0.04", "0.9961947 0 0 0.0871557", "box", "0.035 0.035 0.035", "0.08", "block_red"),
    ("block_left_blue", "0.34 0.36 0.04", "0.9914449 0 0 0.1305262", "box", "0.035 0.035 0.035", "0.08", "block_blue"),
    # Two blocks reachable by the right arm.
    ("block_right_green", "0.48 -0.55 0.04", "0.976296 0 0 -0.21644", "box", "0.035 0.035 0.035", "0.08", "block_green"),
    ("block_right_yellow", "0.34 -0.36 0.04", "0.9659258 0 0 -0.258819", "box", "0.035 0.035 0.035", "0.08", "block_yellow"),
    # Pillars on the left side of the table.
    ("pillar_left_1", "-0.22 0.55 0.055", "1 0 0 0", "cylinder", "0.028 0.05", "0.10", "pillar_mat"),
    ("pillar_left_2", "-0.36 0.38 0.055", "1 0 0 0", "cylinder", "0.028 0.05", "0.10", "pillar_mat"),
    # Planks on the right side of the table.
    ("plank_right_1", "-0.16 -0.52 0.025", "0.9238795 0 0 0.3826834", "box", "0.16 0.035 0.018", "0.14", "wood"),
    ("plank_right_2", "-0.36 -0.34 0.025", "0.8870108 0 0 -0.4617486", "box", "0.16 0.035 0.018", "0.14", "wood"),
)

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
            "meshdir": "franka_emika_panda",
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

    _add_keyframe(root)
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
    ET.SubElement(asset, "material", {"name": "block_yellow", "rgba": "0.95 0.76 0.10 1"})
    ET.SubElement(asset, "material", {"name": "block_orange", "rgba": "0.95 0.42 0.10 1"})
    ET.SubElement(asset, "material", {"name": "block_purple", "rgba": "0.45 0.20 0.85 1"})
    ET.SubElement(asset, "material", {"name": "block_cyan", "rgba": "0.10 0.75 0.82 1"})
    ET.SubElement(asset, "material", {"name": "block_white", "rgba": "0.86 0.88 0.84 1"})
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
    for name, pos, quat, geom_type, size, mass, material in OBJECTS:
        body = ET.SubElement(worldbody, "body", {"name": name, "pos": pos, "quat": quat})
        ET.SubElement(body, "freejoint", {"name": f"{name}_freejoint"})
        geom = {
            "name": f"{name}_geom",
            "type": geom_type,
            "size": size,
            "mass": mass,
            "material": material,
            "friction": "4.0 1.5 0.08",
            "condim": "6",
        }
        if name.startswith("block_"):
            geom.update({"friction": "5.0 2.0 0.1", "solref": "0.002 1", "solimp": "0.95 0.99 0.001"})
        ET.SubElement(body, "geom", geom)
    ET.SubElement(worldbody, "camera", {"name": "overview", "pos": "1.35 -1.45 0.95", "xyaxes": "0.732 0.681 0 -0.35 0.38 0.86"})


def _add_keyframe(root: ET.Element) -> None:
    object_qpos: list[str] = []
    for _name, pos, quat, *_rest in OBJECTS:
        object_qpos.extend(pos.split())
        object_qpos.extend(quat.split())
    qpos = object_qpos + list(PANDA_HOME_QPOS) + list(PANDA_HOME_QPOS)
    ctrl = list(PANDA_HOME_CTRL) + list(PANDA_HOME_CTRL)
    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(keyframe, "key", {"name": "home", "qpos": " ".join(qpos), "ctrl": " ".join(ctrl)})


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


if __name__ == "__main__":
    print(ensure_dual_panda_scene())
