import xml.etree.ElementTree as ET

from bench.tasks.button_asset import BUTTON_FIXED_PARTS, add_button_asset


def test_answer_button_matches_tm01_geometry() -> None:
    root = ET.Element("worldbody")
    add_button_asset(
        root,
        geom_name="answer",
        pos=(1.0, 2.0, 3.0),
        color_rgba="1 0 0 1",
        glow_rgba="1 0.2 0.2 1",
        gravity_compensation=True,
    )
    body = root.find("body")
    assert body is not None and body.get("pos") == "1.000000 2.000000 3.000000"
    fixed = {geom.get("name"): geom for geom in body.findall("geom")}
    assert set(fixed) == {f"answer_{suffix}" for suffix in BUTTON_FIXED_PARTS}
    assert fixed["answer_base"].get("size") == "0.060 0.060 0.020"
    assert fixed["answer_glow_ring_outer"].get("size") == "0.033 0.003"
    plunger = body.find("body")
    assert plunger is not None
    assert plunger.get("gravcomp") == "1"
    inertial = plunger.find("inertial")
    joint = plunger.find("joint")
    geoms = {geom.get("name"): geom for geom in plunger.findall("geom")}
    assert inertial is not None and inertial.get("mass") == "0.020"
    assert joint is not None
    assert joint.get("range") == "-0.012 0"
    assert joint.get("stiffness") == "18"
    assert joint.get("damping") == "2"
    assert geoms["answer"].get("size") == "0.029 0.008"
