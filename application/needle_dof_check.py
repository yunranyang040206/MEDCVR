#!/usr/bin/env python3

from pathlib import Path
import xml.etree.ElementTree as ET

import swift
from roboticstoolbox import Robot
from spatialmath import SE3
import numpy as np
import time

# === Configuration ===
URDF_PATH    = "/home/adalyn/MEDCVR_2025/Needle_Driver/Main/Main.urdf"
dt           = 0.05            # simulation timestep (s)
hinge_names  = ["gripper_left-v2", "gripper_right-v2"]
marker_size  = 0.05            # size of hinge‐frame markers (meters)

# === Load the robot ===
needle = Robot.URDF(URDF_PATH)

# === Build a child->parent map from the URDF ===
tree = ET.parse(URDF_PATH)
root = tree.getroot()
parent_map = {}
for joint in root.findall("joint"):
    c = joint.find("child").attrib["link"]
    p = joint.find("parent").attrib["link"]
    parent_map[c] = p

# === Collect joint info via link.jindex ===
joint_info = []
for link in needle.links:
    if link.jindex is not None and link.jindex >= 0:
        qmin, qmax = link.qlim if link.qlim is not None else (-np.pi, np.pi)
        joint_info.append((link.jindex, link.name, qmin, qmax))

# === Launch Swift and add the robot ===
env = swift.Swift()
env.launch(realtime=True)
env.add(needle)

# === Place hinge‐frame markers for each gripper ===
for i, hinge in enumerate(hinge_names):
    if hinge not in parent_map:
        print(f"Warning: '{hinge}' not found in URDF joints.")
        continue

    parent = parent_map[hinge]
    # Compute transforms: base→parent and base→child
    T_base_parent = needle.fkine(needle.q, parent)
    T_base_child  = needle.fkine(needle.q, hinge)
    # Compute parent→child
    T_hinge = T_base_parent.inv() @ T_base_child

    # Build and add the axis triad marker
    marker = SE3(T_hinge.t) * SE3.RPY(*T_hinge.rpy(order="xyz"))
    marker_name = f"hinge_marker_{i}"
    env.add(marker, marker_name, marker_size)
    env.step(dt)

print("Hinge markers placed. Inspect them in Swift to read off xyz/rpy for your URDF <origin> tags.")

# === Sweep through all joints to verify ===
for q_idx, name, qmin, qmax in joint_info:
    print(f"Sweeping joint q[{q_idx}] '{name}' …")
    for q_val in np.linspace(qmin, qmax, 60):
        needle.q[q_idx] = q_val
        env.step(dt)
    time.sleep(0.3)

env.hold()
