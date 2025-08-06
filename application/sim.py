#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import swift
import roboticstoolbox as rtb
from roboticstoolbox.models import Panda
from spatialmath import SE3
import spatialgeometry as sg

# Helper to check reachability
def check_reach(arm, T_des, tol=1e-6):
    lengths = []
    for link in arm.links:
        A0 = link.A(0)
        t = np.array(A0.t).flatten()
        lengths.append(np.linalg.norm(t))
    max_reach = sum(lengths)
    dist = np.linalg.norm(np.array(T_des.t).flatten())
    ok = dist <= max_reach + tol
    msg = f"→ {'Reachable' if ok else 'Unreachable'}: target at {dist:.3f} m, max ≃{max_reach:.3f} m."
    print(msg)
    return ok

# Paths
URDF_PATH  = "/home/adalyn/MEDCVR_2025/Needle_Driver/Main/Main.urdf"
MESH_DIR   = "/home/adalyn/MEDCVR_2025/Needle_Driver/Main/meshes"
# create legacy symlink if needed
INCORRECT = Path("/home/adalyn/.local/lib/python3.8/site-packages/rtbdata/xacro/meshes")
if not INCORRECT.exists():
    INCORRECT.parent.mkdir(parents=True, exist_ok=True)
    INCORRECT.symlink_to(Path(MESH_DIR))

# Load models
needle_tool  = rtb.Robot.URDF(URDF_PATH)
needle_arm   = rtb.models.Panda()
regular_arm  = rtb.models.Panda()
camera_arm   = rtb.models.Panda()

# Mount bases
R = 0.6; Z0 = 0.05
needle_arm.base  = SE3(R*np.cos(5*np.pi/6), R*np.sin(5*np.pi/6), Z0) * SE3.Rz( np.pi/2)
needle_offset    = SE3.Ry(-np.pi/2)
needle_tool.base = needle_arm.fkine(needle_arm.q) * needle_offset

regular_arm.base = SE3(R*np.cos( np.pi/6), R*np.sin( np.pi/6), Z0) * SE3.Rz(-np.pi/2)
camera_arm.base  = SE3(0, -R, Z0 + 0.3/2) * SE3.Rz(np.pi)

# Launch Swift without real-time throttling
env = swift.Swift()
env.launch(realtime=False)

# Add scene objects
env.add(sg.Box(scale=[0.4,0.4,0.3], pose=SE3(0,-R,Z0+0.15)))
env.add(needle_arm);   needle_arm.q = needle_arm.qr
env.add(needle_tool)
env.add(regular_arm);  regular_arm.q = regular_arm.qr
env.add(camera_arm);   camera_arm.q = camera_arm.qr

# --- Movement 1: Cartesian P-servo on needle_arm ---
R0 = needle_arm.fkine(needle_arm.q).R
Tarm_des = SE3.Rt(R0, [0.1,0,0.5]) * SE3.Rz(np.pi/2)
check_reach(needle_arm, Tarm_des)

dt = 0.05
arrived = False
while not arrived:
    T_cur = needle_arm.fkine(needle_arm.q)
    v, arrived = rtb.p_servo(T_cur, Tarm_des, 1)
    needle_arm.qd = np.linalg.pinv(needle_arm.jacobe(needle_arm.q)) @ v
    needle_arm.q  = needle_arm.q + needle_arm.qd * dt
    # keep tool mounted
    needle_tool.base = needle_arm.fkine(needle_arm.q) * needle_offset
    env.step(dt)

print("Movement 1 complete")
needle_arm.qd = np.zeros_like(needle_arm.qd)




T_link_world   = needle_tool.fkine(needle_tool.q, end='tool_roll_link-v2')
T_world_target = SE3(T_link_world) * SE3.Tz(0.25)  
# Convert that world goal into the regular arm’s base frame for IK
T_reg_des = regular_arm.base.inv() @ T_world_target  


start_marker  = sg.Sphere(radius=0.01, base=regular_arm.fkine(regular_arm.q))
target_marker = sg.Sphere(radius=0.01, base=T_world_target)
env.add(start_marker)
env.add(target_marker)

# Solve IK *on the base-frame target*
ik_sol = regular_arm.ikine_LM(T_reg_des, q0=regular_arm.q, tol=1e-6, mask=[1,1,1,0,0,0])
if not ik_sol.success:
    print("IK failed to converge!")
    env.hold()
    exit()

# Interpolate joints and record *world* trajectory
q_start     = regular_arm.q.copy()
q_target    = ik_sol.q
traj_actual = []
traj_target = []

N   = 100  
dt2 = 0.05
for i in range(1, N+1):
    α = i / N
    regular_arm.q = q_start + α*(q_target - q_start)
    env.step(dt2)

    T_now = regular_arm.fkine(regular_arm.q)
    traj_actual.append(T_now.t)
    traj_target.append(T_world_target.t) 

    if np.linalg.norm(T_now.t - T_world_target.t) < 0.005:
        print(f"Reached target in {i} steps, error {np.linalg.norm(T_now.t - T_world_target.t):.4f} m")
        break
else:
    err = np.linalg.norm(T_now.t - T_world_target.t)
    print(f"Completed {N} steps with final error {err:.4f} m")

traj_actual = np.array(traj_actual)
traj_target = np.array(traj_target)

# 7) Plotting
from mpl_toolkits.mplot3d import Axes3D

# 3D path
fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')
ax.plot(traj_actual[:,0], traj_actual[:,1], traj_actual[:,2], label='actual')
ax.scatter(*traj_target[0], color='r', label='target', s=50)
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.legend()
plt.show()

# Per-axis vs step
steps = np.arange(len(traj_actual))
plt.figure(figsize=(5,8))
plt.subplot(3,1,1)
plt.plot(steps, traj_actual[:,0], label='x_act')
plt.hlines(traj_target[0,0], 0, steps[-1], 'r', '--'); plt.ylabel('X (m)'); plt.legend()
plt.subplot(3,1,2)
plt.plot(steps, traj_actual[:,1], label='y_act')
plt.hlines(traj_target[0,1], 0, steps[-1], 'r', '--'); plt.ylabel('Y (m)'); plt.legend()
plt.subplot(3,1,3)
plt.plot(steps, traj_actual[:,2], label='z_act')
plt.hlines(traj_target[0,2], 0, steps[-1], 'r', '--'); plt.ylabel('Z (m)'); plt.xlabel('Step'); plt.legend()
plt.tight_layout()
plt.show()

print("Movement2 complete")

# === Movement 3: Cooperatively lift the held tool ===

# how far (along flange Z) to lift in each step
lift = 0.2   # 0.2 m
dt3  = 0.05  # 20 Hz update
gain3 = 0.8  # a bit gentler than movement 1

arrived3 = False
while not arrived3:
    # 1) current flange pose
    Tf_cur = needle_arm.fkine(needle_arm.q)

    # 2) desired flange pose: lift up along the flange's Z
    Tf_des = Tf_cur * SE3.Tz(lift)

    # 3) Cartesian servo on needle arm
    v3, arrived3 = rtb.p_servo(Tf_cur, Tf_des, gain3)
    needle_arm.qd = np.linalg.pinv(needle_arm.jacobe(needle_arm.q)) @ v3
    needle_arm.q  = needle_arm.q + needle_arm.qd * dt3

    # 4) re-mount the tool
    needle_tool.base = needle_arm.fkine(needle_arm.q) * needle_offset

    # 5) compute the new world pose of the roll link
    Tlink_w = needle_tool.fkine(needle_tool.q, end='tool_roll_link-v2')

    # 6) IK for the regular arm to track that link
    #    (express target in regular_arm’s base frame)
    Treg_des3 = regular_arm.base.inv() @ Tlink_w
    ik3 = regular_arm.ikine_LM(
        Treg_des3,
        q0=regular_arm.q,
        tol=1e-6,
        mask=[1,1,1,0,0,0]
    )
    if ik3.success:
        regular_arm.q = ik3.q

    # 7) step the sim for both arms
    env.step(dt3)

print("Movement 3 complete")
env.hold()
