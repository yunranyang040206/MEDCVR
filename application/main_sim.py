import swift
import numpy as np
from spatialmath import SE3
import spatialgeometry as sg
import roboticstoolbox as rtb
from pathlib import Path
import matplotlib.pyplot as plt
from helpers import setup_3arm_system, check_reach, plot_trajectory, plot_errors, move_arm_ik
import helpers

# Configuration
URDF_PATH = "/home/adalyn/MEDCVR_2025/Needle_Driver/Main/Main.urdf"
MESHES_DIR = "/home/adalyn/MEDCVR_2025/Needle_Driver/Main/meshes"
INCORRECT_PREFIX = "/home/adalyn/.local/lib/python3.8/site-packages/rtbdata/xacro/meshes/"

# Setup 3-arm system
env, needle_arm, needle_tool, regular_arm, camera_arm, needle_offset = setup_3arm_system(
    URDF_PATH, MESHES_DIR, INCORRECT_PREFIX
)

# Define target ARM pose (not tool pose)
initial_arm_T = needle_arm.fkine(needle_arm.q)
R0 = initial_arm_T.R
Tarm_des = SE3.Rt(R0, [0.1, 0, 0.5]) * SE3.Rz(np.pi/2)
check_reach(needle_arm, Tarm_des)

'''Movement 1: Move needle arm to target pose'''
dt = 0.05     
arrived = False

while not arrived:
    current_arm_T = needle_arm.fkine(needle_arm.q)
    v, arrived = rtb.p_servo(current_arm_T, Tarm_des, 1)
    
    needle_arm.qd = np.linalg.pinv(needle_arm.jacobe(needle_arm.q)) @ v
    needle_arm.q = needle_arm.q + needle_arm.qd * dt
    
    # Update tool position based on new arm position
    needle_tool.base = needle_arm.fkine(needle_arm.q) * needle_offset

    midpoint = helpers.camera_tracking(camera_arm, needle_arm, regular_arm)
    
    env.step(dt)

print("Movement 1 complete")
print("Current needle arm pose:\n", needle_arm.fkine(needle_arm.q))
needle_arm.qd = np.zeros_like(needle_arm.qd) # Stop the needle arm from drifting

"""Movement 2: Regular arm movement"""
T_link_world_current = needle_tool.fkine(needle_tool.q, end='tool_roll_link-v2')
T_world_target = SE3(T_link_world_current) * SE3.Tz(0.25)

# start_marker = sg.Sphere(radius=0.01, base=regular_arm.fkine(regular_arm.q))
# target_marker = sg.Sphere(radius=0.01, base=T_world_target)
# env.add(start_marker)
# env.add(target_marker)

check_reach(regular_arm, T_world_target)


print("=== Movement 2 ===")
print("T_world_target:\n", T_world_target)
print("T_cur_start:\n", regular_arm.fkine(regular_arm.q))

# traj_actual, traj_target, success = move_arm_ik(
#     env, 
#     regular_arm, 
#     T_world_target,
#     q0=regular_arm.q,
#     tol=1e-6,
#     mask=[1,1,1,0,0,0],
#     N=100,
#     dt=0.05,
#     error_thresh=0.01
# )
traj_actual, traj_target, success = helpers.move_arm_ik1(
    env, 
    regular_arm, 
    T_world_target,
    camera_arm=camera_arm,
    other_arms=[needle_arm],
    q0=regular_arm.q,
    tol=1e-6,
    mask=[1,1,1,0,0,0],
    N=100,
    dt=0.05,
    error_thresh=0.01
)

# if success:
#     plot_trajectory(traj_actual, traj_target)
#     plot_errors(traj_actual, traj_target)

print("Movement 2 complete" if success else "Movement 2 failed")

"""Movement 3: Coordinated movement of both arms holding needle driver"""
# Define target pose for needle arm (relative to its current position)
current_needle_pose = needle_arm.fkine(needle_arm.q)
T_needle_target = current_needle_pose * SE3.Ty(-0.1)* SE3.Tz(0.1) * SE3.Rz(np.pi/2)*SE3.Ry(-np.pi/4)
check_reach(needle_arm, T_needle_target)
check_reach(regular_arm, T_needle_target)

# Create markers for visualization
start_marker_needle = sg.Sphere(radius=0.01, base=current_needle_pose, color=(0,1,0))
target_marker_needle = sg.Sphere(radius=0.01, base=T_needle_target, color=(1,0,0))
initial_tool_pos = needle_tool.fkine(needle_tool.q, end='tool_roll_link-v2') * SE3.Tz(0.25)

env.add(start_marker_needle)
env.add(target_marker_needle)


print("=== Movement 3 ===")
print("Needle arm target pose:\n", T_needle_target)
print("Regular arm current pose:\n", regular_arm.fkine(regular_arm.q))

# Setup trajectory recording
traj_actual_needle = []
traj_actual_reg = []
traj_target_needle = []

# Movement parameters
N = 150
dt = 0.05
error_thresh = 0.005

for i in range(1, N+1):
    α = i / N
    needle_arm.q = needle_arm.q + α * (needle_arm.ikine_LM(needle_arm.base.inv() @ T_needle_target, q0=needle_arm.q).q - needle_arm.q)
    needle_tool.base = needle_arm.fkine(needle_arm.q) * needle_offset
    
    # Get current tool_roll_link pose 
    T_tool_link = needle_tool.fkine(needle_tool.q, end='tool_roll_link-v2')
    T_reg_target = T_tool_link * SE3.Tz(0.25) 
    
    # Move regular arm to follow the offset position
    regular_arm.q = regular_arm.ikine_LM(regular_arm.base.inv() @ T_reg_target, q0=regular_arm.q).q
    
    # Record trajectories
    traj_actual_needle.append(needle_arm.fkine(needle_arm.q).t)
    traj_actual_reg.append(regular_arm.fkine(regular_arm.q).t)
    traj_target_needle.append(T_needle_target.t)
    
    midpoint = helpers.camera_tracking(camera_arm, needle_arm, regular_arm)
    env.step(dt)
    
    # Early termination check
    if np.linalg.norm(needle_arm.fkine(needle_arm.q).t - T_needle_target.t) < error_thresh:
        print(f"Reached target in {i} steps")
        break

# Convert trajectories to numpy arrays
traj_actual_needle = np.array(traj_actual_needle)
traj_actual_reg = np.array(traj_actual_reg)
traj_target_needle = np.array(traj_target_needle)

# plot_trajectory(traj_actual_needle, traj_target_needle)

# # Plot errors for both arms
# print("\nNeedle Arm Errors:")
# plot_errors(traj_actual_needle, traj_target_needle)

# Additional combined 3D plot showing both arms
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(traj_actual_needle[:,0], traj_actual_needle[:,1], traj_actual_needle[:,2], 
        'g-', label='Needle Arm Path', linewidth=2)
ax.plot(traj_actual_reg[:,0], traj_actual_reg[:,1], traj_actual_reg[:,2],
        'b-', label='Regular Arm Path', linewidth=2)
ax.scatter(traj_target_needle[0,0], traj_target_needle[0,1], traj_target_needle[0,2],
           c='r', marker='o', s=100, label='Target')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Dual Arm 3D Trajectories')
ax.legend()
plt.tight_layout()
plt.show()

print("Movement 3 complete")

env.hold()