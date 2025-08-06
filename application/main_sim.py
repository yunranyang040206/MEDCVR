import swift
import numpy as np
from spatialmath import SE3
import spatialgeometry as sg
import roboticstoolbox as rtb
from pathlib import Path
from helpers import setup_3arm_system, check_reach, plot_trajectory, plot_errors

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
    
    env.step(dt)

print("Movement 1 complete")
print("Current needle arm pose:\n", needle_arm.fkine(needle_arm.q))
needle_arm.qd = np.zeros_like(needle_arm.qd) # Stop the needle arm from drifting

"""Movement 2: Regular arm movement"""
T_link_world_current = needle_tool.fkine(needle_tool.q, end='tool_roll_link-v2')
T_reg_des = T_reg_des = SE3(T_link_world_current) * SE3.Tz(0.25) 
start_marker = sg.Sphere(radius=0.01, base=regular_arm.fkine(regular_arm.q))    # default color
target_marker = sg.Sphere(radius=0.01, base=T_reg_des)

check_reach(regular_arm, T_reg_des)

arrived_reg = False
traj_actual = []
traj_target = []

print("T_reg_des:\n", T_reg_des)
print("T_cur_start:\n", regular_arm.fkine(regular_arm.q))
print("arrived_reg at start:", arrived_reg)
env.add(start_marker)  
env.add(target_marker) 

# servo the regular arm in a second loop with timeout
max_iters = 250
iters = 0

while not arrived_reg:
    T_cur = regular_arm.fkine(regular_arm.q)
    traj_actual.append(T_cur.t.copy())       
    traj_target.append(T_reg_des.t.copy())  

    v_reg, arrived_reg = rtb.p_servo(T_cur, T_reg_des, 0.2)
    regular_arm.qd = np.linalg.pinv(regular_arm.jacobe(regular_arm.q)) @ v_reg
    error = T_cur.t - T_reg_des.t
    if np.linalg.norm(error) < 0.01:   # 1 cm tolerance
        arrived_reg = True
        regular_arm.qd[:] = 0
        break

    env.step(dt)
    iters += 1
    if iters >= max_iters:
        print(f"Aborting: reached {max_iters} iterations without arrival.")
        break


traj_actual = np.array(traj_actual)   # shape (N,3)
traj_target = np.array(traj_target)
plot_trajectory(traj_actual, traj_target)
plot_errors(traj_actual, traj_target)

print("Movement 2 complete" if arrived_reg else "Movement 2 aborted")
env.hold()