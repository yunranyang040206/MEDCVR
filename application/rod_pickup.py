import roboticstoolbox as rtb
import spatialmath as sm
import spatialgeometry as sg
import swift
import numpy as np
from math import pi
import time

# Parameters
rod_length = 0.6      
rod_radius = 0.02
# Define world poses for rod center and drop hole
rod_center = sm.SE3(0.0, 0.3, 0.1) * sm.SE3.Rx(pi/2)
hole_center = sm.SE3(0.0, -0.3, 0.1) * sm.SE3.Rx(pi/2)
# Frames at rod ends
g1_init = rod_center * sm.SE3().Tz( rod_length/2 )
g2_init = rod_center * sm.SE3().Tz(-rod_length/2 )
# Lift offset
lift_offset = sm.SE3().Tz(0.1)
# Drop frames at hole ends
g1_drop = hole_center * sm.SE3().Tz( rod_length/2 )
g2_drop = hole_center * sm.SE3().Tz(-rod_length/2 )

# Startup Swift visualization
env = swift.Swift()
env.launch()

# Create two Panda robots
global robot1, robot2
robot1 = rtb.models.Panda()
robot2 = rtb.models.Panda()
# Save home poses
g1_home = robot1.qr.copy()
g2_home = robot2.qr.copy()
# Initialize joint states
robot1.q = g1_home.copy()
robot2.q = g2_home.copy()
# Position robot bases
robot1.base = sm.SE3(0.5, 0.3, 0) * sm.SE3.Rz(pi)
robot2.base = sm.SE3(-0.5, 0.3, 0)
# Open grippers
robot1.grippers[0].q = [0.01, 0.01]
robot2.grippers[0].q = [0.01, 0.01]
# Add robots and objects to scene
env.add(robot1)
env.add(robot2)
# Render rod and hole marker
rod = sg.Cylinder(radius=rod_radius, length=rod_length, color='red')
rod.T = rod_center
env.add(rod)
hole = sg.Box([0.2, 0.2, 0.02], color='blue')
hole.T = hole_center * sm.SE3().Tz(0.1)
env.add(hole)

# Pre-compute IK solutions
g1_pick = robot1.ikine_LM(robot1.base.inv() * g1_init, q0=g1_home).q
q2_pick = robot2.ikine_LM(robot2.base.inv() * g2_init, q0=g2_home).q
q1_lift = robot1.ikine_LM(robot1.base.inv() * (g1_init * lift_offset), q0=g1_pick).q
q2_lift = robot2.ikine_LM(robot2.base.inv() * (g2_init * lift_offset), q0=q2_pick).q
q1_drop = robot1.ikine_LM(robot1.base.inv() * g1_drop, q0=q1_lift).q
q2_drop = robot2.ikine_LM(robot2.base.inv() * g2_drop, q0=q2_lift).q

# Trajectory segments: home→pick, pick→lift, lift→drop
N = 100
traj1_home2pick = rtb.jtraj(g1_home, g1_pick, N)
traj2_home2pick = rtb.jtraj(g2_home, g2_home, N)  # robot2 stays
traj1_pick2lift = rtb.jtraj(g1_pick, q1_lift, N)
traj2_pick2lift = rtb.jtraj(q2_pick, q2_lift, N)
traj1_lift2drop = rtb.jtraj(q1_lift, q1_drop, N)
traj2_lift2drop = rtb.jtraj(q2_lift, q2_drop, N)

# Phases for rod attachment logic
PHASE_PRE = 1           # rod static at initial center
PHASE_GRASP1 = 2        # robot1 holds rod
PHASE_BOTH = 3          # both robots hold rod
current_phase = PHASE_PRE

# Helper: compute rod pose based on phase
initial_rod_T = rod_center.copy()
def update_rod(q1, q2):
    global current_phase
    if current_phase == PHASE_PRE:
        return initial_rod_T
    elif current_phase == PHASE_GRASP1:
        # attach rod to robot1's gripper: back half-length along local Z
        return robot1.fkine(q1) * sm.SE3().Tz(-rod_length/2)
    else:
        # both holding: midpoint of two gripper positions, maintain orientation
        p1 = robot1.fkine(q1).t
        p2 = robot2.fkine(q2).t
        center = (p1 + p2) / 2
        return sm.SE3(center[0], center[1], center[2]) * sm.SE3.Rx(pi/2)

# Animation helper
def animate_phase(qs1, qs2, phase_after=None, dt=0.05):
    global current_phase
    for q1, q2 in zip(qs1, qs2):
        robot1.q, robot2.q = q1, q2
        rod.T = update_rod(q1, q2)
        env.step(dt)
        time.sleep(dt)
    if phase_after is not None:
        current_phase = phase_after

# Sequence execution
# 1) Robot1 picks (robot2 waits)
animate_phase(traj1_home2pick.q, traj2_home2pick.q, phase_after=PHASE_GRASP1)
# simulate gripper close
time.sleep(0.5)
# 2) Robot2 picks (robot1 holds)
animate_phase([g1_pick]*N, rtb.jtraj(g2_home, q2_pick, N).q, phase_after=PHASE_BOTH)
# simulate gripper close
time.sleep(0.5)
# 3) Lift together
animate_phase(traj1_pick2lift.q, traj2_pick2lift.q)
# 4) Move to drop
animate_phase(traj1_lift2drop.q, traj2_lift2drop.q)
# simulate gripper open
time.sleep(0.5)

print("Two-arm cooperative pick-and-place complete.")
env.hold()  # keep window open
