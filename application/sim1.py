import roboticstoolbox as rtb
import spatialmath as sm
import spatialgeometry as sg
import swift
from math import pi
import numpy as np
import time

# This is our callback funciton from the sliders in Swift which set
# the joint angles of robot1 to the value of the sliders
def set_joint(j, value):
    robot1.q[j] = np.deg2rad(float(value)) #update robot1 joint value
    robot1_frame.T = robot1.fkine(robot1.q) #update frame pose

    #update cylinder pose and frame in Swift
    cylinder.T = robot1.fkine(robot1.q) * sm.SE3.Ry(pi/2) * sm.SE3().Tz(0.10)
    cylinder_rob2_frame.T=cylinder.T * sm.SE3().Tz(0.15) * sm.SE3().Ry(-pi/2) * sm.SE3().Rz(pi)

    # run IK on robot2
    Tdes = robot2.base.inv() * cylinder_rob2_frame.T # T_r0 * T_0c -> Robot2 base frame to cylinder frame
    q_new = robot2.ikine_LM(Tdes, q0=robot2.q)
    print("Tdes: ", Tdes)
    if q_new.success:
        robot2.q = q_new.q

# Launch the simulator Swift
env = swift.Swift()
env.launch()

# Create two Panda robots
robot1 = rtb.models.Panda()
robot2 = rtb.models.Panda()

# Set the initial pose of the robots
robot1.q = robot1.qr
robot2.q = robot2.qr

# Set the base of the robots
robot1.base = sm.SE3(0.6, 0, 0) * sm.SE3.Rz(pi)
robot2.base = sm.SE3(-0.5, -0.1, 0) 

# Open the grippers
robot1.grippers[0].q = [0.01, 0.01]
robot2.grippers[0].q = [0.01, 0.01]

# Add the robots to the visualizer
env.add(robot1)
env.add(robot2)

# Visualize a frame on robot1
robot1_frame = sg.Axes(0.1, pose=robot1.fkine(robot1.q))
env.add(robot1_frame)

# Create a cylinder object
cylinder = sg.Cylinder(radius=0.01, length=0.3,color='red')
cylinder.T = robot1.fkine(robot1.q) * sm.SE3.Ry(pi/2) * sm.SE3().Tz(0.10)
env.add(cylinder)

cylinder_rob2_desPose = cylinder.T * sm.SE3().Tz(0.15)
cylinder_rob2_frame = sg.Axes(0.1, pose=cylinder_rob2_desPose)
env.add(cylinder_rob2_frame)

# Loop through each link in the Panda and if it is a variable joint,
# add a slider to Swift to control it
j = 0
for link in robot1.links:
    if link.isjoint:

        # We use a lambda as the callback function from Swift
        # j=j is used to set the value of j rather than the variable j
        # We use the HTML unicode format for the degree sign in the unit arg
        env.add(
            swift.Slider(
                lambda x, j=j: set_joint(j, x),
                min=np.round(np.rad2deg(link.qlim[0]), 2),
                max=np.round(np.rad2deg(link.qlim[1]), 2),
                step=1,
                value=np.round(np.rad2deg(robot1.q[j]), 2),
                desc="Panda 1: Joint " + str(j+1),
                unit="&#176;",
            )
        )

        j += 1


while True:
    # Process the event queue from Swift, this invokes the callback functions
    # from the sliders if the slider value was changed
    # env.process_events()

    # Update the environment with the new robot pose
    env.step(0)

    time.sleep(0.01)