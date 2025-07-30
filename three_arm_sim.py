import pybullet as p
import pybullet_data
import time
import numpy as np
from needle_driver import PandaArm, PandaWithNeedle

needle_arm = None
regular_arm = None
camera_arm = None

def setup_simulation():
    global needle_arm, regular_arm, camera_arm
    # Initialize simulation
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Configure debug visualizer
    p.resetDebugVisualizerCamera(
        cameraDistance=2.5,
        cameraYaw=30,  # Better viewing angle
        cameraPitch=-40,
        cameraTargetPosition=[0, 0, 0.5]
    )
    
    # Set physics parameters
    p.setPhysicsEngineParameter(
        numSolverIterations=100,
        numSubSteps=10,
        fixedTimeStep=1.0/240.0
    )
    
    # Load ground plane
    p.loadURDF("plane.urdf")

    # Path to needle driver URDF
    needle_urdf_path = r"C:\Users\Lenovo\OneDrive - University of Toronto\2025-Research\Needle_Driver\simulation_test\Main\Main.urdf"
    
    # Define positions (radius = 0.8m) properly centered around origin
    radius = 0.6
    arm_positions = [
        [radius * np.cos(5*np.pi/6), radius * np.sin(5*np.pi/6), 0],  # Left position 
        [radius * np.cos(np.pi/6), radius * np.sin(np.pi/6), 0],      # Right position
        [0, -radius, 0]  # Front position 
    ]
    
    arm_orientations = [
        [0, 0, np.pi/2],  
        [0, 0, -np.pi/2],  
        [0, 0, np.pi]  ]

    # Create platform for front arm (0.3m high)
    platform_height = 0.3
    platform_pos = [0, -radius, platform_height/2]  # Center of platform
    platform_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, platform_height/2])
    platform_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, platform_height/2], rgbaColor=[0.8, 0.8, 0.8, 1])
    platform = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=platform_shape, 
                               baseVisualShapeIndex=platform_visual, basePosition=platform_pos)

    # Adjust front arm position to be on top of platform
    arm_positions[2][2] = platform_height

    needle_arm = PandaWithNeedle(needle_urdf_path, position=arm_positions[0], orientation=arm_orientations[0])
    regular_arm = PandaArm(position=arm_positions[1], orientation=arm_orientations[1])
    camera_arm = PandaArm(position=arm_positions[2], orientation=arm_orientations[2])

def lock_arm_at_current_pose(robot, force=200, gain=0.1):
    """
    Reads every joint angle of `robot` and commands it back there,
    so gravity can’t make it sag.
    """
    robot_id = robot.panda_id
    for j in range(p.getNumJoints(robot_id)):
        state = p.getJointState(robot_id, j)
        q = state[0] # Joint angle 
        p.setJointMotorControl2(
            bodyIndex=robot_id,
            jointIndex=j,
            controlMode=p.POSITION_CONTROL,
            targetPosition=q,
            positionGain=gain,
            force=force
        )


def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else v

def quat_from_two_vectors(v_from, v_to):
    v0 = normalize(v_from)
    v1 = normalize(v_to)
    dot = np.dot(v0, v1)
    if dot < -0.999999:
        ortho = np.array([1,0,0]) if abs(v0[0]) < 0.9 else np.array([0,1,0])
        axis = normalize(np.cross(v0, ortho))
        angle = np.pi
    else:
        axis = normalize(np.cross(v0, v1))
        angle = np.arccos(np.clip(dot, -1.0, 1.0))
    qw = np.cos(angle/2.0)
    qxyz = axis * np.sin(angle/2.0)
    return [qxyz[0], qxyz[1], qxyz[2], qw]

def camera_tracking(camera_arm, arm1, arm2, follow_distance=0.5):
    """
    Moves & orients camera_arm so it stays `follow_distance` behind
    the midpoint of arm1 & arm2, always looking at the midpoint.
    """

    # 1. Get the two tool tips and compute their midpoint
    pos1, _ = p.getLinkState(arm1.panda_id, arm1.ee_link_index)[:2]
    pos2, _ = p.getLinkState(arm2.panda_id, arm2.ee_link_index)[:2]
    pos1, pos2 = np.array(pos1), np.array(pos2)
    midpoint = (pos1 + pos2) / 2.0

    # 2. Compute current direction from camera → midpoint
    cam_pos, _ = p.getLinkState(camera_arm.panda_id, camera_arm.ee_link_index)[:2]
    cam_pos = np.array(cam_pos)
    target_dir = normalize(midpoint - cam_pos)

    # 3. Compute the new, desired camera position:
    #    sit `follow_distance` meters BACK along that direction
    desired_cam_pos = midpoint - (target_dir * follow_distance)

    # 4. Build quaternion so +X points at the midpoint
    camera_forward = np.array([1.0, 0.0, 0.0])
    q_align = quat_from_two_vectors(camera_forward, target_dir)

    # 5. Call IK with both targetPosition & targetOrientation:
    joint_angles = p.calculateInverseKinematics(
        camera_arm.panda_id,
        camera_arm.ee_link_index,
        targetPosition=desired_cam_pos.tolist(),
        targetOrientation=q_align
    )

    # 6. Apply the solution to each joint
    for j, q in enumerate(joint_angles):
        p.setJointMotorControl2(
            camera_arm.panda_id,
            j,
            p.POSITION_CONTROL,
            targetPosition=q
        )

def run():
    global needle_arm, regular_arm, camera_arm
    print("Three-arm simulation running. Press Ctrl+C to stop.")
    try:
        while True:
            p.stepSimulation()
            lock_arm_at_current_pose(regular_arm)
            lock_arm_at_current_pose(needle_arm)
            lock_arm_at_current_pose(camera_arm)
            camera_tracking(camera_arm, regular_arm, needle_arm)
            time.sleep(1.0/240.0)
    except KeyboardInterrupt:
        print("Simulation stopped by user")
    finally:
        p.disconnect()

if __name__ == "__main__":
    setup_simulation()
    run()