from pathlib import Path
import roboticstoolbox as rtb
from roboticstoolbox.robot import Robot
from roboticstoolbox.models import Panda
import numpy as np
from spatialmath import SE3
import spatialgeometry as sg
import matplotlib.pyplot as plt
import swift
import pybullet

def setup_3arm_system(urdf_path, meshes_dir, incorrect_prefix, radius=0.6, cube_size=[0.4, 0.4, 0.3], base_z_offset=0.05):
    # Legacy Mesh Symlink Hack
    legacy_dir = Path(incorrect_prefix.rstrip("/"))
    if not legacy_dir.exists():
        legacy_dir.parent.mkdir(parents=True, exist_ok=True)
        legacy_dir.symlink_to(Path(meshes_dir))
        print(f"Created symlink: {legacy_dir} -> {meshes_dir}")

    # Load Robot Models
    needle_tool = Robot.URDF(urdf_path)
    needle_arm = Panda()       
    regular_arm = Panda()   
    camera_arm = Panda()   

    # Compute Static Base Transforms
    # 1) Needle arm at left position, yaw = +90°
    pos1 = SE3(radius * np.cos(5*np.pi/6), radius * np.sin(5*np.pi/6), base_z_offset)
    orn1 = SE3.Rz(np.pi/2)
    needle_arm.base = pos1 * orn1

    # Attach needle tool with its offset
    needle_offset = SE3.Ry(-np.pi/2)
    needle_tool.base = needle_arm.fkine(needle_arm.q) * needle_offset

    # 2) Regular arm at right position, yaw = -90°
    pos2 = SE3(radius * np.cos(np.pi/6), radius * np.sin(np.pi/6), base_z_offset)
    orn2 = SE3.Rz(-np.pi/2)
    regular_arm.base = pos2 * orn2

    # 3) Camera arm at front on platform, yaw = 180°
    cube_z = cube_size[2]
    cube_pose = SE3(0, -radius, base_z_offset + cube_z/2)
    camera_arm.base = SE3(0, -radius, base_z_offset + cube_z) * SE3.Rz(np.pi)

    # Create environment and add models
    env = swift.Swift()
    env.launch(realtime=True)
    cube = sg.Box(scale=cube_size, pose=cube_pose)
    env.add(cube)
    env.add(needle_arm)
    needle_arm.q = needle_arm.qr
    env.add(needle_tool)
    env.add(regular_arm)
    regular_arm.q = regular_arm.qr  
    env.add(camera_arm)
    camera_arm.q = camera_arm.qr  

    return env, needle_arm, needle_tool, regular_arm, camera_arm, needle_offset


def check_reach(arm, T_des, tol=1e-6):
    """
    Returns (reachable: bool, dist: float, max_reach: float)
    using a fast spherical‐bound test.
    """
    # 1) build list of link "lengths" from each link's zero‐angle offset
    link_lengths = []
    for link in arm.links:
        A0 = link.A(0)
        # if A0 is BasePoseList or list of transforms, grab the first
        if hasattr(A0, "__len__") and not isinstance(A0, np.ndarray):
            A0 = A0[0]
        # now A0 is an SE3 object; take its translation
        tvec = np.array(A0.t).flatten()   # [x,y,z]
        link_lengths.append(np.linalg.norm(tvec))

    max_reach = sum(link_lengths)

    # 2) distance from base to target
    target_pos = np.array(T_des.t).flatten()
    dist       = np.linalg.norm(target_pos)

    reachable = (dist <= max_reach + tol)
    if not reachable:
        print(f"→ Unreachable: target at {dist:.3f} m, exceeds max ≃{max_reach:.3f} m. Skipping movement 2.")
    else:
        print(f"→ Reachable: target at {dist:.3f} m, within max ≃{max_reach:.3f} m.")

    return reachable, dist, max_reach

def move_arm_ik(env, arm, T_world_target, q0=None, tol=1e-6, mask=[1,1,1,0,0,0], N=100, dt=0.05, error_thresh=0.005):
    """
    Move arm to target pose using IK and joint interpolation.
    """
    if q0 is None:
        q0 = arm.q.copy()
    
    # Convert world target to arm's base frame for IK
    T_arm_target = arm.base.inv() @ T_world_target
    
    # Solve IK
    ik_sol = arm.ikine_LM(T_arm_target, q0=q0, tol=tol, mask=mask)
    if not ik_sol.success:
        print("IK failed to converge!")
        return None, None, False
    
    # Prepare for interpolation
    q_start = arm.q.copy()
    q_target = ik_sol.q
    traj_actual = []
    traj_target = []
    
    # Execute trajectory
    for i in range(1, N+1):
        α = i / N
        arm.q = q_start + α*(q_target - q_start)
        
        env.step(dt)
        
        T_now = arm.fkine(arm.q)
        traj_actual.append(T_now.t)
        traj_target.append(T_world_target.t)
        
        if np.linalg.norm(T_now.t - T_world_target.t) < error_thresh:
            print(f"Reached target in {i} steps, error {np.linalg.norm(T_now.t - T_world_target.t):.4f} m")
            break
    else:
        err = np.linalg.norm(T_now.t - T_world_target.t)
        print(f"Completed {N} steps with final error {err:.4f} m")
    
    return np.array(traj_actual), np.array(traj_target), True


def move_arm_ik1(env, arm, T_world_target, camera_arm=None, other_arms=[], q0=None, tol=1e-6, mask=[1,1,1,0,0,0], N=100, dt=0.05, error_thresh=0.005):
    """
    Move arm to target pose using IK and joint interpolation.
    Returns: (traj_actual, traj_target, success)
    """
    if q0 is None:
        q0 = arm.q.copy()
    
    # Initialize empty arrays for failure case
    empty_traj = np.zeros((0, 3))
    
    # Convert world target to arm's base frame for IK
    T_arm_target = arm.base.inv() @ T_world_target
    
    # Solve IK
    try:
        ik_sol = arm.ikine_LM(T_arm_target, q0=q0, tol=tol, mask=mask)
        if not ik_sol.success:
            print("IK failed to converge!")
            return empty_traj, empty_traj, False
        
        # Prepare for interpolation
        q_start = arm.q.copy()
        q_target = ik_sol.q
        traj_actual = []
        traj_target = []
        
        # Execute trajectory
        for i in range(1, N+1):
            α = i / N
            arm.q = q_start + α*(q_target - q_start)
            
            # Update camera if provided
            if camera_arm and other_arms:
                tracking_arms = [arm] + other_arms
                if len(tracking_arms) >= 2:
                    camera_tracking(camera_arm, tracking_arms[0], tracking_arms[1])
            
            env.step(dt)
            
            T_now = arm.fkine(arm.q)
            traj_actual.append(T_now.t)
            traj_target.append(T_world_target.t)
            
            if np.linalg.norm(T_now.t - T_world_target.t) < error_thresh:
                print(f"Reached target in {i} steps, error {np.linalg.norm(T_now.t - T_world_target.t):.4f} m")
                break
        
        return np.array(traj_actual), np.array(traj_target), True
    
    except Exception as e:
        print(f"Movement failed: {str(e)}")
        return empty_traj, empty_traj, False

def plot_trajectory(traj_actual, traj_target):
    # --- 3D Trajectory Plot ---
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(traj_actual[:,0], traj_actual[:,1], traj_actual[:,2], label='actual')
    ax.scatter(traj_target[0,0], traj_target[0,1], traj_target[0,2],
               color='r', label='target', s=50)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend()
    plt.show()

    # --- Per-axis vs. iteration ---
    steps = np.arange(len(traj_actual))
    plt.figure(figsize=(5,8))

    plt.subplot(3,1,1)
    plt.plot(steps, traj_actual[:,0], label='x_act')
    plt.hlines(traj_target[0,0], xmin=0, xmax=steps[-1], colors='r', linestyles='--')
    plt.ylabel('X (m)'); plt.legend()

    plt.subplot(3,1,2)
    plt.plot(steps, traj_actual[:,1], label='y_act')
    plt.hlines(traj_target[0,1], xmin=0, xmax=steps[-1], colors='r', linestyles='--')
    plt.ylabel('Y (m)'); plt.legend()

    plt.subplot(3,1,3)
    plt.plot(steps, traj_actual[:,2], label='z_act')
    plt.hlines(traj_target[0,2], xmin=0, xmax=steps[-1], colors='r', linestyles='--')
    plt.ylabel('Z (m)'); plt.xlabel('Iteration'); plt.legend()

    plt.tight_layout()
    plt.show()

def plot_errors(traj_actual, traj_target):
    errors = traj_actual - traj_target       # per-axis
    errors_norm = np.linalg.norm(errors, axis=1)

    plt.figure()
    plt.plot(errors_norm, label='‖actual – target‖')
    plt.xlabel('Iteration'); plt.ylabel('Error norm (m)')
    plt.title('Tracking Error over Time')
    plt.legend()
    plt.show()

    print(f"Final error norm: {errors_norm[-1]:.4f} m")
    print(f"Mean error norm:  {errors_norm.mean():.4f} m")
    print(f"Max error norm:   {errors_norm.max():.4f} m")


def setup_hybrid_environment():
    """
    Set up hybrid environment with Swift visualization and PyBullet collision checking
    Returns:
        env_swift: Swift visualization environment
        env_pyb: PyBullet collision environment
    """
    # Swift for visualization
    env_swift = swift()
    env_swift.launch(realtime=True)
    
    # PyBullet for collision checking (headless mode)
    env_pyb = pybullet()
    env_pyb.launch(headless=True)  # Run in background
    
    return env_swift, env_pyb

def setup_collision_objects(env_pyb, robot_models, obstacles):
    """
    Add collision objects to PyBullet environment
    Args:
        env_pyb: PyBullet environment
        robot_models: List of robot models
        obstacles: List of obstacle geometries
    """
    for robot in robot_models:
        env_pyb.add(robot, collision=True)  # Enable collision geometry
        
    for obstacle in obstacles:
        env_pyb.add(obstacle, collision=True)


def camera_tracking(camera_arm, arm1, arm2, follow_distance=0.5):
    """
    Moves & orients camera_arm so it stays `follow_distance` behind
    the midpoint of arm1 & arm2, always looking at the midpoint.
    
    Args:
        camera_arm: The camera arm robot
        arm1: First arm to track
        arm2: Second arm to track
        follow_distance: Distance to maintain behind midpoint (meters)
    """
    # Get the two tool tips and compute their midpoint
    pos1 = arm1.fkine(arm1.q).t
    pos2 = arm2.fkine(arm2.q).t
    midpoint = (pos1 + pos2) / 2.0
    
    # Compute current direction from camera → midpoint
    cam_pos = camera_arm.fkine(camera_arm.q).t
    target_dir = (midpoint - cam_pos) / np.linalg.norm(midpoint - cam_pos)
    
    # Compute desired camera position (follow_distance behind midpoint)
    desired_cam_pos = midpoint - (target_dir * follow_distance)
    
    # Compute desired orientation (camera looking at midpoint)
    # Alternative method without LookAt:
    # 1. Compute the direction vector from camera to midpoint
    look_dir = normalize(midpoint - desired_cam_pos)
    
    # 2. Create a rotation matrix that aligns the camera's x-axis with this direction
    # (Assuming camera's forward direction is +x)
    z_axis = np.array([0, 0, 1])  # World up axis
    y_axis = normalize(np.cross(z_axis, look_dir))
    new_z_axis = normalize(np.cross(look_dir, y_axis))
    
    R = np.column_stack((look_dir, y_axis, new_z_axis))
    desired_orientation = SE3.Rt(R, desired_cam_pos)
    
    # Convert to camera arm's base frame
    T_cam_target = camera_arm.base.inv() @ desired_orientation
    
    # Solve IK
    sol = camera_arm.ikine_LM(T_cam_target, q0=camera_arm.q)
    if sol.success:
        camera_arm.q = sol.q
    
    return midpoint

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else v