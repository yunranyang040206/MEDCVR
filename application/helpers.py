from pathlib import Path
import roboticstoolbox as rtb
from roboticstoolbox.robot import Robot
from roboticstoolbox.models import Panda
import numpy as np
from spatialmath import SE3
import spatialgeometry as sg
import matplotlib.pyplot as plt
import swift

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