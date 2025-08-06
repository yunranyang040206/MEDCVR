import pybullet as p
import pybullet_data
import time
import os

class PandaNeedleSimulation:
    def __init__(self, needle_urdf_path):
        # Initialize simulation
        self.p = p
        self.needle_urdf_path = needle_urdf_path
        self.panda_id = None
        self.needle_id = None
        self.constraint_id = None
        self.ee_link_index = None
        
        self.setup_simulation()
    
    def setup_simulation(self):
        # --- 1. Start simulation with enhanced visualization ---
        self.p.connect(self.p.GUI)
        self.p.setGravity(0, 0, -9.81)
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Configure debug visualizer
        self.p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.5]
        )
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 1)
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 1)
        
        # Set physics parameters
        self.p.setPhysicsEngineParameter(
            numSolverIterations=100,
            numSubSteps=10,
            fixedTimeStep=1.0/240.0
        )
        
        # --- 2. Load ground plane ---
        plane_id = self.p.loadURDF("plane.urdf")
        
        # --- 3. Load Franka Panda ---
        start_pos = [0, 0, 0]
        start_ori = self.p.getQuaternionFromEuler([0, 0, 0])
        self.panda_id = self.p.loadURDF(
            "franka_panda/panda.urdf",
            start_pos, start_ori,
            useFixedBase=True
        )
        
        # --- 4. Identify end-effector link ---
        end_effector_link_name = b"panda_hand"
        self.ee_link_index = None
        for i in range(self.p.getNumJoints(self.panda_id)):
            info = self.p.getJointInfo(self.panda_id, i)
            if info[12] == end_effector_link_name:
                self.ee_link_index = i
                break
        
        if self.ee_link_index is None:
            raise ValueError("Could not find panda_hand link")
        
        # --- 5. Get end-effector AABB ---
        aabb_min, aabb_max = self.p.getAABB(self.panda_id, self.ee_link_index)
        size = [aabb_max[i] - aabb_min[i] for i in range(3)]
        parent_mount_pos = [(aabb_max[i] + aabb_min[i]) / 2 for i in range(3)]
        print(f"End-effector dimensions (x, y, z): {size} meters, center: {parent_mount_pos}")
        
        # --- 6. Get initial end-effector pose ---
        link_state = self.p.getLinkState(
            self.panda_id, 
            self.ee_link_index, 
            computeForwardKinematics=True
        )
        print(f"Link state: {link_state}")
        ee_pos, ee_ori = link_state[4], link_state[5]
        print(f"Initial end-effector position: {ee_pos}")
        print(f"Initial orientation (quaternion): {ee_ori}")
        
        # --- 7. Load needle driver ---
        
        temp_id = p.loadURDF(
            self.needle_urdf_path,
            [0, 0, 0],              # world origin
            [0, 0, 0, 1],           # no rotation
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION
        )

        aabb_min, aabb_max = p.getAABB(temp_id, -1)
        child_mount_pos = [(aabb_min[i] + aabb_max[i]) * 0.5 for i in range(3)]
        print("Tool base‑link mount point (local):", child_mount_pos)

        # --- 3) Remove the temporary body ---
        p.removeBody(temp_id)

        world_offset, _ = p.multiplyTransforms(
            [0,0,0],                 # start at TOOL BASE
            ee_ori,                  # rotate by flange orientation
            child_mount_pos,         # vector from tool base → mount point
            [0,0,0,1]
        )
        # Now world_offset is the vector you must subtract from ee_pos
        tool_base_pos = [ee_pos[i] - world_offset[i] for i in range(3)]

        # Load the needle driver at the end-effector position
        y_90_deg_rotation = p.getQuaternionFromEuler([0, 1.5708, 0])

        self.needle_id = p.loadURDF(
            self.needle_urdf_path,
            tool_base_pos,
            y_90_deg_rotation,  # Just the 90-degree rotation
            useFixedBase=False,
            flags=self.p.URDF_USE_SELF_COLLISION
        )

        
        # --- 8. Create fixed constraint between EE and needle ---

        self.constraint_id = p.createConstraint(
            parentBodyUniqueId    = self.panda_id,
            parentLinkIndex       = self.ee_link_index,
            childBodyUniqueId     = self.needle_id,
            childLinkIndex        = -1,
            jointType             = p.JOINT_FIXED,
            jointAxis             = [0, 0, 0],
            parentFramePosition   = [0,0,0],
            childFramePosition    = [0,0,0],
            parentFrameOrientation= [0,0,0,1],
            childFrameOrientation = [0,0,0,1]
        )

                
        # --- 9. Set initial joint positions ---
    #     self.set_joint_positions([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    
    # def set_joint_positions(self, positions):
    #     """Set joint positions for the Panda arm"""
    #     joint_indices = [i for i in range(self.p.getNumJoints(self.panda_id)) 
    #                    if self.p.getJointInfo(self.panda_id, i)[2] == self.p.JOINT_REVOLUTE]
        
    #     for i, idx in enumerate(joint_indices[:7]):
    #         self.p.resetJointState(self.panda_id, idx, positions[i])
    #         self.p.setJointMotorControl2(
    #             self.panda_id, 
    #             idx, 
    #             self.p.POSITION_CONTROL, 
    #             targetPosition=positions[i],
    #             force=500
    #         )
    
    def run(self):
        """Main simulation loop"""
        print("Simulation running. Press Ctrl+C in the console to quit.")
        try:
            while True:
                self.p.stepSimulation()
                time.sleep(1.0/240.0)
        except KeyboardInterrupt:
            print("Simulation stopped by user")
        finally:
            self.p.disconnect()

if __name__ == "__main__":
    # Path to your needle driver URDF
    needle_urdf_path = r"C:\EngSci\MEDCVR_2025\Needle_Driver\simulation_test\Main\Main.urdf"
    
    # Create and run simulation
    sim = PandaNeedleSimulation(needle_urdf_path)
    sim.run()