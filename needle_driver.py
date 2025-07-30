import pybullet as p
import numpy as np

class PandaArm:
    """Basic Franka Panda arm without any attachments"""
    def __init__(self, position=[0, 0, 0], orientation=[0, 0, 0]):
        self.p = p
        self.panda_id = None
        self.ee_link_index = None
        self.setup_arm(position, orientation)

    def setup_arm(self, position, orientation):
        """Setup basic Franka Panda arm"""
        start_ori = self.p.getQuaternionFromEuler(orientation)
        self.panda_id = self.p.loadURDF(
            "franka_panda/panda.urdf",
            position, start_ori,
            useFixedBase=True
        )
        
        # Identify end-effector link
        end_effector_link_name = b"panda_hand"
        for i in range(self.p.getNumJoints(self.panda_id)):
            info = self.p.getJointInfo(self.panda_id, i)
            if info[12] == end_effector_link_name:
                self.ee_link_index = i
                break
        
        if self.ee_link_index is None:
            raise ValueError("Could not find panda_hand link")

class PandaWithNeedle(PandaArm):
    """Franka Panda arm with needle driver attachment"""
    def __init__(self, needle_urdf_path, position=[0, 0, 0], orientation=[0, 0, 0]):
        super().__init__(position, orientation)
        self.needle_id = None
        self.constraint_id = None
        self.attach_needle(needle_urdf_path)

    def attach_needle(self, needle_urdf_path):
        """Attach needle driver to the end effector"""
        # Get end effector pose
        link_state = self.p.getLinkState(
            self.panda_id, 
            self.ee_link_index, 
            computeForwardKinematics=True
        )
        ee_pos, ee_ori = link_state[4], link_state[5]

        # Load needle URDF
        fusion_vertex_mm = [-11.211, -3.941, 6.12]
        child_mount_pos = [
            fusion_vertex_mm[0] / 1000,
            -fusion_vertex_mm[2] / 1000, 
            fusion_vertex_mm[1] / 1000  
        ]
        
        y_90_deg_rotation = p.getQuaternionFromEuler([np.pi, np.pi, -1.5708])
        rotated_offset = p.rotateVector(y_90_deg_rotation, child_mount_pos)

        tool_base_pos, tool_world_ori = p.multiplyTransforms(
            ee_pos, ee_ori, 
            rotated_offset, y_90_deg_rotation
        )
       
        self.needle_id = p.loadURDF(
            needle_urdf_path,
            tool_base_pos,
            tool_world_ori,
            useFixedBase=False,
            flags=self.p.URDF_USE_SELF_COLLISION
        )

        # Create fixed constraint
        self.constraint_id = p.createConstraint(
            parentBodyUniqueId=self.panda_id,
            parentLinkIndex=self.ee_link_index,
            childBodyUniqueId=self.needle_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=[0, 0, 0, 1],
            childFrameOrientation=[0, 0, 0, 1]
        )