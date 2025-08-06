import pybullet as p
import os
import time
import pybullet_data

# Connect to physics server
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)

# Load plane
planeId = p.loadURDF("plane.urdf")

# 1. SPECIFY THE CORRECT PATH - WINDOWS RAW STRING FORMAT
urdf_path = r"C:\EngSci\MEDCVR_2025\Needle_Driver\simulation_test\Main\Main.urdf"  # Raw string to handle backslashes

# Verify the path exists
if not os.path.exists(urdf_path):
    raise FileNotFoundError(f"URDF not found at:\n{urdf_path}\n"
                          f"Current working directory: {os.getcwd()}")

# 2. LOAD ROBOT WITH PROPER SETTINGS
robotStartPos = [0, 0, 1.0]  # Elevated to prevent collision
robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
robotID = p.loadURDF(urdf_path,
                    robotStartPos,
                    robotStartOrientation,
                    useFixedBase=True,  # Prevent falling until joints are configured
                    flags=p.URDF_USE_INERTIA_FROM_FILE)

# 3. SETUP BETTER VIEW
p.resetDebugVisualizerCamera(
    cameraDistance=1.5,
    cameraYaw=45,
    cameraPitch=-30,
    cameraTargetPosition=robotStartPos)

# 4. DEBUGGING OUTPUT
print(f"Successfully loaded URDF from:\n{urdf_path}")
print(f"Robot ID: {robotID}")
print("All links:", p.getNumJoints(robotID))

# 5. KEEP WINDOW OPEN
print("Simulation running... Press Ctrl+C to exit")
try:
    while True:
        p.stepSimulation()
        time.sleep(1./240.)
except KeyboardInterrupt:
    print("Simulation stopped by user")
finally:
    p.disconnect()