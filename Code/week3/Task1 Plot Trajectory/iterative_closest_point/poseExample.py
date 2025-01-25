import numpy as np

# Position in 3D space
position = np.array([1.0, 2.0, 3.0])  # x, y, z coordinates
print("Position:", position)

from scipy.spatial.transform import Rotation as R

# Euler angles (in radians)
euler_angles = np.array([np.pi / 4, np.pi / 6, np.pi / 3])  # Roll, Pitch, Yaw
rotation = R.from_euler('xyz', euler_angles)
print("Euler Angles (radians):", euler_angles)
print("Rotation Matrix from Euler Angles:\n", rotation.as_matrix())

# Define a rotation matrix directly
rotation_matrix = np.array([
    [0.866, -0.5, 0.0],
    [0.5, 0.866, 0.0],
    [0.0, 0.0, 1.0]
])
print("Rotation Matrix:\n", rotation_matrix)

# Quaternion representation
quaternion = rotation.as_quat()  # Convert from Euler angles
print("Quaternion:", quaternion)

# Homogeneous transformation matrix
transformation_matrix = np.eye(4)
transformation_matrix[:3, :3] = rotation_matrix  # Rotation part
transformation_matrix[:3, 3] = position  # Translation part
print("Homogeneous Transformation Matrix:\n", transformation_matrix)