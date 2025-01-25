import open3d as o3d
import numpy as np

def create_coordinate_frame(size=0.1, origin=[0, 0, 0]):
    """Create a coordinate frame for visualization."""
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)

def apply_transformation(geometry, transformation):
    """Apply a homogeneous transformation to a geometry."""
    return geometry.transform(transformation)

def visualize_poses(poses):
    """Visualize multiple poses with coordinate frames."""
    geometries = []
    for pose in poses:
        # Create a coordinate frame for each pose
        frame = create_coordinate_frame()
        # Apply the transformation to the coordinate frame
        transformed_frame = apply_transformation(frame, pose)
        geometries.append(transformed_frame)
    
    # Visualize all coordinate frames
    o3d.visualization.draw_geometries(geometries, window_name="Homogeneous Transformation Visualization")

# Define four different homogeneous transformation matrices
poses = []

# Pose 1: Identity (no transformation)
pose1 = np.eye(4)
poses.append(pose1)

# Pose 2: Translation along x-axis
pose2 = np.eye(4)
pose2[:3, 3] = [0.5, 0, 0]
poses.append(pose2)

# Pose 3: Rotation around z-axis
angle = np.pi / 4  # 45 degrees
pose3 = np.eye(4)
pose3[:3, :3] = [
    [np.cos(angle), -np.sin(angle), 0],
    [np.sin(angle), np.cos(angle), 0],
    [0, 0, 1]
]
poses.append(pose3)

# Pose 4: Translation and rotation
pose4 = np.eye(4)
pose4[:3, :3] = [
    [np.cos(angle), -np.sin(angle), 0],
    [np.sin(angle), np.cos(angle), 0],
    [0, 0, 1]
]
pose4[:3, 3] = [0.5, 0.5, 0]
poses.append(pose4)

# Visualize the poses
visualize_poses(poses)