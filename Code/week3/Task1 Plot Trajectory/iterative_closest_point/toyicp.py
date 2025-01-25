import open3d as o3d
import numpy as np

def draw_registration_result(source, target, transformation, iteration):
    # Apply the transformation to the source point cloud
    source_temp = source.transform(transformation)
    
    # Set colors for visualization
    source_temp.paint_uniform_color([1, 0, 0])  # Red for source
    target.paint_uniform_color([0, 1, 0])       # Green for target
    
    # Visualize the point clouds
    o3d.visualization.draw_geometries([source_temp, target], window_name=f"ICP Iteration {iteration}")

# Create a target point cloud
target = o3d.geometry.PointCloud()
target.points = o3d.utility.Vector3dVector(np.random.rand(10000, 3))

# Create a source point cloud by applying a known transformation to the target
source = o3d.geometry.PointCloud()
source.points = o3d.utility.Vector3dVector(np.random.rand(10000, 3))

# Apply a known transformation to simulate an initial misalignment
R = source.get_rotation_matrix_from_xyz((0.1, 0.1, 0.1))  # Rotation
t = np.array([0.5, -0.2, 0.3])  # Translation
transformation = np.eye(4)
transformation[:3, :3] = R
transformation[:3, 3] = t
source.transform(transformation)

# Initial transformation
trans_init = np.eye(4)

# Threshold for ICP
threshold = 0.1

# Number of iterations
max_iterations = 10

# Perform ICP registration manually to visualize each iteration
current_transformation = trans_init
for i in range(max_iterations):
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, current_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1)
    )
    
    # Update the transformation
    current_transformation = reg_p2p.transformation @ current_transformation
    
    print("current transformation matrix:")
    print(current_transformation)
    
    # Visualize the result of the current iteration
    draw_registration_result(source, target, current_transformation, i + 1)

# Final transformation
print("Final transformation matrix:")
print(current_transformation)