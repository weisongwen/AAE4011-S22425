import open3d as o3d
import numpy as np

def generate_random_point_cloud():
    """Generate a random point cloud."""
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.random.rand(10000, 3))
    return pc

def apply_random_transformation(pc):
    """Apply a random transformation to a point cloud."""
    R = pc.get_rotation_matrix_from_xyz((np.random.rand() * 0.2, np.random.rand() * 0.2, np.random.rand() * 0.2))
    t = np.random.rand(3) * 1.5 # 0.2
    transformation = np.eye(4)
    transformation[:3, :3] = R
    transformation[:3, 3] = t
    return pc.transform(transformation)

def draw_map(map_cloud):
    """Visualize the accumulated map."""
    o3d.visualization.draw_geometries([map_cloud], window_name="Accumulated Map")

# Initialize an empty point cloud for the map
map_cloud = o3d.geometry.PointCloud()

# Generate and accumulate 10 frames of point clouds
num_frames = 10
threshold = 0.1
trans_init = np.eye(4)

for i in range(num_frames):
    # Generate a new random point cloud and apply a random transformation
    source = generate_random_point_cloud()
    source = apply_random_transformation(source)
    
    # Color the source point cloud differently for each frame
    color = np.random.rand(3)
    source.paint_uniform_color(color)
    
    if i == 0:
        # Initialize the map with the first point cloud
        map_cloud += source
    else:
        # Perform ICP to align the source to the map
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, map_cloud, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        
        # Transform the source point cloud using the ICP result
        source.transform(reg_p2p.transformation)
        
        # Accumulate the transformed source into the map
        map_cloud += source

# Visualize the accumulated ma

draw_map(map_cloud)
