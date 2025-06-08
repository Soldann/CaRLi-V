import supervisely as sly
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import List
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def visualize_points(point_cloud):
    # Extract X, Y, Z coordinates
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]

    # Create 3D scatter plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=z, cmap='viridis', marker='o', s=5)

    # Labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Point Cloud Visualization")

    # Show the plot
    plt.show()

def transform_points(points, rotation_matrix, translation_vector):
    """Applies rotation and translation to the point cloud."""
    return (rotation_matrix @ points.T).T + translation_vector

def apply_rotation(points, rotation_vector):
    """Convert rotation vector to matrix and apply rotation."""
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    return np.dot(rotation_matrix, points.T).T

def get_bounding_box_corners(position, dimensions):
    """Compute 8 corners of the cuboid before rotation."""
    dx, dy, dz = dimensions / 2  # Half-lengths
    corners = np.array([
        [dx, dy, dz], [dx, -dy, dz], [-dx, -dy, dz], [-dx, dy, dz],  # Top
        [dx, dy, -dz], [dx, -dy, -dz], [-dx, -dy, -dz], [-dx, dy, -dz]  # Bottom
    ])
    return corners + position  # Translate to object position

def visualize_pointcloud_with_bbox(point_cloud, position, dimensions, rotation_vector):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot all points
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c='blue', marker='o', s=1)

    # Get bounding box corners & apply rotation
    bbox_corners = get_bounding_box_corners(position, dimensions)
    bbox_corners = apply_rotation(bbox_corners - position, rotation_vector) + position

    # Define box edges
    edges = [
        [bbox_corners[i] for i in [0, 1, 2, 3, 0]],  # Top face
        [bbox_corners[i] for i in [4, 5, 6, 7, 4]],  # Bottom face
        [bbox_corners[i] for i in [0, 4]], [bbox_corners[i] for i in [1, 5]],
        [bbox_corners[i] for i in [2, 6]], [bbox_corners[i] for i in [3, 7]]
    ]

    # Draw bounding box
    ax.add_collection3d(Poly3DCollection(edges, edgecolor='red', linewidths=2, alpha=0.3))

    # Labels
    ax.set_xlabel("X"), ax.set_ylabel("Y"), ax.set_zlabel("Z")
    ax.set_title("3D Point Cloud with Bounding Box")

    plt.show()

def extract_points_inside_bbox(point_cloud, position, dimensions, rotation_vector):
    """Extracts points within the rotated bounding box."""
    # Move points to object-centered frame
    centered_points = point_cloud - position

    # Apply inverse rotation to align with object frame
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    transformed_points = np.dot(rotation_matrix.T, centered_points.T).T

    # Check which points are within the bounding box (axis-aligned in transformed space)
    half_dims = dimensions / 2
    within_x = (-half_dims[0] <= transformed_points[:, 0]) & (transformed_points[:, 0] <= half_dims[0])
    within_y = (-half_dims[1] <= transformed_points[:, 1]) & (transformed_points[:, 1] <= half_dims[1])
    within_z = (-half_dims[2] <= transformed_points[:, 2]) & (transformed_points[:, 2] <= half_dims[2])

    # Extract points inside bounding box
    filtered_points = point_cloud[within_x & within_y & within_z]

    # Transform points back to original frame
    filtered_original_points = apply_rotation(filtered_points, rotation_vector) + position

    return filtered_original_points

class Object():
    def __init__(self, name, points, centroid, timestamp):
        self.name = name # If object is person or reflector
        self.points = points # points inside object
        self.centroid = centroid # object centroid
        self.timestamp = timestamp # object timestamp


### LOAD THE DATASET

# Path to the downloaded annotation JSON file
annotation_path = "datasets/scene_1/annotation.json"

# Path to the mapping file
mapping_file = "datasets/scene_1/frame_pointcloud_map.json"

# Load the mapping
with open(mapping_file, "r") as f:
    frame_map = json.load(f)

# Load project metadata (assuming you have it downloaded as well)
project_meta_json = "datasets/meta.json"
project_meta = sly.ProjectMeta.from_json(sly.json.load_json_file(project_meta_json))

# Load annotation from JSON file
ann = sly.PointcloudEpisodeAnnotation.load_json_file(annotation_path, project_meta)

reflectors : List[Object] = []
persons : List[Object] = []

# RETRIEVE POINTCLOUDS FOR EACH OBJECT
for key in frame_map:
    frame_index = int(key)
    print(f"Processing index {frame_index}")
    frame_data = ann.frames.get(frame_index)  # Retrieve frame details
    objects_on_frame = ann.get_objects_on_frame(frame_index)

    pointcloud_filename = frame_map.get(key)

    figures_on_frame = ann.get_figures_on_frame(frame_index)
    point_cloud_points = sly.pointcloud.read("datasets/scene_1/pointcloud/" + pointcloud_filename) # Shape (Nx3)
    point_cloud_data = point_cloud_points

    timestamp = float(pointcloud_filename.removesuffix(".pcd"))

    if len(figures_on_frame) != 2:
        print("WTFFFFFFF")

    # Extract points associated with each object
    for figure in figures_on_frame:
        object_geometry = figure.geometry
        position = object_geometry.position
        dimensions = object_geometry.dimensions

        rotation = object_geometry.rotation  # Extract rotation vector

        # Convert to np arrays because it's easier
        rotation_vector = np.array([rotation.x, rotation.y, rotation.z]).astype(np.float32)
        dimension_vec = np.array([dimensions.x, dimensions.y, dimensions.z]).astype(np.float32)
        position_vec = np.array([position.x, position.y, position.z]).astype(np.float32)

        object_points = extract_points_inside_bbox(point_cloud_data, position_vec, dimension_vec, rotation_vector)
        # visualize_points(object_points)

        object_centroid = np.mean(object_points, axis=0)

        object = Object(figure.parent_object.obj_class.name, object_points, object_centroid, timestamp)
        if object.name == "Person":
            persons.append(object)
        elif object.name == "Reflector":
            reflectors.append(object)

        print(f"Object {figure.parent_object.obj_class.name} has {len(object_points)} points, with centroid {object_centroid}")

for i in range(len(reflectors) - 1):
    reflector_velocity = (reflectors[i + 1].centroid - reflectors[i].centroid) / (reflectors[i + 1].timestamp - reflectors[i].timestamp)
    person_velocity = (persons[i + 1].centroid - persons[i].centroid) / (persons[i + 1].timestamp - persons[i].timestamp)
    print(f"Velocity at frame {i}: Reflector: {reflector_velocity}, Person: {person_velocity}, time difference: {reflectors[i + 1].timestamp - reflectors[i].timestamp}")
