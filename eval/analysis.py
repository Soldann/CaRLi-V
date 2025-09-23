import supervisely as sly
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import List, Dict
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import open3d as o3d
from pathlib import Path
import pickle
import argparse

SCRIPT_PATH = Path(__file__).parent
DATASET_PATH = SCRIPT_PATH.parent / 'datasets'

def decompose_v(velocity, position):
    '''
    Decompose a velocity vector into radial and tangential components
    :param velocity: np.array of shape (3,) representing the velocity vector of a point
    :param position: np.array of shape (3,) representing the position vector of a point
    :return radial_v, tangent_v: (np.array, np.array) of shape (3,) representing the radial and tangential components of the velocity at point position relative to the origin
    '''
    radial_v = (np.dot(velocity, position) * position / np.dot(position, position))
    tangent_v = velocity - radial_v
    
    return radial_v, tangent_v

def visualize_points(point_cloud, title="Point Cloud Visualization"):
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
    ax.set_title(title)

    # Show the plot
    plt.show()

def transform_points(points, rotation_matrix, translation_vector):
    """Applies rotation and translation to the point cloud. Preserves additional fields"""
    translated_points = (rotation_matrix @ points[:, :3].T).T + translation_vector
    return np.column_stack((translated_points, points[:, 3:]))

def apply_rotation(points, rotation_vector):
    """Convert rotation vector to matrix and apply rotation. Preserves additional fields"""
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    rotated_points =  np.dot(rotation_matrix, points[:, :3].T).T
    return np.column_stack((rotated_points, points[:, 3:]))

def get_bounding_box_corners(position, dimensions):
    """Compute 8 corners of the cuboid before rotation."""
    dx, dy, dz = dimensions / 2  # Half-lengths
    corners = np.array([
        [dx, dy, dz], [dx, -dy, dz], [-dx, -dy, dz], [-dx, dy, dz],  # Top
        [dx, dy, -dz], [dx, -dy, -dz], [-dx, -dy, -dz], [-dx, dy, -dz]  # Bottom
    ])
    return corners + position  # Translate to object position

def visualize_pointcloud_with_bbox(point_cloud, position, dimensions, rotation_vector, title="3D Point Cloud with Bounding Box"):
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
    ax.set_title(title)

    plt.show()

def visualize_pointcloud_with_arrow(point_cloud, position, dimensions, rotation_vector, arrow_origin, arrow_direction):
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

    # Draw an arrow
    ax.quiver(
        arrow_origin[0], arrow_origin[1], arrow_origin[2],  # Start point
        arrow_direction[0], arrow_direction[1], arrow_direction[2],  # Direction
        color='green',
    )

    # Labels
    ax.set_xlabel("X"), ax.set_ylabel("Y"), ax.set_zlabel("Z")
    ax.set_title("3D Point Cloud with Bounding Box and Arrow")

    plt.show()

def extract_points_inside_bbox(point_cloud, position, dimensions, rotation_vector):
    """Extracts points within the rotated bounding box. Retains additional fields"""
    # Move points to object-centered frame
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    transformed_points = transform_points(point_cloud[:, :3], rotation_matrix.T, -rotation_matrix.T @ position)

    # Check which points are within the bounding box (axis-aligned in transformed space)
    half_dims = dimensions / 2
    within_x = (-half_dims[0] <= transformed_points[:, 0]) & (transformed_points[:, 0] <= half_dims[0])
    within_y = (-half_dims[1] <= transformed_points[:, 1]) & (transformed_points[:, 1] <= half_dims[1])
    within_z = (-half_dims[2] <= transformed_points[:, 2]) & (transformed_points[:, 2] <= half_dims[2])

    # Extract points that belong inside bounding box (in the original frame)
    filtered_points = point_cloud[within_x & within_y & within_z]

    return filtered_points

class Object():
    def __init__(self, name, centroid, timestamp, points=None):
        self.name = name # If object is person or reflector
        if points is not None:
            self.points = points # points inside object
        self.centroid = centroid # object centroid
        self.timestamp = timestamp # object timestamp
        self.velocity = None

    def set_velocity(self, velocity):
        self.velocity = velocity

def main(stop_after=-1, visualize_pointclouds=False, force_recompute=False):
    ### LOAD THE DATASET

    # Path to the downloaded annotation JSON file
    annotation_path = DATASET_PATH / "scene_1" / "annotation.json"

    # Path to the mapping file
    mapping_file = DATASET_PATH / "scene_1" / "frame_pointcloud_map.json"

    # Load the mapping
    with mapping_file.open("r") as f:
        frame_map = json.load(f)

    # Load project metadata (assuming you have it downloaded as well)
    project_meta_json_path = DATASET_PATH / "meta.json"
    project_meta = sly.ProjectMeta.from_json(sly.json.load_json_file(str(project_meta_json_path)))

    # Load annotation from JSON file
    ann = sly.PointcloudEpisodeAnnotation.load_json_file(str(annotation_path), project_meta)

    gt_objects : Dict[str, List[Object]] = {}
    timestamp_to_index : Dict[str, int] = {}

    # TRANSFORM FROM LIDAR FRAME TO LEFT_CAMERA_FRAME, WE NEED THIS TO MATCH OUR OUTPUT
    R_lidar_2_left_cam = np.array(
        [[-0.00261783, -0.94086826,  0.33876256],
        [ 0.99994174, -0.00601038, -0.00896588],
        [ 0.01047181,  0.33871935,  0.94082918]]
    )
    t_lidar_2_left_cam = np.array([ 0.0169, -0.049, 0.095 ])

    if not force_recompute and (SCRIPT_PATH / 'objects.pkl').exists():
        print("Precomputed GTs found, loading them...")
        with (SCRIPT_PATH / 'objects.pkl').open('rb') as file:
            gt_objects = pickle.load(file)
        with (SCRIPT_PATH / 'timestamp_to_index.pkl').open('rb') as file:
            timestamp_to_index = pickle.load(file)
    else:
        print("No precomputed GTs found, computing them now...")
        # RETRIEVE POINTCLOUDS FOR EACH OBJECT
        for i, key in enumerate(frame_map):
            dataset_frame_index = int(key)
            print(f"Processing index {dataset_frame_index}")
            frame_data = ann.frames.get(dataset_frame_index)  # Retrieve frame details
            objects_on_frame = ann.get_objects_on_frame(dataset_frame_index)

            pointcloud_filename = frame_map.get(key)

            figures_on_frame = ann.get_figures_on_frame(dataset_frame_index)
            point_cloud_points = sly.pointcloud.read(str(DATASET_PATH / "scene_1" / "pointcloud" / pointcloud_filename)) # Shape (Nx3)
            point_cloud_data = point_cloud_points

            timestamp = float(pointcloud_filename.removesuffix(".pcd"))

            if len(figures_on_frame) != 2:
                raise RuntimeError("Expected only two figures")

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
                object_points = transform_points(object_points, R_lidar_2_left_cam, t_lidar_2_left_cam)
                if visualize_pointclouds:
                    visualize_points(object_points, f'Pointcloud for {figure.parent_object.obj_class.name} in Frame {i}')
                    visualize_pointcloud_with_bbox(np.concatenate((point_cloud_data, object_points)), position_vec, dimension_vec, rotation_vector, f'Bounding Box for {figure.parent_object.obj_class.name} in Full Pointcloud')

                object_centroid = np.mean(object_points, axis=0)

                object = Object(figure.parent_object.obj_class.name, object_centroid, timestamp, object_points)

                if object.name not in gt_objects:
                    gt_objects[object.name] = [] # Initialize list if it is the first time we see this object type

                gt_objects[object.name].append(object) # Hidden assumption we will only find one object of each type

                print(f"Object {figure.parent_object.obj_class.name} has {len(object_points)} points, with centroid {object_centroid}")

            timestamp_to_index[str(timestamp)] = i # Hidden assumption we will only find one object of each type

            if stop_after > 0 and i >= stop_after:
                print(f"Stopping early after {stop_after} frames as requested")
                break

        ### COMPUTE VELOCITIES BETWEEN PCD PAIRS
        for object_type in gt_objects: # Iterate over each object type (eg. person, reflector)
            object = gt_objects[object_type]
            for i in range(len(object) - 1):
                object_velocity = (object[i + 1].centroid - object[i].centroid) / (object[i + 1].timestamp - object[i].timestamp)
                print(f"Velocity at frame {i}: {object[0].name}: {object_velocity}, time difference: {object[i + 1].timestamp - object[i].timestamp}")
                object[i].set_velocity(object_velocity)

        ### SAVE GTs

        with (SCRIPT_PATH / 'objects.pkl').open('wb') as file:
            pickle.dump(gt_objects, file)
        with (SCRIPT_PATH / 'timestamp_to_index.pkl').open('wb') as file:
            pickle.dump(timestamp_to_index, file)

    ### LOAD PREDICTIONS AND CALCULATE DIFFERENCES

    errors = {}

    for i, key in enumerate(frame_map):
        pointcloud_filename = frame_map.get(key)
        timestamp = float(pointcloud_filename.removesuffix(".pcd"))

        if i < 0: # Change this to skip some frames at the beginning if needed
            continue
        point_cloud_points = sly.pointcloud.read(str(DATASET_PATH / "scene_1" / "pointcloud" / pointcloud_filename)) # Shape (Nx3)
        point_cloud_data = transform_points(point_cloud_points, R_lidar_2_left_cam, t_lidar_2_left_cam)

        if (DATASET_PATH / "scene_1" / "predicted" / pointcloud_filename).exists():
            dataset_frame_index = int(key)
            figures_on_frame = ann.get_figures_on_frame(dataset_frame_index)
            predicted_pcd = o3d.t.io.read_point_cloud(str(DATASET_PATH / "scene_1" / "predicted" / pointcloud_filename))
            predicted_pcd = np.column_stack((
                predicted_pcd.point.positions.numpy(),
                predicted_pcd.point.vx.numpy(),
                predicted_pcd.point.vy.numpy(),
                predicted_pcd.point.vz.numpy(),
            ))

            transformed_predicted_pcd = transform_points(predicted_pcd, R_lidar_2_left_cam.T, -R_lidar_2_left_cam.T @ t_lidar_2_left_cam)

            for figure in figures_on_frame:
                object_geometry = figure.geometry
                position = object_geometry.position
                dimensions = object_geometry.dimensions

                rotation = object_geometry.rotation  # Extract rotation vector

                # Convert to np arrays because it's easier
                rotation_vector = np.array([rotation.x, rotation.y, rotation.z]).astype(np.float32)
                dimension_vec = np.array([dimensions.x, dimensions.y, dimensions.z]).astype(np.float32)
                position_vec = np.array([position.x, position.y, position.z]).astype(np.float32)

                object_points = extract_points_inside_bbox(transformed_predicted_pcd, position_vec, dimension_vec, rotation_vector)
                object_points = transform_points(object_points, R_lidar_2_left_cam, t_lidar_2_left_cam)

                # mask = np.linalg.norm(object_points[:,3:6], axis=1) > 0.01

                if object_points.size == 0:
                    # The object likely moved out of range of our setup, skip these frames
                    # visualize_pointcloud_with_bbox(transformed_predicted_pcd, position_vec, dimension_vec, rotation_vector)
                    continue
                if visualize_pointclouds:
                    visualize_points(object_points, f"Points in {figure.parent_object.obj_class.name} BBox of Predicted Frame {timestamp_to_index[str(timestamp)]}")
                    visualize_pointcloud_with_bbox(transformed_predicted_pcd, position_vec, dimension_vec, rotation_vector, f"Location of {figure.parent_object.obj_class.name} BBox in Predicted Frame {timestamp_to_index[str(timestamp)]}")

                object_vx = np.mean(object_points[:,3])
                object_vy = np.mean(object_points[:,4])
                object_vz = np.mean(object_points[:,5])

                object_velocity = np.array([object_vx, object_vy, object_vz])

                object_centroid = np.mean(object_points[:, :3], axis=0)
                # print(object_centroid)
                # visualize_pointcloud_with_arrow(predicted_pcd, position_vec, dimension_vec, rotation_vector, object_centroid, object_velocity) # not working because not all in the same coordinate frame

                mask = np.linalg.norm(point_cloud_data, axis=1) < 6
                mask2 = np.linalg.norm(point_cloud_data, axis=1) > 1

                if figure.parent_object.obj_class.name not in errors:
                    errors[figure.parent_object.obj_class.name] = {
                        "Velocity Error": [],
                        # "Absolute Component Wise Error": [],
                        "Velocity Angular Error": [],
                        "Weighted Velocity Angular Error": [],
                        "Velocity Magnitudes": [], # This is used to compute the weighted velocity angular error (weighted by the gt velocity magnitude)
                        "Radial Error": [],
                        "Tangential Error": [],
                        "Velocity Magnitude Error": [],
                    }

                object_type: str = figure.parent_object.obj_class.name # The name of the object type, eg. person, reflector
                gt_velocity = gt_objects[object_type][timestamp_to_index[str(timestamp)]].velocity

                errors[object_type]["Velocity Error"].append(np.linalg.norm(gt_velocity - object_velocity))
                # errors[object_type]["Absolute Component Wise Error"].append(np.abs(gt_velocity - object_velocity))
                errors[object_type]["Velocity Angular Error"].append(np.arccos(np.dot(object_velocity, gt_velocity) / (np.linalg.norm(object_velocity) * np.linalg.norm(gt_velocity) + 1e-6))  * 180 / np.pi)
                errors[object_type]["Weighted Velocity Angular Error"].append(errors[object_type]["Velocity Angular Error"][-1] * np.linalg.norm(gt_velocity))
                errors[object_type]["Velocity Magnitudes"].append(np.linalg.norm(gt_velocity))

                rad_v, tan_v = decompose_v(object_velocity, object_centroid)
                gt_rad_v, gt_tan_v = decompose_v(gt_velocity, gt_objects[object_type][timestamp_to_index[str(timestamp)]].centroid)
                
                errors[object_type]["Radial Error"].append(np.linalg.norm(rad_v - gt_rad_v))
                errors[object_type]["Tangential Error"].append(np.linalg.norm(tan_v - gt_tan_v))
                errors[object_type]["Velocity Magnitude Error"].append(np.linalg.norm(object_velocity) - np.linalg.norm(gt_velocity))

                # visualize_points(persons[timestamp_to_index[str(timestamp)]].points, f"GT Points Frame {timestamp_to_index[str(timestamp)]}")
                # print(persons[timestamp_to_index[str(timestamp)]].centroid)
                # visualize_pointcloud_with_arrow(np.concatenate((point_cloud_data[mask & mask2], persons[timestamp_to_index[str(timestamp)]].points)), position_vec, dimension_vec, rotation_vector, persons[timestamp_to_index[str(timestamp)]].centroid, gt_velocity) # not working because not all in the same coordinate frame
                print(f'Frame {i}: {object_type} has velocity {gt_velocity}, pred: {object_velocity},'
                      f'Velocity error: {errors[object_type]["Velocity Error"][-1]},'
                    #   f'Component Wise: {errors[object_type]["Absolute Component Wise Error"][-1]},'
                      f'Angular Error: {errors[object_type]["Velocity Angular Error"][-1]}',
                      f'Radial Error: {errors[object_type]["Radial Error"][-1]}',
                      f'Tangential Error: {errors[object_type]["Tangential Error"][-1]}',
                      f'Magnitude Error:{errors[object_type]["Velocity Magnitude Error"][-1]}')

        if i >= len(gt_objects[next(iter(gt_objects))]) - 2: # Don't check the last frame because no gt velocity
            break

    for object_type in errors: # Iterate over each object type (eg. person, reflector)
        print(f'AVE {object_type}: {np.mean(errors[object_type]["Velocity Error"])} (std: {np.std(errors[object_type]["Velocity Error"])}), '
            #   f'AAE: {np.mean(errors[object_type]["Absolute Component Wise Error"], axis=0)} (std: {np.std(errors[object_type]["Absolute Component Wise Error"], axis=0)}), '
              f'Magnitude RMSE: {np.sqrt(np.mean(np.square(errors[object_type]["Velocity Magnitude Error"])))}')

    overall_errors = {}
    for key in errors[next(iter(errors))]: # Each object type has the same keys
        overall_errors[key] = np.concatenate([errors[object_type][key] for object_type in errors])

    print(f'AVE Overall: {np.mean(overall_errors["Velocity Error"])} '
        f' (std: {np.std(overall_errors["Velocity Error"])}), '
        # f' AAE: {np.mean(overall_errors["Absolute Component Wise Error"], axis=0)}'
        # f' (std: {np.std(overall_errors["Absolute Component Wise Error"], axis=0)}),'
        f'Mean Angular Error: {np.mean(overall_errors["Velocity Angular Error"])}, ',
        f'Weighted Mean Angular Error: {np.sum(overall_errors["Weighted Velocity Angular Error"]) / (np.sum(overall_errors["Velocity Magnitudes"]) + 1e-6)}, ',
        f'Mean Radial Error: {np.mean(overall_errors["Radial Error"])}, ',
        f'Mean Tangential Error: {np.mean(overall_errors["Tangential Error"])}, ',
        f' Magnitude RMSE: {np.sqrt(np.mean(np.square(overall_errors["Velocity Magnitude Error"])))}')

if __name__ == "__main__":
    # Read command line arguments
    parser = argparse.ArgumentParser(description="Evaluate point cloud velocity predictions.")
    parser.add_argument('--stop-after', type=int, default=-1, help='Stop processing after this many frames. By default, process all frames.')
    parser.add_argument('--visualize-pointclouds', action='store_true', help='Visualize point clouds and bounding boxes.')
    parser.add_argument('--force-recompute', action='store_true', help='Force recomputation of GTs even if precomputed files exist.')
    args = parser.parse_args()

    main(args.stop_after, args.visualize_pointclouds, args.force_recompute)
