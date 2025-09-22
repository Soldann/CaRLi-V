# CaRLi-V: Camera-RADAR-LiDAR point-wise Velocity estimation

Accurate point-wise velocity estimation in 3D is vital for enabling robot interaction with non-rigid dynamic agents such as humans, enabling robust performance in path planning, collision avoidance, and object manipulation in dynamic settings. To this end, we propose a novel RADAR, LiDAR, and camera fusion pipeline for point-wise 3D velocity estimation named CaRLi-V. This pipeline leverages a novel RADAR representation, the velocity cube, which densely represents radial velocities within the RADAR's field-of-view. By combining the velocity cube for radial velocity extraction, optical flow for tangential velocity estimation, and LiDAR for dense point-wise localization measurements through a closed-form formulation, our approach can produce detailed velocity measurements for a dense array of points. This repository comprises of a ROS2 package for 3D dense point-wise velocity estimation using a camera, RADAR, and LiDAR fusion framework.

![CaRLi-V Pipeline Diagram](assets/pipeline_visualization.png)
The pipeline is divided into three steps: RADAR pre-processing, where raw ADC RADAR data is used to compute the RADAR velocity cube, a dense representation of velocities in the environment; Camera pre-processing, where two consecutive camera images are used to compute optical flow vectors for each pixel in the image; Sensor fusion, where the LiDAR point cloud is projected into both representations to extract radial velocity and optical flow readings, with both estimates combined through a closed-form solution.

![Results Visualization](assets/results_visualization_cropped.png)
The pipeline is able to (a) discern velocities between different dynamic agents; (b) estimate both radial and tangential velocities, as well as combinations of both, and (c, d) extract velocities from single parts of non-rigid moving agents, such as individual limbs in humans.

See a demonstration of our pipeline in action by clicking on the image below:

[![Link to YouTube Video](https://img.youtube.com/vi/oy1B_Mmpvt0/0.jpg)](https://youtu.be/oy1B_Mmpvt0)

## Installation

This repository was built on Ubuntu 22.04 for ROS2 Humble. To install the necessary packages to run the ROS nodes:

```
cd ros
pip install -r src/carli_v/requirements.txt
```

Then build using colcon:

```
colcon build --symlink-install
source install/setup.bash
```

## Usage
Several launch files exist to facilitate the running of the pipeline. If running on live data, the nodes expect LiDAR messages to be broadcast on the topic `/hesai_lidar` as a PointCloud2 topic, RADAR ADC data to be broadcast as an Image on `/radar_ADC`, and image data in `/camera/image_raw`. Otherwise, see [Testing](#testing) for testing with an annotated dataset.

To run the full pipeline:
```
ros2 launch carli_v radar_full_velocity.launch.py
```

To run only the radar-lidar fusion node (for radial velocities only):
```
ros2 launch carli_v radar_mode.launch.py
```

## Testing
To reproduce the test results in our paper, download our annotated ground-truth dataset into the folder `/datasets` under this repository (public link will be provided later, for now on request only). You will need to download the rosbag of sensor data seperately as well.

<details>
<summary>
Steps for using a custom ground-truth dataset
</summary>
If using a custom dataset with our eval script, it should be in [Supervisely format](https://docs.supervisely.com/import-and-export/import/supported-annotation-formats/pointclouds/supervisely). Place the dataset under the folder `/datasets`,  The eval script will analyze objects with the name `Person` or `Reflector` in Scene 1
</details>

You can then export the raw pointclouds from our ROS node implementation as follows:
```
ros2 launch carli_v radar_full_velocity.launch.py save_pcd_as:=scene_1
```
This will save the output of the pipeline as a PCD file every time a new output is published.

Then install and run the eval script:
```
pip install -r eval/requirements.txt
python3 eval/analysis.py
```
