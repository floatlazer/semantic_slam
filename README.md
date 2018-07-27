# Semantic SLAM
***Author:*** Xuan Zhang

Semantic SLAM can generate a 3D voxel based semantic map using only a hand holding RGB-D camera (e.g. Asus xtion) in real time. We use ORB_SLAM2 as SLAM backend, a CNN (PSPNet) to produce semantic prediction and fuse semantic information into a octomap. Note that our system can also be configured to generate rgb octomap without semantic information.

![alt text](docs/images/rviz_screenshot_2018_07_19-17_40_36.png)

### Project Report:

Coming soon...

### Demo:

Coming soon...

### Acknowledgement

This work cannot be done without many open source projets. Special thanks to

- [ORB_SLAM2](https://github.com/raulmur/ORB_SLAM2), used as our SLAM backend.
- [pytorch-semseg](https://github.com/meetshah1995/pytorch-semseg/tree/master/ptsemseg), used as our semantic segmantation library.
- [octomap](https://github.com/OctoMap/octomap), used as our map representation.
- [pcl library](http://pointclouds.org/), used for point cloud processing.

# License

Semantic SLAM is released under a [MIT license](./LICENSE.txt) except for [ORB_SLAM2](https://github.com/raulmur/ORB_SLAM2) which is released under a [GPLv3 license](./ORB_SLAM2/License-gpl.txt).

# Dependencies

- Openni2_launch

```sh
sudo apt-get install ros-kinetic-openni2-launch
```

- ORB_SLAM2

We use [ORB_SLAM2](https://github.com/raulmur/ORB_SLAM2) as SLAM backend. Please refer to the official repo for installation dependency.

- Octomap

```sh
sudo apt-get install ros-kinetic-octomap
```

- Octomap rviz plugins

```sh
sudo apt-get install ros-kinetic-octomap-kinetic-plugins
```

- PyTorch 0.4.0

- To be finished...

# Installation

After installing dependency for ORB_SLAM. You should first build the library.

```sh
cd orb_slam2
./build.sh
```

Then build the package.

```sh
cd <your_catkin_work_space>
catkin_make
```

# Overview

Coming soon...

# Configuration

You can change parameters for launch. Parameters are in `./semantic_slam/params` folder.

### Parameters for octomap_generator node (octomap_generator.yaml)

namespace octomap

- pointcloud_topic: topic of input point cloud topic

- tree_type: OcTree type. 0 for ColorOcTree, 1 for SemanticsOcTree using max fusion (keep the most confident), 2 for SemanticsOcTree using bayesian fusion (fuse top 3 most confident semantic colors). See project report for details of fusion methods.

- world_frame_id: frame id of world frame.

- resolution: resolution of octomap.

- max_range: maximum distance of a point from camera to be inserted into octomap.

- raycast_range: maximum distance of a point from camera be perform raycasting to clear free space.

- clamping_thres_min: octomap parameter, minimum octree node occupancy during update.   

- clamping_thres_max: octomap parameter, maximum octree node occupancy during update.

- occupancy_thres: octomap parameter, octree node occupancy to be considered as occupied

- prob_hit: octomap parameter, hitting probability of the sensor model.

- prob_miss: octomap parameter, missing probability of the sensor model.

### Parameters for semantic_cloud node (semantic_cloud.yaml)

namespace camera

- fx, fy, cx, cy: camera intrinsic matrix parameters.

- width, height: image size.

namespace semantic_pcl

- color_image_topic: topic for input color image.

- depth_image_topic: topic for input depth image.

- point_type: point cloud type, should be same as octomap/tree_type. 0 for color point cloud, 1 for semantic point cloud including top 3 most confident semanic colors and their confidences, 2 for semantic including most confident semantic color and its confident. See project report for details of point cloud types.

- frame_id: point cloud frame id.

- dataset: dataset that PSPNet trained on. "ade20k" or "sunrgbd".

- model_path: path to pytorch trained model.

***Note that you can set octomap/tree_type and semantic_cloud/point_type to 0 to generate a map with rgb color without doing semantic segmantation.***

# Run

First you should have a rgbd camera running.

```sh
roslaunch semantic_slam camera.launch
```

The run ORB_SLAM2 node.

```sh
roslaunch semantic_slam slam.launch
```

When the slam system has finished initialization and the camera trajectory in the viewer is reasonable, you can run the semantic_cloud node and the octomap_generator node.

```sh
roslaunch semantic_slam semantic_mapping.launch
```

This will also launch rviz for visualization.

You can then move around the camera and construct semantic map. Make sure SLAM is not losing itself.

If you are constructing a semantic map, you can toggle the display color between semantic color and rgb color by running

```sh
rosservice call toggle_use_semantic_color
```
