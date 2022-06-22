# ximea_ros_cam

This is a fork of the ximea_ros_cam ros driver. It is modified for control and operation of the WildPose data collection rig.

# Installation

The ROS branch of the WildPose rig software is built on ROS Melodic for Ubuntu 18.04. It has been tested on Jetpack 4.x on the Jetson AGX Xavier platform. However, the installation and setup process should be near identical for other Jetson platforms such as the Jetson Nano.

## Prerequisites

This repository has a couple major dependencies before it can be built and run:

- The [Ximea XiAPI](https://www.ximea.com/support/wiki/apis/xiapi). Ensure that you have installed the correct firmware for your model.
- [Livox ROS Driver](https://github.com/Livox-SDK/livox_ros_driver). Installing this repository and following the setup instructions will guide you through ROS setup, the Livox SDK, and installation of the ROS driver itself.

## WildPose Setup

Once you have installed the prerequisites, you may follow these steps to install the WildPose ROS driver:

1.  Ensure that you have created a [catkin workspace](http://wiki.ros.org/catkin/Tutorials/create_a_workspace) on your system. You should have done this already for installation of the Livox ROS driver; you should continue with the installation in this same workspace.
2.  Clone this repository in your catkin_ws/src directory with `git clone -b ros https://github.com/African-Robotics-Unit/wildpose.git`
3.  Return to your `/catkin_ws` directory and run `catkin_make` to build the project.
4.  Run `source devel/setup.bash` (assuming you are in a BASH shell).

If this all went off without a hitch you are now set up and ready to run! You can run the example program to capture lidar and RGB data by running `roslaunch ximea_ros_cam example_cam.launch`

# Usage

To start recording on the wildpose rig from a powered off state, perform the following steps:

1.  Press the power button on the Xavier (left most button)
2.  Wait for the boot process to complete. This will take about 50 seconds.
3.  Connect the lidar's ethernet cable to the Xavier
4.  Flick the switch on the PCB.
5.  The lidar should boot. Wait approximately 20 seconds for the startup process.
6.  Double-tap the wildpose_start script on the homescreen.

# Contact

If you notice any bugs, issues, or have any other suggestions to improve the repository, you may either raise an issue on this repository or contact me at:

Email: JSKDAN001@myuct.ac.za
