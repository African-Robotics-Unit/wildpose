# WildPose: 3D Motion Tracking of Cheetahs in the Wild using LIDAR-RGB Sensor Fusion and Deep Learning

This repository contains code used for for the M.Sc. thesis written by Daniel Joska.

## Installation

### C++

1.  Requirements

For the C++ portion of this repository, there are 3 dependencies:

- The Livox SDK, for interfacing with the Livox Tele-15 LIDAR.
- The Ximea XiAPI, for interfacing with the Ximea XiQ USB3.0 camera.
- Libtiff, for image formatting

Ensure that these dependencies are compiled and installed on your version of Linux.

2.  Program Installation

Simply open a terminal window within this directory and run the following:

```
mkdir build
cd build
cmake ..
make
sudo make install
```

Since this process requires `sudo` privileges you may be prompted for a password.

### Python

# TODO

## Usage

Connect the Livox Tele-15 and the Ximea camera to your system as instructed in their respective documentation. Once both are connected and working, you may simply run the following command to capture 10 image-pointcloud pairs:

`wildpose`