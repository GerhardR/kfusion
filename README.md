
KFusion 0.4
=============

Copyright TU Graz, Gerhard Reitmayr, 2011 - 2013

This is an implementation sketch of the KinectFusion system described by
Richard Newcombe et al. in "KinectFusion: Real-Time Dense Surface Mapping and Tracking",
ISMAR 2011, 2011. It is a dense surface reconstruction and tracking framework
using a single Kinect camera as the input sensor.

http://research.microsoft.com/en-us/projects/surfacerecon/

KFusion is mainly written in CUDA with some interface code to display graphics output.

Requirements
------------

You need a depth camera: Either a Microsoft Kinect or any camera supported
by OpenNI 2 (such as the Asus Xtion Pro Live).

KFusion depends on the following libraries:

* http://www.edwardrosten.com/cvd/toon.html
* GLUT

On Windows use the MS Kinect SDK:

* http://www.microsoft.com/en-us/kinectforwindows/develop/overview.aspx

while on other platforms use either:

* libfreenect: http://openkinect.org or
* OpenNI: https://github.com/OpenNI/OpenNI2

and of course the CUDA 5 SDK by NVidia

* http://developer.nvidia.com/cuda

Install
-----

Use CMake to create build files for your platform. Some tips and tricks
- On Windows, make sure to use a 64-bit version of GLUT
- On Apple OSX, set CUDA_HOST_COMPILER to /usr/bin/g++

Altenatively, On Unix/OSX platforms, tweak the Makefile for your setup, then make.

Have a look at kfusion.h for a description of most parameters and kinect.cpp for setting them.

Todo
-----
- rendering
  - integrate with GL for additional 3D graphics
- write an inverse tracking method that moves the camera in the system
- save size through combined depth + 2D normal maps

Done
-----
- MSKinect SDK interface for Windows, libfreenect on other platforms
- rendering with static model view + projected RGB + interactive viewpoint
- registered depth input from libfreenect, uses more time unfortunately
- integration speed up
- CMake build system (contributed by Hartmut Seichter)
- fixed a substantial bug in tracking
- improved raycasting by an implementation closer to the paper. This also seems to take care of the following issue:
    - tracking works much better with a detailed model and sharp bounds on normals (0.9), problem for low resultion ?
- replaced libcvd with GLUT in the master branch
- created dedicated Image class templated on different memory locations,
  reduces most dependencies on libcvd
- removed all 3D grids to reduce code and maybe speed up as well
- 2D grid for volume integration is 40% faster
- fixed a difference in computing normals in raycasting and from kinect input... maybe influences tracking ?
- added a combined tracking & reduce implementation that saves a bit of time
- make configuration more automatic, compute dependent variables, scales for tracking
- make a version that uses one volume for both SDF + weight
  - split 32bit into 16bit float [-1,-1] and unsigned int for weight
- scaling for better optimization, didn't really do much, removed it again
- multi level tracking
- split
  - core operations
  - testing with extra volume etc
  - rendering
  - make OO ?
- speed up raycasting with mu step sizes
- reduction somewhat better through local memory -> shared memory
- test larger volume sizes
  - 3D operations over 2D grid - works nicely !
- pitched images no change, because they are already all pitched !
- template volume on data type and convert to/from float on the fly - did not change speed at all !
- ambient lighting
- bilateral filtering

Contributors
-------------
Hartmut Seichter
