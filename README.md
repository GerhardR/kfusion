
KFusion 0.2
=============

Copyright TU Graz, Gerhard Reitmayr, 2011 - 2012

This is an implementation sketch of the KinectFusion system described by
Newcombe et al. in "KinectFusion: Real-Time Dense Surface Mapping and Tracking",
ISMAR 2011, 2011. It is a dense surface reconstruction and tracking framework
using a single Kinect camera as the input sensor.

KFusion is mainly written in CUDA with some interface code to display graphics output.


Requirements
------------

KFusion depends on the following libraries:

* http://www.edwardrosten.com/cvd/toon.html
* http://openkinect.org/
* GLUT

and of course the CUDA 4.1 SDK by NVidia

* http://developer.nvidia.com/category/zone/cuda-zone

Install
-----
To get started, tweak the Makefile for your setup and have a look at kfusion.h
to set some defines to adapt to your compute capabilities. Then make.

Todo
-----
- rendering
  - integrate with GL for additional 3D graphics
  - interactive viewpoint
- write an inverse tracking method that moves the camera in the system
- save size through combined depth + 2D normal maps

Done
-----
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
