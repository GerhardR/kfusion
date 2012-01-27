
KFusion 0.1
=============

Copyright TU Graz, Gerhard Reitmayr, 2011

This is an implementation sketch of the KinectFusion system described by
Newcombe et al. in "KinectFusion: Real-Time Dense Surface Mapping and Tracking",
ISMAR 2011, 2011. It is a dense surface reconstruction and tracking framework
using a single Kinect camera as the input sensor.

KFusion is mainly written in CUDA with some interface code to display graphics output.


Requirements
------------

KFusion depends on the following libraries:

* http://www.edwardrosten.com/cvd/toon.html
* http://www.edwardrosten.com/cvd/index.html
* http://openkinect.org/

and of course the CUDA 4.0 SDK by NVidia

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
- tracking works much better with a detailed model and sharp bounds on normals (0.9), problem for low resultion ?

Done
-----
- fixed a difference in computing normals in raycasting and from kinect input... maybe influences tracking ?
- added a combined tracking & reduce implementation that saves a bit of time
- make configuration more automatic, compute dependent variables, scales for tracking
- make a version that uses one volume for both SDF + weight
  - split 32bit into 16bit float [-1,-1] and unsigned int for weight
- scaling for better optimization
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
