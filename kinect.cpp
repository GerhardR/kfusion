/*
Copyright (c) 2011-2013 Gerhard Reitmayr, TU Graz

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include "kfusion.h"
#include "helpers.h"
#include "interface.h"
#include "perfstats.h"

#include <cstring>
#include <iostream>
#include <sstream>
#include <iomanip>

#ifdef __APPLE__
#include <GLUT/glut.h>
#elif defined(WIN32)
#define GLUT_NO_LIB_PRAGMA
#include <glut.h>
#else
#include <GL/glut.h>
#endif

using namespace std;
using namespace TooN;

KFusion kfusion;
Image<uchar4, HostDevice> lightScene, trackModel, lightModel, texModel;
Image<uint16_t, HostDevice> depthImage[2];
Image<uchar3, HostDevice> rgbImage;

const float3 light = make_float3(1, 1, -1.0);
const float3 ambient = make_float3(0.1, 0.1, 0.1);

SE3<float> initPose;

int counter = 0;
int integration_rate = 2;
bool reset = true;
bool should_integrate = true;
bool render_texture = false;

Image<float3, Device> pos, normals;
Image<float, Device> dep;

SE3<float> preTrans, trans, rot(makeVector(0.0, 0, 0, 0, 0, 0));
bool redraw_big_view = false;

void display(void){
    const uint2 imageSize = kfusion.configuration.inputSize;
    static bool integrate = true;

    glClear( GL_COLOR_BUFFER_BIT );
    const double startFrame = Stats.start();
    const double startProcessing = Stats.sample("kinect");

    kfusion.setKinectDeviceDepth(depthImage[GetKinectFrame()].getDeviceImage());
    Stats.sample("raw to cooked");

    integrate = kfusion.Track();
    Stats.sample("track");

    if((should_integrate && integrate && ((counter % integration_rate) == 0)) || reset){
        kfusion.Integrate();
        kfusion.Raycast();
        Stats.sample("integrate");
        if(counter > 2) // use the first two frames to initialize
            reset = false;
    }

    renderLight( lightScene.getDeviceImage(), kfusion.inputVertex[0], kfusion.inputNormal[0], light, ambient );
    renderLight( lightModel.getDeviceImage(), kfusion.vertex, kfusion.normal, light, ambient);
    renderTrackResult(trackModel.getDeviceImage(), kfusion.reduction);
    static int count = 4;
    if(count > 3 || redraw_big_view){
        renderInput( pos, normals, dep, kfusion.integration, toMatrix4( trans * rot * preTrans ) * getInverseCameraMatrix(kfusion.configuration.camera * 2), kfusion.configuration.nearPlane, kfusion.configuration.farPlane, kfusion.configuration.stepSize(), 0.75 * kfusion.configuration.mu);
        count = 0;
        redraw_big_view = false;
    } else
        count++;
    if(render_texture)
        renderTexture( texModel.getDeviceImage(), pos, normals, rgbImage.getDeviceImage(), getCameraMatrix(2*kfusion.configuration.camera) * inverse(kfusion.pose), light);
    else
        renderLight( texModel.getDeviceImage(), pos, normals, light, ambient);
    cudaDeviceSynchronize();

    Stats.sample("render");

    glClear(GL_COLOR_BUFFER_BIT);
    glRasterPos2i(0, 0);
    glDrawPixels(lightScene);
    glRasterPos2i(0, 240);
    glPixelZoom(0.5, -0.5);
    glDrawPixels(rgbImage);
    glPixelZoom(1,-1);
    glRasterPos2i(320,0);
    glDrawPixels(lightModel);
    glRasterPos2i(320,240);
    glDrawPixels(trackModel);
    glRasterPos2i(640, 0);
    glDrawPixels(texModel);
    const double endProcessing = Stats.sample("draw");

    Stats.sample("total", endProcessing - startFrame, PerfStats::TIME);
    Stats.sample("total_proc", endProcessing - startProcessing, PerfStats::TIME);

    if(printCUDAError())
        exit(1);

    ++counter;

    if(counter % 50 == 0){
        Stats.print();
        Stats.reset();
        cout << endl;
    }

    glutSwapBuffers();
}

void idle(void){
    if(KinectFrameAvailable())
        glutPostRedisplay();
}

void keys(unsigned char key, int x, int y){
    switch(key){
    case 'c':
        kfusion.Reset();
        kfusion.setPose(toMatrix4(initPose));
        reset = true;
        break;
    case 'q':
        exit(0);
        break;
    case 'i':
        should_integrate = !should_integrate;
        break;
    case 't':
        render_texture = !render_texture;
        break;
    }
}

void specials(int key, int x, int y){
    switch(key){
    case GLUT_KEY_LEFT:
        rot = SE3<float>(makeVector(0.0, 0, 0, 0, 0.1, 0)) * rot;
        break;
    case GLUT_KEY_RIGHT:
        rot = SE3<float>(makeVector(0.0, 0, 0, 0, -0.1, 0)) * rot;
        break;
    case GLUT_KEY_UP:
        rot *= SE3<float>(makeVector(0.0, 0, 0, -0.1, 0, 0));
        break;
    case GLUT_KEY_DOWN:
        rot *= SE3<float>(makeVector(0.0, 0, 0, 0.1, 0, 0));
        break;
    }
    redraw_big_view = true;
}

void reshape(int width, int height){
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glColor3f(1.0f,1.0f,1.0f);
    glRasterPos2f(-1, 1);
    glOrtho(-0.375, width-0.375, height-0.375, -0.375, -1 , 1); //offsets to make (0,0) the top left pixel (rather than off the display)
    glPixelZoom(1,-1);
}

void exitFunc(void){
    CloseKinect();
    kfusion.Clear();
    cudaDeviceReset();
}

int main(int argc, char ** argv) {
    const float default_size = 2.f;

    KFusionConfig config;

    // Search for --help argument
    for (int i = 0; i < argc; ++i) {
        if (string(argv[i]) == "--help") {
            cout << "Usage: kinect [size] [dist_threshold] [normal_threshold]" << endl;
            cout << endl;
            cout << "Defaults:" << endl;
            cout << "  size: " << default_size << endl;
            cout << "  dist_threshold: " << config.dist_threshold << endl;
            cout << "  normal_threshold: " << config.normal_threshold << endl;
            return 0;
        }
    }

    const float size = (argc > 1) ? atof(argv[1]) : default_size;

    // it is enough now to set the volume resolution once.
    // everything else is derived from that.
    // config.volumeSize = make_uint3(64);
    // config.volumeSize = make_uint3(128);
    config.volumeSize = make_uint3(256);

    // these are physical dimensions in meters
    config.volumeDimensions = make_float3(size);
    config.nearPlane = 0.4f;
    config.farPlane = 5.0f;
    config.mu = 0.1;
    config.combinedTrackAndReduce = false;

    // change the following parameters for using 640 x 480 input images
    config.inputSize = make_uint2(320,240);
    config.camera =  make_float4(531.15/2, 531.15/2, 640/4, 480/4);

    // config.iterations is a vector<int>, the length determines
    // the number of levels to be used in tracking
    // push back more then 3 iteraton numbers to get more levels.
    config.iterations[0] = 10;
    config.iterations[1] = 5;
    config.iterations[2] = 4;

    config.dist_threshold = (argc > 2 ) ? atof(argv[2]) : config.dist_threshold;
    config.normal_threshold = (argc > 3 ) ? atof(argv[3]) : config.normal_threshold;

    initPose = SE3<float>(makeVector(size/2, size/2, 0, 0, 0, 0));

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE );
    glutInitWindowSize(config.inputSize.x * 2 + 640, max(config.inputSize.y * 2, 480));
    glutCreateWindow("kfusion");

    kfusion.Init(config);

    // input buffers
    depthImage[0].alloc(make_uint2(640, 480));
    depthImage[1].alloc(make_uint2(640, 480));
    rgbImage.alloc(make_uint2(640, 480));

    // render buffers
    lightScene.alloc(config.inputSize), trackModel.alloc(config.inputSize), lightModel.alloc(config.inputSize);
    pos.alloc(make_uint2(640, 480)), normals.alloc(make_uint2(640, 480)), dep.alloc(make_uint2(640, 480)), texModel.alloc(make_uint2(640, 480));

    if(printCUDAError()) {
        cudaDeviceReset();
        return 1;
    }

    memset(depthImage[0].data(), 0, depthImage[0].size.x*depthImage[0].size.y * sizeof(uint16_t));
    memset(depthImage[1].data(), 0, depthImage[1].size.x*depthImage[1].size.y * sizeof(uint16_t));
    memset(rgbImage.data(), 0, rgbImage.size.x*rgbImage.size.y * sizeof(uchar3));

    uint16_t * buffers[2] = {depthImage[0].data(), depthImage[1].data()};
    if(InitKinect(buffers, (unsigned char *)rgbImage.data())){
        cudaDeviceReset();
        return 1;
    }

    kfusion.setPose(toMatrix4(initPose));

    // model rendering parameters
    preTrans = SE3<float>::exp(makeVector(0.0, 0, -size, 0, 0, 0));
    trans = SE3<float>::exp(makeVector(0.5, 0.5, 0.5, 0, 0, 0) * size);

    atexit(exitFunc);
    glutDisplayFunc(display);
    glutKeyboardFunc(keys);
    glutSpecialFunc(specials);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);

    glutMainLoop();

    CloseKinect();

    return 0;
}
