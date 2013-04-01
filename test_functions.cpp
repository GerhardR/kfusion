
#include "kfusion.h"
#include "helpers.h"

#include <iostream>
#include <sstream>
#include <iomanip>

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include "perfstats.h"

using namespace std;
using namespace TooN;

SE3<float> preTrans(makeVector(0.0, 0, -0.9, 0, 0, 0));
SE3<float> rot(makeVector(0.0, 0, 0, 0, 0, 0));
SE3<float> trans(makeVector(0.5, 0.5, 0.5, 0, 0, 0));

uint2 imageSize;

Raycast raycaster;
Integrate integrater;
Track tracker;

Volume integration, reference;
Image<float3, HostDevice> vertex, normal;
Image<float, HostDevice> depth;
Image<uchar4, HostDevice> rgb;

Matrix4 pose;

int counter = 0;
bool benchmark = false;

void display(void) {

    static bool integrate = true;

    const double start = Stats.start();
    raycaster(vertex.getDeviceImage(), normal.getDeviceImage(), depth.getDeviceImage(), reference, toMatrix4( trans * rot * preTrans ));
    CUDA_ERROR;
    cudaDeviceSynchronize();
    CUDA_ERROR;
    Stats.sample("ground raycast");
    Stats.sample("ground copy");

    glRasterPos2i(0,0);
    glDrawPixels(vertex);
    glRasterPos2i(imageSize.x, 0);
    glDrawPixels(normal);
    glRasterPos2i(imageSize.x * 2, 0);
    glDrawPixels(depth);
    const double track_start = Stats.sample("ground draw");

#if 1

    if(counter > 1){
        raycaster(vertex.getDeviceImage(), normal.getDeviceImage(), integration, pose);
        CUDA_ERROR;
        pair<Matrix4, bool> result = tracker(depth.getDeviceImage(), vertex.getDeviceImage(), normal.getDeviceImage(),pose);
        CUDA_ERROR;
        cudaDeviceSynchronize();
        CUDA_ERROR;
        pose = result.first;
        integrate = result.second;
        Stats.sample("track");
    }

#endif

    renderTrackResult(rgb.getDeviceImage(), tracker.reduction);
    CUDA_ERROR;
    cudaDeviceSynchronize();
    CUDA_ERROR;
    Stats.sample("track render");
    Stats.sample("track copy");

    glRasterPos2i(0,imageSize.y * 1);
    glDrawPixels(vertex);
    glRasterPos2i(imageSize.x, imageSize.y * 1);
    glDrawPixels(normal);
    glRasterPos2i(2 * imageSize.x, imageSize.y * 1);
    glDrawPixels(rgb);
    Stats.sample("track draw");

    if(integrate){
        integrater(integration, depth.getDeviceImage(), pose);
        CUDA_ERROR;
        cudaDeviceSynchronize();
        CUDA_ERROR;
        Stats.sample("integration");
    }
    Stats.sample("total track", Stats.get_time() - track_start, PerfStats::TIME);

    raycaster(vertex.getDeviceImage(), normal.getDeviceImage(), depth.getDeviceImage(), integration, pose);
    CUDA_ERROR;
    cudaDeviceSynchronize();
    CUDA_ERROR;

    Stats.sample("view raycast");
    Stats.sample("view copy");

    glRasterPos2i(0,imageSize.y * 2);
    glDrawPixels(vertex);
    glRasterPos2i(imageSize.x, imageSize.y * 2);
    glDrawPixels(normal);
    glRasterPos2i(imageSize.x * 2, imageSize.y * 2);
    glDrawPixels(depth);
    Stats.sample("view draw");

    Stats.sample("events");
    Stats.sample("total all", Stats.get_time() - start, PerfStats::TIME);

    if(counter % 30 == 0){
        Stats.print();
        Stats.reset();
        cout << endl;
    }

    ++counter;

    CUDA_ERROR;

    glutSwapBuffers();
}

void keys(unsigned char key, int x, int y) {
    switch(key){
    case 'r':
        pose = toMatrix4( trans * rot * preTrans );
        break;
    case 'c':
        InitVolume(integration);
        pose = toMatrix4( trans * rot * preTrans );
        break;
    case 'd':
        cout << pose << endl;
        break;
    case 'q':
        exit(0);
        break;
    }
    glutPostRedisplay();
}

void specials(int key, int x, int y){
    switch(key){
    case GLUT_KEY_LEFT:
        rot *= SE3<float>(makeVector(0.0, 0, 0, 0, 0.1, 0));
        break;
    case GLUT_KEY_RIGHT:
        rot *= SE3<float>(makeVector(0.0, 0, 0, 0, -0.1, 0));
        break;
    case GLUT_KEY_UP:
        rot *= SE3<float>(makeVector(0.0, 0, 0, -0.1, 0, 0));
        break;
    case GLUT_KEY_DOWN:
        rot *= SE3<float>(makeVector(0.0, 0, 0, 0.1, 0, 0));
        break;
    }
    glutPostRedisplay();
}

void idle(void) {
    if(counter > 100 && benchmark)
        exit(0);

    if(benchmark)
        rot *= SE3<float>(makeVector(0.0, 0, 0, 0, 0.02, 0));

    glutPostRedisplay();
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

int main(int argc, char ** argv) {

    benchmark = argc > 1 && string(argv[1]) == "-b";

    cudaSetDeviceFlags(cudaDeviceMapHost);

    KFusionConfig config;
    config.volumeSize = make_uint3(128);

    config.combinedTrackAndReduce = false;

    config.iterations[0] = 10;
    config.iterations[1] = 5;
    config.iterations[2] = 5;

    config.inputSize = make_uint2(320, 240);
    config.camera = make_float4(100, 100, 160, 120);
    config.nearPlane = 0.001;

    config.maxweight = 100;
    config.mu = 0.1;

    config.dist_threshold = 0.2f;
    config.normal_threshold = 0.8f;

    imageSize = config.inputSize;
    raycaster = config;
    raycaster.largeSteps = 0.01;
    integrater = config;
    tracker = config;

    integration.init(config.volumeSize, config.volumeDimensions);
    CUDA_ERROR;
    InitVolume(integration);
    CUDA_ERROR;

    reference.init(config.volumeSize, config.volumeDimensions);
    CUDA_ERROR;
    InitVolume(reference);
    CUDA_ERROR;

    setBoxWrap(reference, make_float3(0.1f,0.1f,0.8f), make_float3(0.9f, 0.9f, 0.9f), -1.0f);
    setBoxWrap(reference, make_float3(0.1f,0.8f,0.1f), make_float3(0.9f, 0.9f, 0.9f), -1.0f);
    setBoxWrap(reference, make_float3(0.8f,0.1f,0.1f), make_float3(0.9f, 0.9f, 0.9f), -1.0f);
    setSphereWrap(reference, make_float3(0.5f), 0.2f, -1.0f);

    CUDA_ERROR;

    pose = toMatrix4( trans * rot * preTrans );

    vertex.alloc(config.inputSize);
    normal.alloc(config.inputSize);
    depth.alloc(config.inputSize);
    rgb.alloc(config.inputSize);

    CUDA_ERROR;

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE );
    glutInitWindowSize(config.inputSize.x * 3, config.inputSize.y * 3);
    glutCreateWindow("kfusion test");

    glutDisplayFunc(display);
    glutKeyboardFunc(keys);
    glutSpecialFunc(specials);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);

    glutMainLoop();

    return 0;
}
