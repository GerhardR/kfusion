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

using namespace TooN;
using namespace std;

#include "perfstats.h"

SE3<float> preTrans(makeVector(0.0, 0, -0.9, 0, 0, 0));
SE3<float> rot(makeVector(0.0, 0, 0, 0, 0, 0));
SE3<float> trans(makeVector(0.5, 0.5, 0.5, 0, 0, 0));

KFusion kfusion;
Volume reference;
Image<float3, HostDevice> vertex, normal;
Image<float, HostDevice> depth;
Image<uchar4, HostDevice> rgb;

int counter = 0;
bool benchmark = false;

void display(void) {

    static bool integrate = true;

    const uint2 imageSize = kfusion.configuration.renderSize();

    const double start = Stats.start();
    raycastWrap(vertex.getDeviceImage(), normal.getDeviceImage(), depth.getDeviceImage(), reference, toMatrix4( trans * rot * preTrans ) * getInverseCameraMatrix(kfusion.configuration.camera), kfusion.configuration.nearPlane, kfusion.configuration.farPlane, kfusion.configuration.stepSize(), 0.01 );
    cudaDeviceSynchronize();
    Stats.sample("ground raycast");
    Stats.sample("ground copy");

    glRasterPos2i(0,0);
    glDrawPixels(vertex);
    glRasterPos2i(imageSize.x, 0);
    glDrawPixels(normal);
    glRasterPos2i(imageSize.x * 2, 0);
    glDrawPixels(depth);
    Stats.sample("ground draw");

    kfusion.setDepth( depth.getDeviceImage() );
    cudaDeviceSynchronize();
    const double track_start = Stats.sample("process depth");

    if(counter > 1){
        integrate = kfusion.Track();
        cudaDeviceSynchronize();
        Stats.sample("track");
    }

    vertex = kfusion.vertex;
    normal = kfusion.normal;
    Stats.sample("track get");

    renderTrackResult(rgb.getDeviceImage(), kfusion.reduction);
    cudaDeviceSynchronize();
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
        kfusion.Integrate();
        cudaDeviceSynchronize();
        Stats.sample("integration");
    }
    Stats.sample("total track", Stats.get_time() - track_start, PerfStats::TIME);

    raycastWrap(vertex.getDeviceImage(), normal.getDeviceImage(), depth.getDeviceImage(), kfusion.integration,  kfusion.pose * getInverseCameraMatrix(kfusion.configuration.camera), kfusion.configuration.nearPlane, kfusion.configuration.farPlane, kfusion.configuration.stepSize(), 0.7 * kfusion.configuration.mu );
    cudaDeviceSynchronize();
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

    printCUDAError();

    glutSwapBuffers();
}

void keys(unsigned char key, int x, int y) {
    switch(key){
    case 'r':
        kfusion.setPose( toMatrix4( trans * rot * preTrans ));
        break;
    case 'c':
        kfusion.Reset();
        kfusion.setPose( toMatrix4( trans * rot * preTrans ));
        break;
    case 'd':
        cout << kfusion.pose << endl;
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

    KFusionConfig config;
    config.volumeSize = make_uint3(128);

    config.combinedTrackAndReduce = false;

    config.iterations[0] = 10;
    config.iterations[1] = 5;
    config.iterations[2] = 5;

    config.camera = make_float4(100, 100, 160, 120);
    config.nearPlane = 0.001;

    config.maxweight = 100;
    config.mu = 0.1;

    config.dist_threshold = 0.2f;
    config.normal_threshold = 0.8f;

    kfusion.Init(config);
    reference.init(config.volumeSize, config.volumeDimensions);

    initVolumeWrap(reference, 1.0f);
    setBoxWrap(reference, make_float3(0.1f,0.1f,0.8f), make_float3(0.9f, 0.9f, 0.9f), -1.0f);
    setBoxWrap(reference, make_float3(0.1f,0.8f,0.1f), make_float3(0.9f, 0.9f, 0.9f), -1.0f);
    setBoxWrap(reference, make_float3(0.8f,0.1f,0.1f), make_float3(0.9f, 0.9f, 0.9f), -1.0f);
    setSphereWrap(reference, make_float3(0.5f), 0.2f, -1.0f);

    kfusion.setPose( toMatrix4( trans * rot * preTrans ));

    vertex.alloc(config.renderSize());
    normal.alloc(config.renderSize());
    depth.alloc(config.renderSize());
    rgb.alloc(config.renderSize());

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE );
    glutInitWindowSize(config.renderSize().x * 3, config.renderSize().y * 3);
    glutCreateWindow("kfusion test");

    glutDisplayFunc(display);
    glutKeyboardFunc(keys);
    glutSpecialFunc(specials);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);

    glutMainLoop();

    return 0;
}
