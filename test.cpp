#include "kfusion.h"
#include "helpers.h"

#include <iostream>
#include <sstream>
#include <iomanip>

#include <cvd/glwindow.h>
#include "perfstats.h"

using namespace std;
using namespace TooN;

int main(int argc, char ** argv) {

    const bool benchmark = argc > 1 && string(argv[1]) == "-b";

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

    KFusion kfusion;
    kfusion.Init(config);

    Volume reference;
    reference.init(config.volumeSize, config.volumeDimensions);

    initVolumeWrap(reference, 1.0f);
    setBoxWrap(reference, make_float3(0.1f,0.1f,0.8f), make_float3(0.9f, 0.9f, 0.9f), -1.0f);
    setBoxWrap(reference, make_float3(0.1f,0.8f,0.1f), make_float3(0.9f, 0.9f, 0.9f), -1.0f);
    setBoxWrap(reference, make_float3(0.8f,0.1f,0.1f), make_float3(0.9f, 0.9f, 0.9f), -1.0f);
    setSphereWrap(reference, make_float3(0.5f), 0.2f, -1.0f);

    SE3<float> preTrans(makeVector(0.0, 0, -0.9, 0, 0, 0));
    SE3<float> rot(makeVector(0.0, 0, 0, 0, 0, 0));
    SE3<float> trans(makeVector(0.5, 0.5, 0.5, 0, 0, 0));

    kfusion.setPose( toMatrix4( trans * rot * preTrans ), toMatrix4( (trans * rot * preTrans).inverse() ));

    SE3<float> mapRot(makeVector(0.0, 0, 0, 0, 0, 0));
    SE3<float> mapTrans(makeVector(0.5, 0.5, 0, 0, 0, 0));

    const uint2 imageSize = config.renderSize();

    Image<float3, HostDevice> vertex(config.renderSize()), normal(config.renderSize());
    Image<float, HostDevice> depth(config.renderSize());
    Image<uchar4, HostDevice> rgb(config.renderSize());

    CVD::GLWindow window(CVD::ImageRef(imageSize.x * 3, imageSize.y * 3));
    CVD::GLWindow::EventSummary events;
    glDisable(GL_DEPTH_TEST);

    int counter = 0;
    bool integrate = true;

    while(!events.should_quit() && ( counter < 100 || !benchmark)){
        ++ counter;

        if(events.key_up.count('j'))
            rot *= SE3<float>(makeVector(0.0, 0, 0, 0, 0.1, 0));
        if(events.key_up.count('l'))
            rot *= SE3<float>(makeVector(0.0, 0, 0, 0, -0.1, 0));
        if(events.key_up.count('i'))
            rot *= SE3<float>(makeVector(0.0, 0, 0, -0.1, 0, 0));
        if(events.key_up.count('k'))
            rot *= SE3<float>(makeVector(0.0, 0, 0, 0.1, 0, 0));

        if(benchmark)
            rot *= SE3<float>(makeVector(0.0, 0, 0, 0, 0.02, 0));

        const double start = Stats.start();
        raycastWrap(vertex.getDeviceImage(), normal.getDeviceImage(), depth.getDeviceImage(), reference, toMatrix4( trans * rot * preTrans ) * getInverseCameraMatrix(config.camera), config.nearPlane, config.farPlane, config.stepSize(), 0.01 );
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

        raycastWrap(vertex.getDeviceImage(), normal.getDeviceImage(), depth.getDeviceImage(), kfusion.integration,  kfusion.pose * getInverseCameraMatrix(config.camera), config.nearPlane, config.farPlane, config.stepSize(), 0.5 * config.mu );
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

        window.swap_buffers();
        events.clear();
        window.get_events(events);

        if(events.key_up.count('r')){
            kfusion.setPose( toMatrix4( trans * rot * preTrans ), toMatrix4( (trans * rot * preTrans).inverse() ));
        }
        if(events.key_up.count('c')){
            kfusion.Reset();
            kfusion.setPose( toMatrix4( trans * rot * preTrans ), toMatrix4( (trans * rot * preTrans).inverse() ));
        }
        if(events.key_up.count('d')){
            cout << kfusion.pose << endl;
        }
        Stats.sample("events");
        Stats.sample("total all", Stats.get_time() - start, PerfStats::TIME);

        if(counter % 30 == 0){
            Stats.print();
            Stats.reset();
            cout << endl;
        }

        if(printCUDAError())
            break;
    }

    kfusion.Clear();
    reference.release();

    cudaDeviceReset();
    return 0;
}
