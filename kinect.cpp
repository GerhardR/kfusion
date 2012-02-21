#undef isnan
#undef isfinite

#include "kfusion.h"
#include "helpers.h"

#include <iostream>
#include <sstream>
#include <iomanip>

#include <cvd/glwindow.h>
#include "perfstats.h"

using namespace std;
using namespace TooN;

#include <libfreenect/libfreenect.h>

freenect_context *f_ctx;
freenect_device *f_dev;
bool gotDepth;

void depth_cb(freenect_device *dev, void *v_depth, uint32_t timestamp)
{
    gotDepth = true;
}

int InitKinect( uint16_t * buffer ){
    if (freenect_init(&f_ctx, NULL) < 0) {
        cout << "freenect_init() failed" << endl;
        return 1;
    }

    freenect_set_log_level(f_ctx, FREENECT_LOG_WARNING);
    freenect_select_subdevices(f_ctx, (freenect_device_flags)(FREENECT_DEVICE_MOTOR | FREENECT_DEVICE_CAMERA));

    int nr_devices = freenect_num_devices (f_ctx);
    cout << "Number of devices found: " << nr_devices << endl;

    if (nr_devices < 1)
        return 1;

    if (freenect_open_device(f_ctx, &f_dev, 0) < 0) {
        cout << "Could not open device" << endl;
        return 1;
    }

    freenect_set_depth_callback(f_dev, depth_cb);
    freenect_set_depth_mode(f_dev, freenect_find_depth_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_DEPTH_11BIT));
    freenect_set_depth_buffer(f_dev, buffer);
    freenect_start_depth(f_dev);

    gotDepth = false;

    return 0;
}

void CloseKinect(){
    freenect_stop_depth(f_dev);
    freenect_close_device(f_dev);
    freenect_shutdown(f_ctx);
}

void DepthFrameKinect() {
    while (!gotDepth && freenect_process_events(f_ctx) >= 0){
    }
    gotDepth = false;
}

int main(int argc, char ** argv) {
    const float size = (argc > 1) ? atof(argv[1]) : 2.f;

    KFusionConfig config;

    // it is enough now to set the volume resolution once.
    // everything else is derived from that.
    // config.volumeSize = make_uint3(64);
    config.volumeSize = make_uint3(128);
    // config.volumeSize = make_uint3(256);

    // these are physical dimensions in meters
    config.volumeDimensions = make_float3(size);
    config.nearPlane = 0.4f;
    config.farPlane = 5.0f;
    config.mu = 0.1;
    config.combinedTrackAndReduce = false;

    config.camera =  make_float4(297.12732, 296.24240, 169.89365, 121.25151);

    config.iterations[0] = 10;
    config.iterations[1] = 5;
    config.iterations[2] = 4;

    config.dist_threshold = (argc > 2 ) ? atof(argv[2]) : config.dist_threshold;
    config.normal_threshold = (argc > 3 ) ? atof(argv[3]) : config.normal_threshold;

    const uint2 imageSize = config.renderSize();

    CVD::GLWindow window(CVD::ImageRef(imageSize.x * 2, imageSize.y * 2));
    CVD::GLWindow::EventSummary events;
    glDisable(GL_DEPTH_TEST);

    KFusion kfusion;
    kfusion.Init(config);
    if(printCUDAError()){
        kfusion.Clear();
        cudaDeviceReset();
        return 1;
    }

    Image<uchar4, HostDevice> lightScene(imageSize), depth(imageSize), lightModel(imageSize);
    Image<uint16_t, HostDevice> depthImage(make_uint2(640, 480));

    const float3 light = make_float3(1.0, -2, 1.0);
    const float3 ambient = make_float3(0.1, 0.1, 0.1);
    const float4 renderCamera = make_float4(297.12732, 296.24240, 169.89365+160, 121.25151);

    SE3<float> initPose(makeVector(size/2, size/2, 0, 0, 0, 0));

    kfusion.setPose(toMatrix4(initPose), toMatrix4(initPose.inverse()));

    if(InitKinect(depthImage.data()))
        return 1;

    bool integrate = true;
    bool reset = true;

    int counter = 0;
    while(!events.should_quit()){
        glClear( GL_COLOR_BUFFER_BIT );
        const double startFrame = Stats.start();
        ++counter;

        DepthFrameKinect();
        const double startProcessing = Stats.sample("kinect");

        kfusion.setKinectDeviceDepth(depthImage.getDeviceImage());
        Stats.sample("raw to cooked");

        integrate = kfusion.Track();
        Stats.sample("track");

        if(integrate || reset ){
            kfusion.Integrate();
            Stats.sample("integrate");
            reset = false;
        }

        renderLight( lightModel.getDeviceImage(), kfusion.vertex, kfusion.normal, light, ambient);
        renderLight( lightScene.getDeviceImage(), kfusion.inputVertex[0], kfusion.inputNormal[0], light, ambient );
        renderTrackResult( depth.getDeviceImage(), kfusion.reduction );
        cudaDeviceSynchronize();

        Stats.sample("render");

        glClear( GL_COLOR_BUFFER_BIT );
        glRasterPos2i(0,imageSize.y * 0);
        glDrawPixels(lightScene);
        glRasterPos2i(imageSize.x, imageSize.y * 0);
        glDrawPixels(depth);
        glRasterPos2i(0,imageSize.y * 1);
        glDrawPixels(lightModel);
        Stats.sample("draw");

        window.swap_buffers();
        events.clear();
        window.get_events(events);

        if(events.key_up.count('c')){
            kfusion.Reset();
            kfusion.setPose(toMatrix4(initPose), toMatrix4(initPose.inverse()));
            reset = true;
        }

        if(counter % 50 == 0){
            Stats.print();
            Stats.reset();
            cout << endl;
        }

        if(printCUDAError())
            break;

        const double endProcessing = Stats.sample("events");
        Stats.sample("total", endProcessing - startFrame, PerfStats::TIME);
        Stats.sample("total_proc", endProcessing - startProcessing, PerfStats::TIME);
    }

    CloseKinect();
    kfusion.Clear();

    cudaDeviceReset();
    return 0;
}
