#undef isnan
#undef isfinite

#include "kfusion.h"
#include "helpers.h"

#include <iostream>
#include <sstream>
#include <iomanip>

#include <cvd/image_io.h>
#include <cvd/colourspaces.h>
#include <cvd/glwindow.h>
#include <cvd/gl_helpers.h>
#include <cvd/vision.h>
#include <cvd/thread.h>
#include "perfstats.h"

using namespace std;
using namespace TooN;

#include <libfreenect/libfreenect.h>

freenect_context *f_ctx;
freenect_device *f_dev;
bool gotDepth;

CVD::Image<uint16_t> depthImage(CVD::ImageRef(640,480));

void depth_cb(freenect_device *dev, void *v_depth, uint32_t timestamp)
{
    depthImage.copy_from(CVD::BasicImage<uint16_t>((uint16_t*)v_depth, depthImage.size()));
    gotDepth = true;
}

int InitKinect(){
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
    freenect_start_depth(f_dev);

    return 0;
}

void CloseKinect(){
    freenect_stop_depth(f_dev);
    freenect_close_device(f_dev);
    freenect_shutdown(f_ctx);
}

CVD::Image<uint16_t> DepthFrameKinect() {
    gotDepth = false;
    while (!gotDepth && freenect_process_events(f_ctx) >= 0){
        CVD::Thread::sleep(0);
    }
    
    return depthImage;
}

int main(int argc, char ** argv) {
    const float size = (argc > 1) ? atof(argv[1]) : 2.f;

    KFusionConfig config;

    // it is enough now to set the volume resolution once. 
    // everything else is derived from that.
    // config.volumeSize = make_uint3(64);
    config.volumeSize = make_uint3(128);
    //config.volumeSize = make_uint3(256);
	config.maxweight = 100.f; // default 100
	config.fullFrame = true;

#if 0
    config.radius = (argc > 1 ) ? atoi(argv[1]) : 3;
    config.delta = (argc > 2 ) ? atof(argv[2]) : 4.0f;
    config.e_delta = (argc > 3 ) ? atof(argv[3]) : 0.1f;
#endif
  
    // these are physical dimensions in meters
    config.volumeDimensions = make_float3(size);
    config.nearPlane = 0.4f;
    config.farPlane = 5.0f;
    config.mu = 0.1;
    
    config.camera =  make_float4(297.12732, 296.24240, 169.89365, 121.25151);
    
    config.iterations[0] = 10;
    config.iterations[1] = 5;
    config.iterations[2] = 4;
    
    config.dist_threshold = (argc > 2 ) ? atof(argv[2]) : config.dist_threshold;
    config.normal_threshold = (argc > 3 ) ? atof(argv[3]) : config.normal_threshold;
    
    KFusion kfusion;
    kfusion.Init(config);
    if(printCUDAError()){
        kfusion.Clear();
        cudaDeviceReset();
        return 1;
    }
    
    const CVD::ImageRef imageSize(config.renderSize().x, config.renderSize().y);
    
    CVD::GLWindow window(imageSize.dot_times(CVD::ImageRef(2,2)));
    CVD::GLWindow::EventSummary events;
    glDisable(GL_DEPTH_TEST);

    CVD::Image<CVD::Rgb<CVD::byte> > lightScene(imageSize), depth(imageSize), lightWide(CVD::ImageRef(imageSize.x * 2, imageSize.y));
    
    const float3 light = make_float3(1.0, -2, 1.0);
    const float3 ambient = make_float3(0.1, 0.1, 0.1);
    const float4 renderCamera = make_float4(297.12732, 296.24240, 169.89365+160, 121.25151);

    SE3<float> initPose(makeVector(size/2, size/2, 0, 0, 0, 0));

    kfusion.setPose(toMatrix4(initPose), toMatrix4(initPose.inverse()));

    if(InitKinect())
        return 1;

    bool integrate = true;
    bool reset = true;

    int counter = 0;
    while(!events.should_quit()){
        glClear( GL_COLOR_BUFFER_BIT );
        Stats.start();
        ++counter;

        CVD::Image<uint16_t> img = DepthFrameKinect();
        const double startProcessing = Stats.sample("kinect");

        kfusion.setKinectDepth(img.data());
        Stats.sample("raw to cooked");

        integrate = kfusion.Track();
        Stats.sample("track");
        
        if(integrate || reset ){
            kfusion.Integrate();
            Stats.sample("integrate");
            reset = false;
        }

        renderVolumeLight(lightWide.data(), make_uint2(lightWide.size().x, lightWide.size().y), kfusion.integration, kfusion.pose * getInverseCameraMatrix(renderCamera), 0.4f, 5.0f, kfusion.configuration.mu * 0.7, light, ambient );
        Stats.sample("renderview");

        renderLight( lightScene.data(), kfusion.inputVertex[0], kfusion.inputNormal[0], light, ambient );
        renderTrackResult( depth.data(), kfusion.reduction );
        Stats.sample("render");

        glRasterPos2i(0,imageSize.y * 0);
        glDrawPixels(lightScene);
        glRasterPos2i(imageSize.x, imageSize.y * 0);
        glDrawPixels(depth);
        glRasterPos2i(0,imageSize.y * 1);
        glDrawPixels(lightWide);
        Stats.sample("draw");
        
        window.swap_buffers();
        events.clear();
        window.get_events(events);

        if(events.key_up.count('p')){
            CVD::Image<CVD::Rgb<CVD::byte> > screen(window.size());
            glReadPixels(screen);
            flipVertical(screen);
            img_save(screen, "kinect_screen.jpg");
        }
        if(events.key_up.count('c')){
            kfusion.Reset();
            kfusion.setPose(toMatrix4(initPose), toMatrix4(initPose.inverse()));
            reset = true;
        }        
        if(events.key_up.count('v')){
            CVD::Image<float> vol = getVolume(kfusion.integration);
            ostringstream sout;
            sout << "volume_" << setw(3) << setfill('0') << counter << ".jpg";
            img_save(vol, sout.str());
        }
        const double endProcessing = Stats.sample("events");
        
        Stats.sample("total", endProcessing - startProcessing, PerfStats::TIME);
        if(counter % 50 == 0){
            Stats.print();
            Stats.reset();
            cout << endl;
        }
        
        if(printCUDAError())
            break;
    }
    
    CloseKinect();
    kfusion.Clear();
    cudaDeviceReset();
    return 0;
}
