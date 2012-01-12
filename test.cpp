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
#include "perfstats.h"

using namespace std;
using namespace TooN;

int main(int argc, char ** argv) {

    const bool benchmark = argc > 1 && string(argv[1]) == "-b";

    KFusionConfig config;
    config.nearPlane = 0.001;
#if 1
    config.volumeSize = make_uint3(128);
#endif

    config.iterations[0] = 10;
    config.iterations[1] = 5;
    config.iterations[2] = 5;
    config.maxweight = 100;
    
    config.imageBlock = dim3(20,20);

    config.camera = make_float4(100, 100, 160, 120);

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
    
    Image<float> depth;
    Image<float3> vertex, normal;
    
    depth.init(config.renderSize());
    vertex.init(config.renderSize());
    normal.init(config.renderSize());

    if(printCUDAError()){
        kfusion.Clear();
        reference.release();
        vertex.release();
        normal.release();
        depth.release();
        return 1;
    }

    SE3<float> preTrans(makeVector(0.0, 0, -0.9, 0, 0, 0));
    SE3<float> rot(makeVector(0.0, 0, 0, 0, 0, 0));
    SE3<float> trans(makeVector(0.5, 0.5, 0.5, 0, 0, 0));

    kfusion.setPose( toMatrix4( trans * rot * preTrans ), toMatrix4( (trans * rot * preTrans).inverse() ));

    SE3<float> mapRot(makeVector(0.0, 0, 0, 0, 0, 0));
    SE3<float> mapTrans(makeVector(0.5, 0.5, 0, 0, 0, 0));

    const CVD::ImageRef imageSize(config.renderSize().x, config.renderSize().y);
    
    CVD::Image<CVD::Rgb<float> > vertexImage(imageSize), normalImage(imageSize);
    CVD::Image<float> depthImage(imageSize);
    
    CVD::Image<CVD::Rgb<CVD::byte> > rgb(imageSize);
    
    CVD::GLWindow window(imageSize.dot_times(CVD::ImageRef(3,3)));
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
        raycastWrap(vertex, normal, depth, reference,  toMatrix4( trans * rot * preTrans ) * getInverseCameraMatrix(config.camera), config.nearPlane, config.farPlane, config.stepSize(), 0.01 );
        cudaDeviceSynchronize();
        Stats.sample("raycast ground");

        vertex.get(vertexImage.data());
        normal.get(normalImage.data());
        depth.get(depthImage.data());
        cudaDeviceSynchronize();
        Stats.sample("get ground");

        glRasterPos2i(0,0);
        glDrawPixels(vertexImage);
        glRasterPos2i(imageSize.x, 0);
        glDrawPixels(normalImage);
        glRasterPos2i(imageSize.x * 2, 0);
        glDrawPixels(depthImage);
        Stats.sample("draw ground");

        kfusion.setDeviceDepth( depth.data );
        cudaDeviceSynchronize();
        const double track_start = Stats.sample("process depth");

        if(counter > 1){
            integrate = kfusion.Track();
            cudaDeviceSynchronize();
            Stats.sample("track");
        }

        kfusion.vertex.get(vertexImage.data());
        kfusion.normal.get(normalImage.data());
        renderTrackResult(rgb.data(), kfusion.reduction);
        cudaDeviceSynchronize();
        Stats.sample("render track");

        glRasterPos2i(0,imageSize.y * 1);
        glDrawPixels(vertexImage);
        glRasterPos2i(imageSize.x, imageSize.y * 1);
        glDrawPixels(normalImage);
        glRasterPos2i(2 * imageSize.x, imageSize.y * 1);
        glDrawPixels(rgb);
        Stats.sample("draw track");

        if(integrate){
            kfusion.Integrate();
            cudaDeviceSynchronize();
            Stats.sample("integration");
        }

        raycastWrap(vertex, normal, depth, kfusion.integration,  kfusion.pose * getInverseCameraMatrix(config.camera), config.nearPlane, config.farPlane, config.stepSize(), 0.5 * config.mu );
        cudaDeviceSynchronize();
        Stats.sample("raycast view");
        
        vertex.get(vertexImage.data());
        normal.get(normalImage.data());
        depth.get(depthImage.data());
        cudaDeviceSynchronize();
        Stats.sample("get view");
        
        glRasterPos2i(0,imageSize.y * 2);
        glDrawPixels(vertexImage);
        glRasterPos2i(imageSize.x, imageSize.y * 2);
        glDrawPixels(normalImage);
        glRasterPos2i(imageSize.x * 2, imageSize.y * 2);
        glDrawPixels(depthImage);
        Stats.sample("draw view");
        
        window.swap_buffers();
        events.clear();
        window.get_events(events);

        if(events.key_up.count('p')){
            CVD::Image<CVD::Rgb<CVD::byte> > screen(window.size());
            glReadPixels(screen);
            flipVertical(screen);
            img_save(screen, "screen.jpg");
        }
        if(events.key_up.count('r')){
            kfusion.setPose( toMatrix4( trans * rot * preTrans ), toMatrix4( (trans * rot * preTrans).inverse() ));
        }
        if(events.key_up.count('c')){
            kfusion.Reset();
            kfusion.setPose( toMatrix4( trans * rot * preTrans ), toMatrix4( (trans * rot * preTrans).inverse() ));
        }
        if(events.key_up.count('v')){
            CVD::Image<float> vol = getVolume(kfusion.integration);
            ostringstream sout;
            sout << "volume_" << setw(3) << setfill('0') << counter << ".jpg";
            img_save(vol, sout.str());
            sout.str("");
            vol = getVolume(reference);
            sout << "reference_" << setw(3) << setfill('0') << counter << ".jpg";
            img_save(vol, sout.str());
        }
        if(events.key_up.count('d')){
            cout << kfusion.pose << endl;
        }
        Stats.sample("events");
        Stats.sample("total track", CVD::timer.get_time() - track_start, PerfStats::TIME);
        Stats.sample("total all", CVD::timer.get_time() - start, PerfStats::TIME);
#if 0
       while(events.key_up.empty()){
          window.get_events(events); 
       }
#endif
        if(counter % 30 == 0){
            Stats.print();
            Stats.reset();
            cout << endl;
        }
        
       printCUDAError();
    }

    kfusion.Clear();
    reference.release();
    vertex.release();
    normal.release();
    depth.release();
    
    cudaDeviceReset();
    return 0;
}
