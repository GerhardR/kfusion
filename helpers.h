#ifndef HELPERS_H
#define HELPERS_H

#include "kfusion.h"

#include <TooN/se3.h>
#include <cvd/image.h>

inline Matrix4 toMatrix4( const TooN::SE3<float> & p){
    static TooN::Matrix<4,4,float> I = TooN::Identity; 
    TooN::Matrix<4,4,float> T = p * I;
    Matrix4 R;
    memcpy(R.data, T.get_data_ptr(), 4*4*sizeof(float));
    return R;
}

void renderDepthMap( void *, const Image<float> &, const float nearPlane, const float farPlane ); // scales the depth map from near-far to 0-1
void renderNormalMap( void *, const Image<float3> & ); // renders into a RGB normalmap
void renderLight( void *, const Image<float3> & vertex, const Image<float3> & normal, const float3 light, const float3 ambient); // renders into a grayscale intensity map with lightsource
void renderTrackResult( void * ptr, const Image<TrackData> & data );
void renderVolumeLight( void *, const uint2 size, const Volume & volume, const Matrix4 view, const float nearPlane, const float farPlane, const float largestep, const float3 light, const float3 ambient );

inline CVD::Image<float> getVolume( const Volume & volume){
    CVD::ImageRef size(volume.size.x, volume.size.y);
    const int count = sqrt((float)volume.size.z);
    
    CVD::Image<float> temp(CVD::ImageRef(volume.size.x, volume.size.y * volume.size.z));
    volume.get(temp.data());
    
    const CVD::ImageRef colSize = size.dot_times(CVD::ImageRef(1,count));
    
    CVD::Image<float> to(size * count);
    for(int c = 0; c < count; ++c)
        to.sub_image(size.dot_times(CVD::ImageRef(c, 0)), colSize).copy_from(temp.sub_image(colSize.dot_times(CVD::ImageRef(0,c)), colSize));
        
    for(CVD::Image<float>::iterator p = to.begin(); p != to.end(); ++p){
        *p = (1.f + *p)/2;
    }
    return to;
}

// simple wrappers around the kfusion.h kernel functions
void initVolumeWrap( Volume volume, const float val );
void raycastWrap( Image<float3> pos3D, Image<float3> normal, Image<float> depth, const Volume volume, const Matrix4 view, const float near, const float far, const float step, const float largestep);

// some more wrappers around simple test kernels
void setBoxWrap(Volume volume, const float3 min_corner, const float3 max_corner, const float val );
void setSphereWrap(Volume volume, const float3 center, const float radius, const float val );

#endif // HELPERS_H
