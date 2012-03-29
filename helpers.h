#ifndef HELPERS_H
#define HELPERS_H

#include "kfusion.h"
#include "glproxy.h"

#include <TooN/se3.h>
#include <cstring>


inline Matrix4 toMatrix4( const TooN::SE3<float> & p){
    static TooN::Matrix<4,4,float> I = TooN::Identity; 
    TooN::Matrix<4,4,float> T = p * I;
    Matrix4 R;
    memcpy(R.data, T.get_data_ptr(), 4*4*sizeof(float));
    return R;
}

void renderDepthMap( Image<uchar3> out, const Image<float> &, const float nearPlane, const float farPlane ); // scales the depth map from near-far to 0-1
void renderNormalMap( Image<uchar3> out, const Image<float3> & ); // renders into a RGB normalmap
void renderLight( Image<uchar4> out, const Image<float3> & vertex, const Image<float3> & normal, const float3 light, const float3 ambient); // renders into a grayscale intensity map with lightsource
void renderTrackResult( Image<uchar4> out, const Image<TrackData> & data );
void renderVolumeLight( Image<uchar4> out, const Volume & volume, const Matrix4 view, const float nearPlane, const float farPlane, const float largestep, const float3 light, const float3 ambient );

// simple wrappers around the kfusion.h kernel functions
void initVolumeWrap( Volume volume, const float val );
void raycastWrap( Image<float3> pos3D, Image<float3> normal, Image<float> depth, const Volume volume, const Matrix4 view, const float nearPlane, const float farPlane, const float step, const float largestep);

// some more wrappers around simple test kernels
void setBoxWrap(Volume volume, const float3 min_corner, const float3 max_corner, const float val );
void setSphereWrap(Volume volume, const float3 center, const float radius, const float val );

template <typename T> struct gl;

template<> struct gl<unsigned char> {
    static const int format=GL_LUMINANCE;
    static const int type  =GL_UNSIGNED_BYTE;
};

template<> struct gl<uchar3> {
    static const int format=GL_RGB;
    static const int type  =GL_UNSIGNED_BYTE;
};

template<> struct gl<uchar4> {
    static const int format=GL_RGBA;
    static const int type  =GL_UNSIGNED_BYTE;
};

 template<> struct gl<float> {
    static const int format=GL_LUMINANCE;
    static const int type  =GL_FLOAT;
};

 template<> struct gl<float3> {
    static const int format=GL_RGB;
    static const int type  =GL_FLOAT;
};

template <typename T, typename A>
inline void glDrawPixels( const Image<T, A> & i ){
    ::glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    ::glPixelStorei(GL_UNPACK_ROW_LENGTH, i.size.x);
    ::glDrawPixels(i.size.x, i.size.y, gl<T>::format, gl<T>::type, i.data());
}

template <typename T>
inline void glDrawPixels( const Image<T, PBO> & i ){
    ::glBindBuffer(GL_PIXEL_UNPACK_BUFFER, i.id);
    ::glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    ::glPixelStorei(GL_UNPACK_ROW_LENGTH, i.size.x);
    ::glDrawPixels(i.size.x, i.size.y, gl<T>::format, gl<T>::type, 0);
    ::glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

#endif // HELPERS_H
