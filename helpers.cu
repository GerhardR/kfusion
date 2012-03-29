#include "kfusion.h"
#include "perfstats.h"
#include <iostream>

using namespace std;

PerfStats Stats;

__global__ void setSphere( Volume volume, const float3 center, const float radius, const float val ){
    uint3 pos = make_uint3(thr2pos2());
    for(pos.z = 0; pos.z < volume.size.z; ++pos.z) {
        const float d = length(volume.pos(pos) - center);
        if(d < radius)
            volume.set(pos, make_float2(val, 0.0f));
    }
}

__global__ void setBox( Volume volume, const float3 min_corner, const float3 max_corner, const float val ){
    uint3 pos = make_uint3(thr2pos2());
    for(pos.z = 0; pos.z < volume.size.z; ++pos.z) {
        const float3 p = volume.pos(pos);
        if(min_corner.x < p.x && min_corner.y < p.y && min_corner.z < p.z && 
           p.x < max_corner.x && p.y < max_corner.y && p.z < max_corner.z )
            volume.set(pos, make_float2(val, 0.0f));
    }
}

void initVolumeWrap( Volume volume, const float val ){
    dim3 block(32,16);
    initVolume<<<divup(dim3(volume.size.x, volume.size.y), block), block>>>(volume, make_float2(val, 0.0f));
}

void setBoxWrap(Volume volume, const float3 min_corner, const float3 max_corner, const float val ){
    dim3 block(32,16);
    setBox<<<divup(dim3(volume.size.x, volume.size.y), block), block>>>(volume, min_corner, max_corner, val);
}

void setSphereWrap(Volume volume, const float3 center, const float radius, const float val ){
    dim3 block(32,16);
    setSphere<<<divup(dim3(volume.size.x, volume.size.y), block), block>>>(volume, center, radius, val);
}

__global__ void renderNormals( Image<uchar3> out, const Image<float3> in ){
    float3 n = in.el();
    if(length(n) < 0.000001f)
        out.el() = make_uchar3(0,0,0);
    else {
        n = normalize(n);
        out.el() = make_uchar3(n.x*128 + 128, n.y*128+128, n.z*128+128);
    }
}

void renderNormalMap( Image<uchar3> out, const Image<float3> & normal ){
    dim3 block(20,20);
    renderNormals<<<divup(normal.size, block), block>>>( out, normal );
}

__global__ void renderLightKernel( Image<uchar4> out, const Image<float3> vertex, const Image<float3> normal, const float3 light, const float3 ambient ){
    if(normal.el().z == -2)
        out.el() = make_uchar4(0,0,0,0);
    else {
        const float3 diff = normalize(light - vertex.el());
        const float dir = fmaxf(dot(normal.el(), diff), 0.f);
        const float3 col = clamp(make_float3(dir) + ambient, 0.f, 1.f) * 255;
        out.el() = make_uchar4(col.x, col.y, col.z, 1);
    }
}

void renderLight( Image<uchar4> out, const Image<float3> & vertex, const Image<float3> & normal, const float3 light, const float3 ambient ){
    dim3 block(32,16);
    renderLightKernel<<<divup(normal.size, block), block>>>( out, vertex, normal, light, ambient );
}

__global__ void renderDepth( Image<uchar3> out, const Image<float> depth, const float nearPlane, const float farPlane){
    const float d = (clamp(depth.el(), nearPlane, farPlane) - nearPlane) / (farPlane - nearPlane);
    out.el() = make_uchar3(d * 255, d * 255, d * 255);
}

void renderDepthMap( Image<uchar3> out, const Image<float> & depth, const float nearPlane, const float farPlane ){
    dim3 block(20,20);
    renderDepth<<<divup(depth.size, block), block>>>( out, depth, nearPlane, farPlane );
}

__global__ void renderTrack( Image<uchar4> out, const Image<TrackData> data ){
    const uint2 pos = thr2pos2();
    switch(data[pos].result){
    case 1: out[pos] = make_uchar4(128, 128, 128,0);  // ok
        break;
    case -1: out[pos] = make_uchar4(0, 0, 0,0);      // no input
        break;
    case -2: out[pos] = make_uchar4(255,0,0,0);        // not in image
        break;
    case -3:  out[pos] = make_uchar4(0,255,0,0);        // no correspondence
        break;
    case -4: out[pos] = make_uchar4(0,0,255,0);        // to far away
        break;
    case -5: out[pos] = make_uchar4(255,255,0,0);     // wrong normal
        break;
    }
}

void renderTrackResult( Image<uchar4> out, const Image<TrackData> & data ){
    dim3 block(20,20);
    renderTrack<<<divup(out.size, block), block>>>( out, data );
}

__global__ void raycastLight( Image<uchar4> render, const Volume volume, const Matrix4 view, const float nearPlane, const float farPlane, const float step, const float largestep, const float3 light, const float3 ambient){
    const uint2 pos = thr2pos2();

    float3 origin = view.get_translation();
    float3 direction = rotate(view, make_float3(pos.x, pos.y, 1.f));

    // intersect ray with a box
    // http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm
    // compute intersection of ray with all six bbox planes
    float3 invR = make_float3(1.0f) / direction;
    float3 tbot = -invR * origin;
    float3 ttop = invR * (volume.dim - origin);

    // re-order intersections to find smallest and largest on each axis
    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

    // check against near and far plane
    float tnear = fmaxf(largest_tmin, nearPlane);
    float tfar = fminf(smallest_tmax, farPlane);

    if(tnear < tfar) {
        // first walk with largesteps until we found a hit
        float t = tnear;
        float stepsize = largestep;
        float f_t = volume.interp(origin + direction * t);
        float f_tt = 0;
        if( f_t > 0){     // ups, if we were already in it, then don't render anything here
            for(; t < tfar; t += stepsize){
                f_tt = volume.interp(origin + direction * t);
                if(f_tt < 0)                               // got it, jump out of inner loop
                    break;
                if(f_tt < 0.8f)                            // coming closer, reduce stepsize
                    stepsize = step;
                f_t = f_tt;
            }
            if(f_tt < 0){                               // got it, calculate accurate intersection
                t = t + stepsize * f_tt / (f_t - f_tt);
                const float3 test = origin + direction * t;
                const float3 surfNorm = volume.grad(test);
                if(length(surfNorm) > 0){
                    const float3 diff = normalize(light - test);
                    const float dir = fmaxf(dot(normalize(surfNorm), diff), 0.f);
                    const float3 col = clamp(make_float3(dir) + ambient, 0.f, 1.f) * 255;
                    render.el() = make_uchar4(col.x, col.y, col.z,0);
                } else {
                    render.el() = make_uchar4(0,0,0,0);
                }
                return;
            }
        }
    }
    render.el() = make_uchar4(0,0,0,0);
}

void renderVolumeLight( Image<uchar4> out, const Volume & volume, const Matrix4 view, const float nearPlane, const float farPlane, const float largestep, const float3 light, const float3 ambient ){
    dim3 block(16,16);
    raycastLight<<<divup(out.size, block), block>>>( out,  volume, view, nearPlane, farPlane, volume.dim.x/volume.size.x, largestep, light, ambient );
}

void raycastWrap( Image<float3> pos3D, Image<float3> normal, Image<float> depth, const Volume volume, const Matrix4 view, const float nearPlane, const float farPlane, const float step, const float largestep){
    dim3 block(16,16);
    raycast<<<divup(pos3D.size, block), block>>>(pos3D, normal, depth, volume, view, nearPlane, farPlane, step, largestep);

    printCUDAError(__LINE__,__FUNCTION__);
}
