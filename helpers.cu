#include "kfusion.h"
#include "perfstats.h"
#include <iostream>

using namespace std;

PerfStats Stats;

__global__ void setSphere( Volume volume, const float3 center, const float radius, const float val ){
    const float d = length(volume.pos() - center);
    if(d < radius)
        volume.set(make_float2(val, 0.0f));
}

__global__ void setBox( Volume volume, const float3 min_corner, const float3 max_corner, const float val ){
    const float3 p = volume.pos();
    if(min_corner.x < p.x && min_corner.y < p.y && min_corner.z < p.z && 
       p.x < max_corner.x && p.y < max_corner.y && p.z < max_corner.z )
        volume.set(make_float2(val, 0.0f));
}

void initVolumeWrap( Volume volume, const float val ){
    dim3 grid, block;
    computeVolumeConfiguration(grid, block, volume.size);
    initVolume<<<grid, block>>>(volume, make_float2(val, 0.0f));
}

void setBoxWrap(Volume volume, const float3 min_corner, const float3 max_corner, const float val ){
    dim3 grid, block;
    computeVolumeConfiguration(grid, block, volume.size);
    setBox<<<grid, block>>>(volume, min_corner, max_corner, val);
}

void setSphereWrap(Volume volume, const float3 center, const float radius, const float val ){
    dim3 grid, block;
    computeVolumeConfiguration(grid, block, volume.size);
    setSphere<<<grid, block>>>(volume, center, radius, val);
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

void renderNormalMap( void * ptr, const Image<float3> & normal ){
    static Image<uchar3> renderOutput;
    if(renderOutput.size.x != normal.size.x || renderOutput.size.y != normal.size.y){
        renderOutput.release();
        renderOutput.init(normal.size);
    }
    dim3 block(20,20);
    renderNormals<<<divup(normal.size, block), block>>>( renderOutput, normal );
    renderOutput.get(ptr);
}

__global__ void renderLight( Image<uchar3> out, const Image<float3> vertex, const Image<float3> normal, const float3 light, const float3 ambient ){
    if(length(normal.el()) < 0.00001f)
        out.el() = make_uchar3(0, 0, 0);
    else {
        const float3 diff = normalize(light - vertex.el());
        const float dir = fmaxf(dot(normal.el(), diff), 0.f);
        const float3 col = clamp(make_float3(dir) + ambient, 0.f, 1.f) * 255;
        out.el() = make_uchar3(col.x, col.y, col.z);
    }
}

void renderLight( void * ptr, const Image<float3> & vertex, const Image<float3> & normal, const float3 light, const float3 ambient ){
    static Image<uchar3> renderOutput;
    if(renderOutput.size.x != normal.size.x || renderOutput.size.y != normal.size.y){
        renderOutput.release();
        renderOutput.init(normal.size);
    }
    dim3 block(20,20);
    renderLight<<<divup(normal.size, block), block>>>( renderOutput, vertex, normal, light, ambient );
    renderOutput.get(ptr);
}

__global__ void renderDepth( Image<uchar3> out, const Image<float> depth, const float near, const float far){
    const float d = (clamp(depth.el(), near, far) - near) / (far - near);
    out.el() = make_uchar3(d * 255, d * 255, d * 255);
}

void renderDepthMap( void * ptr, const Image<float> & depth, const float nearPlane, const float farPlane ){
    static Image<uchar3> renderOutput;
    if(renderOutput.size.x != depth.size.x || renderOutput.size.y != depth.size.y){
        renderOutput.release();
        renderOutput.init(depth.size);
    }
    dim3 block(20,20);
    renderDepth<<<divup(depth.size, block), block>>>( renderOutput, depth, nearPlane, farPlane );
    renderOutput.get(ptr);
}

__global__ void renderTrack( Image<uchar3> out, const Image<TrackData> data ){
    const uint2 pos = thr2pos2();
    switch(data[pos].result){
    case 1: out[pos] = make_uchar3(128, 128, 128);  // ok
        break;
    case -1: out[pos] = make_uchar3(0, 0, 0);      // no input
        break;
    case -2: out[pos] = make_uchar3(255,0,0);        // not in image
        break;
    case -3:  out[pos] = make_uchar3(0,255,0);        // no correspondence
        break;
    case -4: out[pos] = make_uchar3(0,0,255);        // to far away
        break;
    case -5: out[pos] = make_uchar3(255,255,0);     // wrong normal
        break;
    }
}

void renderTrackResult( void * ptr, const Image<TrackData> & data ){
    static Image<uchar3> renderOutput;
    if(renderOutput.size.x != data.size.x || renderOutput.size.y != data.size.y){
        renderOutput.release();
        renderOutput.init(data.size);
    }
    dim3 block(20,20);
    renderTrack<<<divup(data.size, block), block>>>( renderOutput, data );
    renderOutput.get(ptr);
}

__global__ void raycastLight( Image<uchar3> render, const Volume volume, const Matrix4 view, const float near, const float far, const float step, const float largestep, const float3 light, const float3 ambient){
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
    float tnear = fmaxf(largest_tmin, near);
    float tfar = fminf(smallest_tmax, far);

    if(tnear < tfar) {
        // first walk with largesteps until we found a hit
        if( volume.interp(origin + direction * tnear) > 0){                // ups, if we were already in it, then don't render anything here
            for(float d = tnear + largestep; d < tfar; d += largestep ){
                if(volume.interp(origin + direction * d) < 0){          // got it, now bisect the interval
                    float dp = d - largestep;
                    float dm = d;

                    while(fabsf(dp - dm) > step/10) {    // bisection until we are really small
                        const float middle = (dp + dm) * 0.5f;
                        const float decide = volume.interp(origin + direction * middle);
                        dp = (decide >= 0) ? middle : dp;
                        dm = (decide <= 0) ? middle : dm;
                    }

                    const float3 test = origin + direction * ((dm + dp) * 0.5f);
                    const float3 surfNorm = volume.grad(test);
                    if(length(surfNorm) > 0){
                        const float3 diff = normalize(light - test);
                        const float dir = fmaxf(dot(normalize(surfNorm), diff), 0.f);
                        const float3 col = clamp(make_float3(dir) + ambient, 0.f, 1.f) * 255;
                        render.el() = make_uchar3(col.x, col.y, col.z);
                    } else {
                        render.el() = make_uchar3(0,0,0);
                    }
                    return;
                }
            }
        }
    }
    render.el() = make_uchar3(0,0,0);
}

void renderVolumeLight( void * ptr, const uint2 size, const Volume & volume, const Matrix4 view, const float nearPlane, const float farPlane, const float largestep, const float3 light, const float3 ambient ){
    static Image<uchar3> renderOutput;
    if(renderOutput.size.x != size.x || renderOutput.size.y != size.y){
        renderOutput.release();
        renderOutput.init(size);
    }
    dim3 block(16,16);
    raycastLight<<<divup(renderOutput.size, block), block>>>( renderOutput,  volume, view, nearPlane, farPlane, 0.5f * volume.dim.x/volume.size.x, largestep, light, ambient );
    renderOutput.get(ptr);
}

void raycastWrap( Image<float3> pos3D, Image<float3> normal, Image<float> depth, const Volume volume, const Matrix4 view, const float near, const float far, const float step, const float largestep){
    dim3 block(16,16);
    raycast<<<divup(pos3D.size, block), block>>>(pos3D, normal, depth, volume, view, near, far, step, largestep);
}
