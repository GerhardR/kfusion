/*
Copyright (c) 2011-2013 Gerhard Reitmayr, TU Graz

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include "kfusion.h"

#undef isnan
#undef isfinite

#include <iostream>

#include <TooN/TooN.h>
#include <TooN/se3.h>
#include <TooN/GR_SVD.h>

#define INVALID -2   // this is used to mark invalid entries in normal or vertex maps

using namespace std;

__global__ void initVolume( Volume volume, const float2 val ){
    uint3 pos = make_uint3(thr2pos2());
    for(pos.z = 0; pos.z < volume.size.z; ++pos.z)
        volume.set(pos, val);
}

__global__ void raycast( Image<float3> pos3D, Image<float3> normal, const Volume volume, const Matrix4 view, const float nearPlane, const float farPlane, const float step, const float largestep){
    const uint2 pos = thr2pos2();

    const float4 hit = raycast( volume, pos, view, nearPlane, farPlane, step, largestep );
    if(hit.w > 0){
        pos3D[pos] = make_float3(hit);
        float3 surfNorm = volume.grad(make_float3(hit));
        if(length(surfNorm) == 0){
            normal[pos].x = INVALID;
        } else {
            normal[pos] = normalize(surfNorm);
        }
    } else {
        pos3D[pos] = make_float3(0);
        normal[pos] = make_float3(INVALID, 0, 0);
    }
}

__forceinline__ __device__ float sq( const float x ){
    return x*x;
}

__global__ void integrate( Volume vol, const Image<float> depth, const Matrix4 invTrack, const Matrix4 K, const float mu, const float maxweight){
    uint3 pix = make_uint3(thr2pos2());
    float3 pos = invTrack * vol.pos(pix);
    float3 cameraX = K * pos;
    const float3 delta = rotate(invTrack, make_float3(0,0, vol.dim.z / vol.size.z));
    const float3 cameraDelta = rotate(K, delta);

    for(pix.z = 0; pix.z < vol.size.z; ++pix.z, pos += delta, cameraX += cameraDelta){
       if(pos.z < 0.0001f) // some near plane constraint
            continue;
        const float2 pixel = make_float2(cameraX.x/cameraX.z + 0.5f, cameraX.y/cameraX.z + 0.5f);
        if(pixel.x < 0 || pixel.x > depth.size.x-1 || pixel.y < 0 || pixel.y > depth.size.y-1)
            continue;
        const uint2 px = make_uint2(pixel.x, pixel.y);
        if(depth[px] == 0)
            continue;
        const float diff = (depth[px] - cameraX.z) * sqrt(1+sq(pos.x/pos.z) + sq(pos.y/pos.z));
        if(diff > -mu){
            const float sdf = fminf(1.f, diff/mu);
            float2 data = vol[pix];
            data.x = clamp((data.y*data.x + sdf)/(data.y + 1), -1.f, 1.f);
            data.y = fminf(data.y+1, maxweight);
            vol.set(pix, data);
        }
    }
}

__global__ void depth2vertex( Image<float3> vertex, const Image<float> depth, const Matrix4 invK ){
    const uint2 pixel = thr2pos2();
    if(pixel.x >= depth.size.x || pixel.y >= depth.size.y )
        return;

    if(depth[pixel] > 0){
        vertex[pixel] = depth[pixel] * (rotate(invK, make_float3(pixel.x, pixel.y, 1.f)));
    } else {
        vertex[pixel] = make_float3(0);
    }
}

__global__ void vertex2normal( Image<float3> normal, const Image<float3> vertex ){
    const uint2 pixel = thr2pos2();
    if(pixel.x >= vertex.size.x || pixel.y >= vertex.size.y )
        return;

    const float3 left = vertex[make_uint2(max(int(pixel.x)-1,0), pixel.y)];
    const float3 right = vertex[make_uint2(min(pixel.x+1,vertex.size.x-1), pixel.y)];
    const float3 up = vertex[make_uint2(pixel.x, max(int(pixel.y)-1,0))];
    const float3 down = vertex[make_uint2(pixel.x, min(pixel.y+1,vertex.size.y-1))];

    if(left.z == 0 || right.z == 0 || up.z == 0 || down.z == 0) {
         normal[pixel].x = INVALID;
         return;
    }

    const float3 dxv = right - left;
    const float3 dyv = down - up;
    normal[pixel] = normalize(cross(dyv, dxv)); // switched dx and dy to get factor -1
}

template <int HALFSAMPLE>
__global__ void mm2meters( Image<float> depth, const Image<ushort> in ){
    const uint2 pixel = thr2pos2();
    depth[pixel] = in[pixel * (HALFSAMPLE+1)] / 1000.0f;
}

//column pass using coalesced global memory reads
__global__ void bilateral_filter(Image<float> out, const Image<float> in, const Image<float> gaussian, const float e_d, const int r) {
    const uint2 pos = thr2pos2();

    if(in[pos] == 0){
        out[pos] = 0;
        return;
    }

    float sum = 0.0f;
    float t = 0.0f;
    const float center = in[pos];

    for(int i = -r; i <= r; ++i) {
        for(int j = -r; j <= r; ++j) {
            const float curPix = in[make_uint2(clamp(pos.x + i, 0u, in.size.x-1), clamp(pos.y + j, 0u, in.size.y-1))];
            if(curPix > 0){
                const float mod = sq(curPix - center);
                const float factor = gaussian[make_uint2(i + r, 0)] * gaussian[make_uint2(j + r, 0)] * __expf(-mod / (2 * e_d * e_d));
                t += factor * curPix;
                sum += factor;
            }
        }
    }
    out[pos] = t / sum;
}

// filter and halfsample
__global__ void halfSampleRobust( Image<float> out, const Image<float> in, const float e_d, const int r){
    const uint2 pixel = thr2pos2();
    const uint2 centerPixel = 2 * pixel;

    if(pixel.x >= out.size.x || pixel.y >= out.size.y )
        return;

    float sum = 0.0f;
    float t = 0.0f;
    const float center = in[centerPixel];
    for(int i = -r + 1; i <= r; ++i){
        for(int j = -r + 1; j <= r; ++j){
            float current = in[make_uint2(clamp(make_int2(centerPixel.x + j, centerPixel.y + i), make_int2(0), make_int2(in.size.x - 1, in.size.y - 1)))]; // TODO simplify this!
            if(fabsf(current - center) < e_d){
                sum += 1.0f;
                t += current;
            }
        }
    }
    out[pixel] = t / sum;
}

__global__ void generate_gaussian(Image<float> out, float delta, int radius) {
    int x = threadIdx.x - radius;
    out[make_uint2(threadIdx.x,0)] = __expf(-(x * x) / (2 * delta * delta));
}

__global__ void track( Image<TrackData> output, const Image<float3> inVertex, const Image<float3> inNormal , const Image<float3> refVertex, const Image<float3> refNormal, const Matrix4 Ttrack, const Matrix4 view, const float dist_threshold, const float normal_threshold ) {
    const uint2 pixel = thr2pos2();
    if(pixel.x >= inVertex.size.x || pixel.y >= inVertex.size.y )
        return;

    TrackData & row = output[pixel];

    if(inNormal[pixel].x == INVALID ){
        row.result = -1;
        return;
    }

    const float3 projectedVertex = Ttrack * inVertex[pixel];
    const float3 projectedPos = view * projectedVertex;
    const float2 projPixel = make_float2( projectedPos.x / projectedPos.z + 0.5f, projectedPos.y / projectedPos.z + 0.5f);

    if(projPixel.x < 0 || projPixel.x > refVertex.size.x-1 || projPixel.y < 0 || projPixel.y > refVertex.size.y-1 ){
        row.result = -2;
        return;
    }

    const uint2 refPixel = make_uint2(projPixel.x, projPixel.y);
    const float3 referenceNormal = refNormal[refPixel];

    if(referenceNormal.x == INVALID){
        row.result = -3;
        return;
    }

    const float3 diff = refVertex[refPixel] - projectedVertex;
    const float3 projectedNormal = rotate(Ttrack, inNormal[pixel]);

    if(length(diff) > dist_threshold ){
        row.result = -4;
        return;
    }
    if(dot(projectedNormal, referenceNormal) < normal_threshold){
        row.result = -5;
        return;
    }

    row.result = 1;
    row.error = dot(referenceNormal, diff);
    ((float3 *)row.J)[0] = referenceNormal;
    ((float3 *)row.J)[1] = cross(projectedVertex, referenceNormal);
}

__global__ void reduce( float * out, const Image<TrackData> J, const uint2 size){
    __shared__ float S[112][32]; // this is for the final accumulation
    const uint sline = threadIdx.x;

    float sums[32];
    float * jtj = sums + 7;
    float * info = sums + 28;

    for(uint i = 0; i < 32; ++i)
        sums[i] = 0;

    for(uint y = blockIdx.x; y < size.y; y += gridDim.x){
        for(uint x = sline; x < size.x; x += blockDim.x ){
            const TrackData & row = J[make_uint2(x, y)];
            if(row.result < 1){
                info[1] += row.result == -4 ? 1 : 0;
                info[2] += row.result == -5 ? 1 : 0;
                info[3] += row.result > -4 ? 1 : 0;
                continue;
            }

            // Error part
            sums[0] += row.error * row.error;

            // JTe part
            for(int i = 0; i < 6; ++i)
                sums[i+1] += row.error * row.J[i];

            // JTJ part, unfortunatly the double loop is not unrolled well...
            jtj[0] += row.J[0] * row.J[0];
            jtj[1] += row.J[0] * row.J[1];
            jtj[2] += row.J[0] * row.J[2];
            jtj[3] += row.J[0] * row.J[3];
            jtj[4] += row.J[0] * row.J[4];
            jtj[5] += row.J[0] * row.J[5];

            jtj[6] += row.J[1] * row.J[1];
            jtj[7] += row.J[1] * row.J[2];
            jtj[8] += row.J[1] * row.J[3];
            jtj[9] += row.J[1] * row.J[4];
           jtj[10] += row.J[1] * row.J[5];

           jtj[11] += row.J[2] * row.J[2];
           jtj[12] += row.J[2] * row.J[3];
           jtj[13] += row.J[2] * row.J[4];
           jtj[14] += row.J[2] * row.J[5];

           jtj[15] += row.J[3] * row.J[3];
           jtj[16] += row.J[3] * row.J[4];
           jtj[17] += row.J[3] * row.J[5];

           jtj[18] += row.J[4] * row.J[4];
           jtj[19] += row.J[4] * row.J[5];

           jtj[20] += row.J[5] * row.J[5];

           // extra info here
           info[0] += 1;
        }
    }

    for(int i = 0; i < 32; ++i) // copy over to shared memory
        S[sline][i] = sums[i];

    __syncthreads();            // wait for everyone to finish

    if(sline < 32){ // sum up columns and copy to global memory in the final 32 threads
        for(unsigned i = 1; i < blockDim.x; ++i)
            S[0][sline] += S[i][sline];
        out[sline+blockIdx.x*32] = S[0][sline];
    }
}

__global__ void trackAndReduce( float * out, const Image<float3> inVertex, const Image<float3> inNormal , const Image<float3> refVertex, const Image<float3> refNormal, const Matrix4 Ttrack, const Matrix4 view, const float dist_threshold, const float normal_threshold ){
    __shared__ float S[112][32]; // this is for the final accumulation
    const uint sline = threadIdx.x;

    float sums[32];
    float * jtj = sums + 7;
    float * info = sums + 28;

    for(uint i = 0; i < 32; ++i)
        sums[i] = 0;

    float J[6];

    for(uint y = blockIdx.x; y < inVertex.size.y; y += gridDim.x){
        for(uint x = sline; x < inVertex.size.x; x += blockDim.x ){
            const uint2 pixel = make_uint2(x,y);

            if(inNormal[pixel].x == INVALID){
                continue;
            }

            const float3 projectedVertex = Ttrack * inVertex[pixel];
            const float3 projectedPos = view * projectedVertex;
            const float2 projPixel = make_float2( projectedPos.x / projectedPos.z + 0.5f, projectedPos.y / projectedPos.z + 0.5f);

            if(projPixel.x < 0 || projPixel.x > refVertex.size.x-1 || projPixel.y < 0 || projPixel.y > refVertex.size.y-1 ){
                info[3] += 1;
                continue;
            }

            const uint2 refPixel = make_uint2(projPixel.x, projPixel.y);

            if(refNormal[refPixel].x == INVALID){
                info[3] += 1;
                continue;
            }

            const float3 referenceNormal = refNormal[refPixel];
            const float3 diff = refVertex[refPixel] - projectedVertex;
            const float3 projectedNormal = rotate(Ttrack, inNormal[pixel]);

            if(length(diff) > dist_threshold ){
                info[1] += 1;
                continue;
            }
            if(dot(projectedNormal, referenceNormal) < normal_threshold){
                info[2] += 1;
                continue;
            }

            const float error = dot(referenceNormal, diff);
            ((float3 *)J)[0] = referenceNormal;
            ((float3 *)J)[1] = cross(projectedVertex, referenceNormal);

            // Error part
            sums[0] += error * error;

            // JTe part
            for(int i = 0; i < 6; ++i)
                sums[i+1] += error * J[i];

            // JTJ part
            jtj[0] += J[0] * J[0];
            jtj[1] += J[0] * J[1];
            jtj[2] += J[0] * J[2];
            jtj[3] += J[0] * J[3];
            jtj[4] += J[0] * J[4];
            jtj[5] += J[0] * J[5];

            jtj[6] += J[1] * J[1];
            jtj[7] += J[1] * J[2];
            jtj[8] += J[1] * J[3];
            jtj[9] += J[1] * J[4];
           jtj[10] += J[1] * J[5];

           jtj[11] += J[2] * J[2];
           jtj[12] += J[2] * J[3];
           jtj[13] += J[2] * J[4];
           jtj[14] += J[2] * J[5];

           jtj[15] += J[3] * J[3];
           jtj[16] += J[3] * J[4];
           jtj[17] += J[3] * J[5];

           jtj[18] += J[4] * J[4];
           jtj[19] += J[4] * J[5];

           jtj[20] += J[5] * J[5];

           // extra info here
           info[0] += 1;
        }
    }

    for(int i = 0; i < 32; ++i) // copy over to shared memory
        S[sline][i] = sums[i];

    __syncthreads();            // wait for everyone to finish

    if(sline < 32){ // sum up columns and copy to global memory in the final 32 threads
        for(unsigned i = 1; i < blockDim.x; ++i)
            S[0][sline] += S[i][sline];
        out[sline+blockIdx.x*32] = S[0][sline];
    }
}

void KFusion::Init( const KFusionConfig & config ) {
    configuration = config;

    cudaSetDeviceFlags(cudaDeviceMapHost);

    integration.init(config.volumeSize, config.volumeDimensions);

    reduction.alloc(config.inputSize);
    vertex.alloc(config.inputSize);
    normal.alloc(config.inputSize);
    rawDepth.alloc(config.inputSize);

    inputDepth.resize(config.iterations.size());
    inputVertex.resize(config.iterations.size());
    inputNormal.resize(config.iterations.size());

    for(int i = 0; i < config.iterations.size(); ++i){
        inputDepth[i].alloc(config.inputSize >> i);
        inputVertex[i].alloc(config.inputSize >> i);
        inputNormal[i].alloc(config.inputSize >> i);
    }

    gaussian.alloc(make_uint2(config.radius * 2 + 1, 1));
    output.alloc(make_uint2(32,8));

    //generate gaussian array
    generate_gaussian<<< 1, gaussian.size.x>>>(gaussian, config.delta, config.radius);

    Reset();
}

void KFusion::Reset(){
    dim3 block(32,16);
    dim3 grid = divup(dim3(integration.size.x, integration.size.y), block);
    initVolume<<<grid, block>>>(integration, make_float2(1.0f, 0.0f));
 }

void KFusion::Clear(){
    integration.release();
}

void KFusion::setPose( const Matrix4 & p ){
    pose = p;
}

void KFusion::setKinectDeviceDepth( const Image<uint16_t> & in){
    if(configuration.inputSize.x == in.size.x)
        mm2meters<0><<<divup(rawDepth.size, configuration.imageBlock), configuration.imageBlock>>>(rawDepth, in);
    else if(configuration.inputSize.x == in.size.x / 2 )
        mm2meters<1><<<divup(rawDepth.size, configuration.imageBlock), configuration.imageBlock>>>(rawDepth, in);
    else
        assert(false);
}

Matrix4 operator*( const Matrix4 & A, const Matrix4 & B){
    Matrix4 R;
    TooN::wrapMatrix<4,4>(&R.data[0].x) = TooN::wrapMatrix<4,4>(&A.data[0].x) * TooN::wrapMatrix<4,4>(&B.data[0].x);
    return R;
}

template<typename P>
inline Matrix4 toMatrix4( const TooN::SE3<P> & p){
    const TooN::Matrix<4, 4, float> I = TooN::Identity;
    Matrix4 R;
    TooN::wrapMatrix<4,4>(&R.data[0].x) = p * I;
    return R;
}

Matrix4 inverse( const Matrix4 & A ){
    static TooN::Matrix<4, 4, float> I = TooN::Identity;
    TooN::Matrix<4,4,float> temp =  TooN::wrapMatrix<4,4>(&A.data[0].x);
    Matrix4 R;
    TooN::wrapMatrix<4,4>(&R.data[0].x) = TooN::gaussian_elimination(temp , I );
    return R;
}

std::ostream & operator<<( std::ostream & out, const Matrix4 & m ){
    for(unsigned i = 0; i < 4; ++i)
        out << m.data[i].x << "  " << m.data[i].y << "  " << m.data[i].z << "  " << m.data[i].w << "\n";
    return out;
}

template <typename P, typename A>
TooN::Matrix<6> makeJTJ( const TooN::Vector<21, P, A> & v ){
    TooN::Matrix<6> C = TooN::Zeros;
    C[0]              = v.template slice<0,6>();
    C[1].template slice<1,5>() = v.template slice<6,5>();
    C[2].template slice<2,4>() = v.template slice<11,4>();
    C[3].template slice<3,3>() = v.template slice<15,3>();
    C[4].template slice<4,2>() = v.template slice<18,2>();
    C[5][5] = v[20];

    for(int r = 1; r < 6; ++r)
        for(int c = 0; c < r; ++c)
            C[r][c] = C[c][r];

    return C;
}

template <typename T, typename A>
TooN::Vector<6> solve( const TooN::Vector<27, T, A> & vals ){
    const TooN::Vector<6> b = vals.template slice<0,6>();
    const TooN::Matrix<6> C = makeJTJ(vals.template slice<6,21>());

    TooN::GR_SVD<6,6> svd(C);
    return svd.backsub(b, 1e6);
}

void KFusion::Raycast(){
    // raycast integration volume into the depth, vertex, normal buffers
    raycastPose = pose;
    raycast<<<divup(configuration.inputSize, configuration.raycastBlock), configuration.raycastBlock>>>(vertex, normal, integration, raycastPose * getInverseCameraMatrix(configuration.camera), configuration.nearPlane, configuration.farPlane, configuration.stepSize(), 0.75f * configuration.mu);
}

bool KFusion::Track() {
    const Matrix4 invK = getInverseCameraMatrix(configuration.camera);

    vector<dim3> grids;
    for(int i = 0; i < configuration.iterations.size(); ++i)
        grids.push_back(divup(configuration.inputSize >> i, configuration.imageBlock));

    // filter the input depth map
    bilateral_filter<<<grids[0], configuration.imageBlock>>>(inputDepth[0], rawDepth, gaussian, configuration.e_delta, configuration.radius);

    // half sample the input depth maps into the pyramid levels
    for(int i = 1; i < configuration.iterations.size(); ++i)
        halfSampleRobust<<<grids[i], configuration.imageBlock>>>(inputDepth[i], inputDepth[i-1], configuration.e_delta * 3, 1);

    // prepare the 3D information from the input depth maps
    for(int i = 0; i < configuration.iterations.size(); ++i){
        depth2vertex<<<grids[i], configuration.imageBlock>>>( inputVertex[i], inputDepth[i], getInverseCameraMatrix(configuration.camera / float(1 << i))); // inverse camera matrix depends on level
        vertex2normal<<<grids[i], configuration.imageBlock>>>( inputNormal[i], inputVertex[i] );
    }

    const Matrix4 oldPose = pose;
    const Matrix4 projectReference = getCameraMatrix(configuration.camera) * inverse(raycastPose);

    TooN::Matrix<8, 32, float, TooN::Reference::RowMajor> values(output.data());
    for(int level = configuration.iterations.size()-1; level >= 0; --level){
        for(int i = 0; i < configuration.iterations[level]; ++i){
            if(configuration.combinedTrackAndReduce){
                trackAndReduce<<<8, 112>>>( output.getDeviceImage().data(), inputVertex[level], inputNormal[level], vertex, normal, pose, projectReference, configuration.dist_threshold, configuration.normal_threshold );
            } else {
                track<<<grids[level], configuration.imageBlock>>>( reduction, inputVertex[level], inputNormal[level], vertex, normal, pose, projectReference, configuration.dist_threshold, configuration.normal_threshold);
                reduce<<<8, 112>>>( output.getDeviceImage().data(), reduction, inputVertex[level].size );             // compute the linear system to solve
            }
            cudaDeviceSynchronize(); // important due to async nature of kernel call
            for(int j = 1; j < 8; ++j)
                values[0] += values[j];
            TooN::Vector<6> x = solve(values[0].slice<1,27>());
            TooN::SE3<> delta(x);
            pose = toMatrix4( delta ) * pose;
            if(norm(x) < 1e-5)
                break;
        }
    }

    // test on both RSME per pixel and percent of pixels tracked
    if((sqrt(values(0,0) / values(0,28)) > 2e-2) || (values(0,28) / (rawDepth.size.x * rawDepth.size.y) < configuration.track_threshold) ){
        pose = oldPose;
        return false;
    }
    return true;
}

void KFusion::Integrate() {
    integrate<<<divup(dim3(integration.size.x, integration.size.y), configuration.imageBlock), configuration.imageBlock>>>( integration, rawDepth, inverse(pose), getCameraMatrix(configuration.camera), configuration.mu, configuration.maxweight );
}

int printCUDAError() {
    cudaError_t error = cudaGetLastError();
    if(error)
        std::cout << cudaGetErrorString(error) << std::endl;
    return error;
}
