/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2011, Willow Garage, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#pragma once

#include <pcl2/kinfu/cuda/device_array.hpp>
#include "safe_call.hpp"

//#define USE_DEPTH

#if defined(__CUDACC__) 
    #define ___DEVICE__ __device__ __forceinline__ 
#else
    #define ___DEVICE__
#endif  

namespace pcl
{
  namespace device
  {
    typedef float4 Normal;
    typedef float4 Point;

    typedef unsigned short ushort;
    typedef unsigned char uchar;

    typedef PtrStepSz<ushort> Dists;
    typedef DeviceArray2D<ushort> Depth;
    typedef DeviceArray2D<Normal> Normals;
    typedef DeviceArray2D<Point> Points;
    typedef DeviceArray2D<uchar4> Image;

    struct Mat3f
    {
      float3 data[3];
    };
    typedef float3 Vec3f;
    typedef int3   Vec3i;
    

    struct TsdfVolume
    {
    public:
      typedef ushort2 elem_type;

      elem_type *const __restrict__ data;
      const int3 dims;
      const float3 voxel_size;
      const float trunc_dist;
      const int max_weight;

      TsdfVolume(elem_type* data, int3 dims, float3 voxel_size, float trunc_dist, int max_weight);
      //TsdfVolume(const TsdfVolume&);

      ___DEVICE__ elem_type* operator()(int x, int y, int z);
      ___DEVICE__ const elem_type* operator() (int x, int y, int z) const ;
      ___DEVICE__ elem_type* beg(int x, int y) const;
      ___DEVICE__ elem_type* zstep(elem_type *const ptr) const;
    private:
      TsdfVolume& operator=(const TsdfVolume&);
    };

    class CyclicTsdfVolume : public TsdfVolume
    {
    public:
      const int3 origin;

      CyclicTsdfVolume(elem_type* data, int3 dims, float3 voxel_size, float trunc_dist, int max_weight, int3 origin);
      //CyclicTsdfVolume(const CyclicTsdfVolume&);

      ___DEVICE__ elem_type* operator()(int x, int y, int z);
      ___DEVICE__ const elem_type* operator()(int x, int y, int z) const;

      ___DEVICE__ elem_type* beg(int x, int y);
      ___DEVICE__ elem_type* zstep(elem_type *const ptr) const;

    private:
      CyclicTsdfVolume& operator=(const CyclicTsdfVolume&);
    };

    struct Projector
    {
      float2 f, c;
      Projector(){}
      Projector(float fx, float fy, float cx, float cy);
      ___DEVICE__ float2 operator()(const float3& p) const;
    };

    struct Reprojector
    {
      Reprojector() {}
      Reprojector(float fx, float fy, float cx, float cy);
      float2 finv, c;
      ___DEVICE__ float3 operator()(int x, int y, float z) const;
    };

    struct ComputeIcpHelper
    {
      struct Policy;
      struct PageLockHelper
      {
        float* data;
        PageLockHelper();
        ~PageLockHelper();
      };

      float min_cosine;
      float dist2_thres;

      Mat3f R;
      Vec3f t;

      float rows, cols;
      float2 f, c, finv;

      PtrStep<ushort> dcurr;
      PtrStep<Normal> ncurr;
      PtrStep<Point> vcurr;

      ComputeIcpHelper(float dist_thres, float angle_thres);
      void setLevelIntr(int level_index, float fx, float fy, float cx, float cy);

      void operator()(const Depth& dprev, const Normals& nprev, DeviceArray2D<float>& buffer, float* data, cudaStream_t stream);
      void operator()(const Points& vprev, const Normals& nprev, DeviceArray2D<float>& buffer, float* data, cudaStream_t stream);

      static void allocate_buffer(DeviceArray2D<float>& buffer, int partials_count = -1);

    //private:
      ___DEVICE__ int find_coresp(int x, int y, float3& n, float3& d, float3& s) const;
      ___DEVICE__ void partial_reduce(const float row[7], PtrStep<float>& partial_buffer) const;
      ___DEVICE__ float2 proj(const float3& p) const;
      ___DEVICE__ float3 reproj(float x, float y, float z)  const;
    };

    //tsdf volume functions
    void clear_volume(TsdfVolume volume);
    
    template<typename Volume>
    void integrate(const Dists& depth, Volume volume, const Mat3f& R, const Vec3f& t, const Projector& proj);

    template<typename Volume>
    void raycast(const Volume& volume, const Mat3f& R, const Vec3f& t, const Mat3f& Rinv, 
        const Reprojector& reproj, Depth& depth, Normals& normals, float step_factor, float delta_factor);
    
    template<typename Volume>
    void raycast(const Volume& volume, const Mat3f& R, const Vec3f& t, const Mat3f& Rinv, 
        const Reprojector& reproj, Points& points, Normals& normals, float step_factor, float delta_factor);
    
    ___DEVICE__ ushort2 pack_tsdf(float tsdf, int weight);
    ___DEVICE__ float unpack_tsdf(ushort2 value, int& weight);
    ___DEVICE__ float unpack_tsdf(ushort2 value);


    //image proc functions
    void compute_dists(const Depth& depth, Dists dists, float2 f, float2 c);

    void truncateDepth(Depth& depth, float max_dist /*meters*/);
    void bilateralFilter(const Depth& src, Depth& dst, int kernel_size, float sigma_spatial, float sigma_depth);
    void depthPyr(const Depth& source, Depth& pyramid, float sigma_depth);

    void resizeDepthNormals(const Depth& depth, const Normals& normals, Depth& depth_out, Normals& normals_out);
    void resizePointsNormals(const Points& points, const Normals& normals, Points& points_out, Normals& normals_out);

    void computeNormalsAndMaskDepth(const Reprojector& reproj, Depth& depth, Normals& normals);
    void computePointNormals(const Reprojector& reproj, const Depth& depth, Points& points, Normals& normals);

    void renderImage(const Depth& depth, const Normals& normals, const Reprojector& reproj, const Vec3f& light_pose, Image& image);
    void renderImage(const Points& points, const Normals& normals, const Reprojector& reproj, const Vec3f& light_pose, Image& image);
    void renderTangentColors(const Normals& normals, Image& image);


    //exctraction functionality
    struct float8  { float x, y, z, w, c1, c2, c3, c4; };
    struct float12 { float x, y, z, w, normal_x, normal_y, normal_z, n4, c1, c2, c3, c4; };
    template<typename Vol> size_t extractCloud(const Vol& volume, const Mat3f& R, const Vec3f& t, PtrSz<Point> output);
    template<typename Vol>  void extractNormals(const Vol& volume, const PtrSz<Point>& points, const Mat3f& R, const Vec3f& t, const Mat3f& Rinv, 
        float gradient_delta_factor, float8* output);

    void mergePointNormal(const DeviceArray<Point>& cloud, const DeviceArray<float8>& normals, const DeviceArray<float12>& output);
  }
}
