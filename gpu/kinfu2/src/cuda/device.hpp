/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Willow Garage, Inc.
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

#include "internal.hpp"
#include "temp_utils.hpp"



////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// TsdfVolume

//__device__ __forceinline__
//pcl::device::TsdfVolume::TsdfVolume(elem_type* _data, int3 _dims, float3 _voxel_size, float _trunc_dist, int _max_weight) 
//  : data(_data), dims(_dims), voxel_size(_voxel_size), trunc_dist(_trunc_dist), max_weight(_max_weight) {}

//__device__ __forceinline__
//pcl::device::TsdfVolume::TsdfVolume(const TsdfVolume& other)
//  : data(other.data), dims(other.dims), voxel_size(other.voxel_size), trunc_dist(other.trunc_dist), max_weight(other.max_weight) {}

__device__ __forceinline__
inline pcl::device::TsdfVolume::elem_type* pcl::device::TsdfVolume::operator()(int x, int y, int z)
{ return data + x + y*dims.x + z*dims.y*dims.x; }

__device__ __forceinline__
const pcl::device::TsdfVolume::elem_type* pcl::device::TsdfVolume::operator() (int x, int y, int z) const
{ return data + x + y*dims.x + z*dims.y*dims.x; }

__device__ __forceinline__
inline pcl::device::TsdfVolume::elem_type* pcl::device::TsdfVolume::beg(int x, int y) const
{ return data + x + dims.x * y; }

__device__ __forceinline__
inline pcl::device::TsdfVolume::elem_type* pcl::device::TsdfVolume::zstep(elem_type *const ptr) const
{ return ptr + dims.x * dims.y; }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// CyclicTsdfVolume

//__device__ __forceinline__
//pcl::device::CyclicTsdfVolume::CyclicTsdfVolume(elem_type* data, int3 dims, float3 voxel_size, float trunc_dist, int max_weight, int3 _origin) 
//   : TsdfVolume(data, dims, voxel_size, trunc_dist, max_weight), origin(_origin) {}

//__device__ __forceinline__
//pcl::device::CyclicTsdfVolume::CyclicTsdfVolume(const CyclicTsdfVolume& other) : TsdfVolume(other), origin(other.origin) {}

__device__ __forceinline__
inline pcl::device::TsdfVolume::elem_type* pcl::device::CyclicTsdfVolume::operator()(int x, int y, int z)
{
  x = (x + origin.x) % dims.x;
  y = (y + origin.y) % dims.y;
  z = (z + origin.z) % dims.z;
  return TsdfVolume::operator()(x, y, z);
}

__device__ __forceinline__
const pcl::device::TsdfVolume::elem_type* pcl::device::CyclicTsdfVolume::operator() (int x, int y, int z) const
{
  x = (x + origin.x) % dims.x;
  y = (y + origin.y) % dims.y;
  z = (z + origin.z) % dims.z;
  return TsdfVolume::operator()(x, y, z);
}

__device__ __forceinline__
inline pcl::device::TsdfVolume::elem_type* pcl::device::CyclicTsdfVolume::beg(int x, int y)
{
  x = (x + origin.x) % dims.x;
  y = (y + origin.y) % dims.y;
  return TsdfVolume::operator()(x, y, origin.z);
}

__device__ __forceinline__
inline pcl::device::TsdfVolume::elem_type* pcl::device::CyclicTsdfVolume::zstep(elem_type *const ptr) const
{
  int index = (ptr - data) + dims.x * dims.y;
  index = index % (dims.x*dims.y*dims.z);
  return data + index;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Projector

__device__ __forceinline__
inline float2 pcl::device::Projector::operator()(const float3& p) const
{
  float2 coo;
  coo.x = __fmaf_rn(f.x, __fdividef(p.x, p.z), c.x);
  coo.y = __fmaf_rn(f.y, __fdividef(p.y, p.z), c.y);
  return coo;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Reprojector

__device__ __forceinline__
inline float3 pcl::device::Reprojector::operator()(int u, int v, float z) const
{
  float x = z * (u - c.x) * finv.x;
  float y = z * (v - c.y) * finv.y;
  return make_float3(x, y, z);
}

 ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// packing/unpacking tsdf volume element

__device__ __forceinline__
inline ushort2 pcl::device::pack_tsdf (float tsdf, int weight)
{ return make_ushort2 (__float2half_rn (tsdf), weight); }

__device__ __forceinline__ 
inline float pcl::device::unpack_tsdf(ushort2 value, int& weight)
{
  weight = value.y;
  return __half2float (value.x);
}
__device__ __forceinline__ 
inline float pcl::device::unpack_tsdf (ushort2 value)
{ return __half2float (value.x); }


////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Utility

namespace pcl
{
  namespace device
  {
    __device__ __forceinline__
    inline Vec3f operator* (const Mat3f& m, const Vec3f& v)
    { return make_float3 (dot (m.data[0], v), dot (m.data[1], v), dot (m.data[2], v)); }

    __device__ __forceinline__ 
    inline Vec3f tr(const float4& v) 
    { return make_float3(v.x, v.y, v.z); }

    struct plus
    {
      __forceinline__ __device__ 
      float operator () (const float &l, const volatile float& r) const 
      { return l + r; }
    };

    struct gmem
    {
        template<typename T>
      __device__ __forceinline__ static T LdCs(T *ptr);

      template<typename T>
      __device__ __forceinline__ static void StCs(const T& val, T *ptr);
    };
  }
}


#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 200

  #if defined(_WIN64) || defined(__LP64__)
    #define _ASM_PTR_ "l"
  #else
    #define _ASM_PTR_ "r"
  #endif
  
  template<> ushort2 pcl::device::gmem::LdCs(ushort2* __restrict__ ptr)
  {
    ushort2 val;
    asm("ld.global.cs.v2.u16 {%0, %1}, [%2];" : "=h"(reinterpret_cast<ushort&>(val.x)), "=h"(reinterpret_cast<ushort&>(val.y)) : _ASM_PTR_(ptr));
    return val;
  }

  template<> void pcl::device::gmem::StCs(const ushort2& val, ushort2* __restrict__ ptr)
  {
    short cx = val.x, cy = val.y;
    asm("st.global.cs.v2.u16 [%0], {%1, %2};" : : _ASM_PTR_(ptr), "h"(reinterpret_cast<ushort&>(cx)), "h"(reinterpret_cast<ushort&>(cy)));
  }
  #undef _ASM_PTR_
#else
  template<> ushort2 pcl::device::gmem::LdCs(ushort2* __restrict__ ptr) { return *ptr; }
  template<> void pcl::device::gmem::StCs(const ushort2& val, ushort2* __restrict__ ptr) { *ptr = val; }
#endif






