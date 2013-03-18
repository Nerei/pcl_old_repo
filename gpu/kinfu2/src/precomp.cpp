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

#include "precomp.hpp"
#include "internal.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Kinfu/types implementation

pcl::Intr::Intr () {};
pcl::Intr::Intr (float fx_, float fy_, float cx_, float cy_) : fx(fx_), fy(fy_), cx(cx_), cy(cy_) {}
      
pcl::Intr pcl::Intr::operator()(int level_index) const
{
  int div = 1 << level_index;
  return (Intr (fx / div, fy / div, cx / div, cy / div));
}

std::ostream& operator << (std::ostream& os, const pcl::Intr& intr)
{
  return os << "([f = " << intr.fx << ", " << intr.fy << "] [cp = " << intr.cx << ", " << intr.cy << "])";
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// TsdfVolume host implementation

pcl::device::TsdfVolume::TsdfVolume(elem_type* _data, int3 _dims, float3 _voxel_size, float _trunc_dist, int _max_weight) 
: data(_data), dims(_dims), voxel_size(_voxel_size), trunc_dist(_trunc_dist), max_weight(_max_weight) {}

//pcl::device::TsdfVolume::elem_type* pcl::device::TsdfVolume::operator()(int x, int y, int z)
//{ return data + x + y*dims.x + z*dims.y*dims.x; }
//
//const pcl::device::TsdfVolume::elem_type* pcl::device::TsdfVolume::operator() (int x, int y, int z) const
//{ return data + x + y*dims.x + z*dims.y*dims.x; }
//
//pcl::device::TsdfVolume::elem_type* pcl::device::TsdfVolume::beg(int x, int y) const
//{ return data + x + dims.x * y; }
//
//pcl::device::TsdfVolume::elem_type* pcl::device::TsdfVolume::zstep(elem_type *const ptr) const
//{ return data + dims.x * dims.y; }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// CyclicTsdfVolume host implementation

pcl::device::CyclicTsdfVolume::CyclicTsdfVolume(elem_type* data, int3 dims, float3 voxel_size, float trunc_dist, int max_weight, int3 _origin) 
  : TsdfVolume(data, dims, voxel_size, trunc_dist, max_weight), origin(_origin) {}

//pcl::device::TsdfVolume::elem_type* pcl::device::CyclicTsdfVolume::operator()(int x, int y, int z)
//{
//  x = (x + origin.x) % dims.x;
//  y = (y + origin.y) % dims.y;
//  z = (z + origin.z) % dims.z;
//  return TsdfVolume::operator()(x, y, z);
//}
//const pcl::device::TsdfVolume::elem_type* pcl::device::CyclicTsdfVolume::operator() (int x, int y, int z) const
//{
//  x = (x + origin.x) % dims.x;
//  y = (y + origin.y) % dims.y;
//  z = (z + origin.z) % dims.z;
//  return TsdfVolume::operator()(x, y, z);
//}
//
//pcl::device::TsdfVolume::elem_type* pcl::device::CyclicTsdfVolume::beg(int x, int y)
//{
//  x = (x + origin.x) % dims.x;
//  y = (y + origin.y) % dims.y;
//  return TsdfVolume::operator()(x, y, origin.z);
//}
//
//pcl::device::TsdfVolume::elem_type* pcl::device::CyclicTsdfVolume::zstep(elem_type *const ptr) const
//{
//  int index = (ptr - data) + dims.x * dims.y;
//  index = index % (dims.x*dims.y*dims.z);
//  return data + index;
//}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Projector host implementation

pcl::device::Projector::Projector(float fx, float fy, float cx, float cy) : f(make_float2(fx, fy)), c(make_float2(cx, cy)) {}

//float2 pcl::device::Projector::operator()(const float3& p) const
//{
//  float2 coo;
//  coo.x = p.x * f.x / p.z + c.x;
//  coo.y = p.y * f.y / p.z + c.y;
//  return coo;
//}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Reprojector host implementation

pcl::device::Reprojector::Reprojector(float fx, float fy, float cx, float cy) : finv(make_float2(1.f/fx, 1.f/fy)), c(make_float2(cx, cy)) {}

//float3 pcl::device::Reprojector::operator()(int u, int v, float z) const
//{
//  float x = z * (u - c.x) * finv.x;
//  float y = z * (v - c.y) * finv.y;
//  return make_float3(x, y, z);
//}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Host implementation of packing/unpacking tsdf volume element

//ushort2 pcl::device::pack_tsdf(float tsdf, int weight) { throw "Not implemented"; return ushort2(); }
//float pcl::device::unpack_tsdf(ushort2 value, int& weight) { throw "Not implemented"; return 0; }
//float pcl::device::unpack_tsdf(ushort2 value) { throw "Not implemented"; return 0; }

