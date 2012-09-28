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

#include "precomp.hpp"

using namespace pcl;
using namespace pcl::cuda;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// TsdfVolume::Entry

float pcl::cuda::TsdfVolume::Entry::half2float(half)
{ throw "Not implemented"; }

pcl::cuda::TsdfVolume::Entry::half pcl::cuda::TsdfVolume::Entry::float2half(float value)
{ throw "Not implemented"; }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// TsdfVolume

pcl::cuda::TsdfVolume::TsdfVolume(const Vec3i& dims) : data_(), trunc_dist_(0.03), max_weight_(128), dims_(dims), 
  size_(Vec3f::Constant(3.f)), pose_(Affine3f::Identity()), grid_origin_(Vec3i::Constant(0)), gradient_delta_factor_(0.75f), raycast_step_factor_(0.75f)
{ create(dims_); }

pcl::cuda::TsdfVolume::~TsdfVolume() {}

void pcl::cuda::TsdfVolume::create(const Vec3i& dims)
{
  dims_ = dims;
  int voxels_number = dims_.prod();
  data_.create(voxels_number * sizeof(int));
  setTruncDist(trunc_dist_);
  clear();
}

Vec3i pcl::cuda::TsdfVolume::getDims() const
{ return dims_; }

Vec3f pcl::cuda::TsdfVolume::getVoxelSize() const
{ return size_.array() / dims_.array().cast<float>(); }

const CudaData pcl::cuda::TsdfVolume::data() const 
{ return data_; }

CudaData pcl::cuda::TsdfVolume::data() 
{  return data_; }

Vec3f pcl::cuda::TsdfVolume::getSize() const
{ return size_; }

void pcl::cuda::TsdfVolume::setSize(const Vec3f& size)
{ size_ = size; setTruncDist(trunc_dist_); }

float pcl::cuda::TsdfVolume::getTruncDist() const
{ return trunc_dist_; }

void pcl::cuda::TsdfVolume::setTruncDist(float distance)
{ trunc_dist_ = std::max (distance, 2.1f * getVoxelSize().maxCoeff()); }

int pcl::cuda::TsdfVolume::getMaxWeight() const
{ return max_weight_; }

void pcl::cuda::TsdfVolume::setMaxWeight(int weight)
{ max_weight_ = weight; }

Affine3f pcl::cuda::TsdfVolume::getPose() const 
{ return pose_; }

void pcl::cuda::TsdfVolume::setPose(const Affine3f& pose) 
{ pose_ = pose; }

Vec3i pcl::cuda::TsdfVolume::getGridOrigin() const
{ return grid_origin_; }

void pcl::cuda::TsdfVolume::setGridOrigin(const Vec3i& origin)
{ grid_origin_ = origin; }

float pcl::cuda::TsdfVolume::getRaycastStepFactor() const
{ return raycast_step_factor_; }

void pcl::cuda::TsdfVolume::setRaycastStepFactor(float factor) 
{ raycast_step_factor_ = factor; }

float pcl::cuda::TsdfVolume::getGradientDeltaFactor() const
{ return gradient_delta_factor_; }

void pcl::cuda::TsdfVolume::setGradientDeltaFactor(float factor) 
{ gradient_delta_factor_ = factor; }

void pcl::cuda::TsdfVolume::swap(CudaData& data)
{ data_.swap(data); }

void pcl::cuda::TsdfVolume::applyAffine(const Affine3f& affine)
{ pose_ = affine * pose_; }

void pcl::cuda::TsdfVolume::clear()
{ 
  device::Vec3i dims = eigen_cast<device::Vec3i>(dims_);
  device::Vec3f vsz  = eigen_cast<device::Vec3f>(getVoxelSize());

  device::TsdfVolume volume(data_.ptr<ushort2>(), dims, vsz, trunc_dist_, max_weight_);
  device::clear_volume(volume);
}

void pcl::cuda::TsdfVolume::integrate(const Dists& dists, const Affine3f& camera_pose, const Intr& intr)
{
  Affine3f vol2cam = camera_pose.inverse() * pose_;

  device::Mat3f R = eigen_cast<device::Mat3f, Mat3f>(vol2cam.rotation());
  device::Vec3f t = eigen_cast<device::Vec3f, Vec3f>(vol2cam.translation());

  device::Projector proj(intr.fx, intr.fy, intr.cx, intr.cy);

  device::Vec3i dims = eigen_cast<device::Vec3i>(dims_);
  device::Vec3f vsz  = eigen_cast<device::Vec3f>(getVoxelSize());
  device::Vec3i orig = eigen_cast<device::Vec3i>(grid_origin_);

  bool default_origin = (orig.x == 0 && orig.y == 0 && orig.z == 0);

  if (default_origin)
  {
    device::TsdfVolume volume(data_.ptr<ushort2>(), dims, vsz, trunc_dist_, max_weight_);
    device::integrate(dists, volume, R, t, proj);
  }
  else
  {
    device::CyclicTsdfVolume volume(data_.ptr<ushort2>(), dims, vsz, trunc_dist_, max_weight_, orig);
    device::integrate(dists, volume, R, t, proj);
  }
}

void pcl::cuda::TsdfVolume::raycast(const Affine3f& camera_pose, const Intr& intr, Depth& depth, Normals& normals)
{
  DeviceArray2D<device::Normal>& n = (DeviceArray2D<device::Normal>&)normals;

  Affine3f cam2vol = pose_.inverse() * camera_pose;

  device::Mat3f R = eigen_cast<device::Mat3f, Mat3f>(cam2vol.rotation());
  device::Vec3f t = eigen_cast<device::Vec3f, Vec3f>(cam2vol.translation());
  device::Mat3f Rinv = eigen_cast<device::Mat3f, Mat3f>(cam2vol.rotation().inverse());

  device::Reprojector reproj(intr.fx, intr.fy, intr.cx, intr.cy);
  
  device::Vec3i dims = eigen_cast<device::Vec3i>(dims_);
  device::Vec3f vsz  = eigen_cast<device::Vec3f>(getVoxelSize());
  device::Vec3i orig = eigen_cast<device::Vec3i>(grid_origin_);

  bool default_origin = (orig.x == 0 && orig.y == 0 && orig.z == 0);

  if (default_origin)
  {
    device::TsdfVolume volume(data_.ptr<ushort2>(), dims, vsz, trunc_dist_, max_weight_);
    device::raycast(volume, R, t, Rinv, reproj, depth, n, raycast_step_factor_, gradient_delta_factor_);
  }
  else
  {
    device::CyclicTsdfVolume volume(data_.ptr<ushort2>(), dims, vsz, trunc_dist_, max_weight_, orig);
    device::raycast(volume, R, t, Rinv, reproj, depth, n, raycast_step_factor_, gradient_delta_factor_);
  }
}

void pcl::cuda::TsdfVolume::raycast(const Affine3f& camera_pose, const Intr& intr, Points& points, Normals& normals)
{
  device::Normals& n = (device::Normals&)normals;
  device::Points& p = (device::Points&)points;

  Affine3f cam2vol = pose_.inverse() * camera_pose;

  device::Mat3f R = eigen_cast<device::Mat3f, Mat3f>(cam2vol.rotation());
  device::Vec3f t = eigen_cast<device::Vec3f, Vec3f>(cam2vol.translation());
  device::Mat3f Rinv = eigen_cast<device::Mat3f, Mat3f>(cam2vol.rotation().inverse());

  device::Reprojector reproj(intr.fx, intr.fy, intr.cx, intr.cy);

  device::Vec3i dims = eigen_cast<device::Vec3i>(dims_);
  device::Vec3f vsz  = eigen_cast<device::Vec3f>(getVoxelSize());
  device::Vec3i orig = eigen_cast<device::Vec3i>(grid_origin_);

  bool default_origin = (orig.x == 0 && orig.y == 0 && orig.z == 0);

  if (default_origin)
  {
    device::TsdfVolume volume(data_.ptr<ushort2>(), dims, vsz, trunc_dist_, max_weight_);
    device::raycast(volume, R, t, Rinv, reproj, p, n, raycast_step_factor_, gradient_delta_factor_);
  }
  else
  {
    device::CyclicTsdfVolume volume(data_.ptr<ushort2>(), dims, vsz, trunc_dist_, max_weight_, orig);
    device::raycast(volume, R, t, Rinv, reproj, p, n, raycast_step_factor_, gradient_delta_factor_);
  }
}

DeviceArray<Point>pcl::cuda::TsdfVolume::fetchCloud(DeviceArray<Point>& cloud_buffer) const
{
  enum { DEFAULT_CLOUD_BUFFER_SIZE = 10 * 1000 * 1000 };

  if (cloud_buffer.empty ())
    cloud_buffer.create (DEFAULT_CLOUD_BUFFER_SIZE);

  DeviceArray<device::Point>& b = (DeviceArray<device::Point>&)cloud_buffer;

  device::Vec3i dims = eigen_cast<device::Vec3i>(dims_);
  device::Vec3f vsz  = eigen_cast<device::Vec3f>(getVoxelSize());
  device::Vec3i orig = eigen_cast<device::Vec3i>(grid_origin_);

  device::Mat3f R = eigen_cast<device::Mat3f, Mat3f>(pose_.rotation());
  device::Vec3f t = eigen_cast<device::Vec3f, Vec3f>(pose_.translation());

  bool default_origin = (orig.x == 0 && orig.y == 0 && orig.z == 0);

  size_t size;
  if (default_origin)
  {
    device::TsdfVolume volume((ushort2*)data_.ptr<ushort2>(), dims, vsz, trunc_dist_, max_weight_);
    size = extractCloud(volume, R, t, b);
  }
  else
  {
    device::CyclicTsdfVolume volume((ushort2*)data_.ptr<ushort2>(), dims, vsz, trunc_dist_, max_weight_, orig);
    size = extractCloud(volume, R, t, b);
  }
  return DeviceArray<Point>((Point*)cloud_buffer.ptr(), size);
}

void pcl::cuda::TsdfVolume::fetchNormals(const DeviceArray<Point>& cloud, DeviceArray<pcl::Normal>& normals) const
{
  normals.create(cloud.size());
  DeviceArray<device::Point>& c = (DeviceArray<device::Point>&)cloud;

  device::Vec3i dims = eigen_cast<device::Vec3i>(dims_);
  device::Vec3f vsz  = eigen_cast<device::Vec3f>(getVoxelSize());
  device::Vec3i orig = eigen_cast<device::Vec3i>(grid_origin_);

  device::Mat3f R = eigen_cast<device::Mat3f, Mat3f>(pose_.rotation());
  device::Vec3f t = eigen_cast<device::Vec3f, Vec3f>(pose_.translation());
  device::Mat3f Rinv = eigen_cast<device::Mat3f, Mat3f>(pose_.rotation().inverse());

  bool default_origin = (orig.x == 0 && orig.y == 0 && orig.z == 0);

  if (default_origin)
  {
    device::TsdfVolume volume((ushort2*)data_.ptr<ushort2>(), dims, vsz, trunc_dist_, max_weight_);
    device::extractNormals(volume, c, R, t, Rinv, gradient_delta_factor_, (device::float8*)normals.ptr());
  }
  else
  {
    device::CyclicTsdfVolume volume((ushort2*)data_.ptr<ushort2>(), dims, vsz, trunc_dist_, max_weight_, orig);
    device::extractNormals(volume, c, R, t, Rinv, gradient_delta_factor_, (device::float8*)normals.ptr());
  }
}