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

#include <pcl2/kinfu/types.hpp>

namespace pcl
{
  namespace cuda
  {
    class PCL_EXPORTS TsdfVolume
    {
    public:
      TsdfVolume(const Vec3i& dims);
      virtual ~TsdfVolume();

      void create(const Vec3i& dims);
      
      Vec3i getDims() const;
      Vec3f getVoxelSize() const;

      const CudaData data() const;
      CudaData data();

      Vec3f getSize() const;
      void setSize(const Vec3f& size);

      float getTruncDist() const;
      void setTruncDist(float distance);

      int getMaxWeight() const;
      void setMaxWeight(int weight);

      Affine3f getPose() const;
      void setPose(const Affine3f& pose);

      float getRaycastStepFactor() const;
      void setRaycastStepFactor(float factor);

      float getGradientDeltaFactor() const;
      void setGradientDeltaFactor(float factor);

      Vec3i getGridOrigin() const;
      void setGridOrigin(const Vec3i& origin);

      virtual void clear();
      virtual void applyAffine(const Affine3f& affine);
      virtual void integrate(const Dists& dists, const Affine3f& camera_pose, const Intr& intr);
      virtual void raycast(const Affine3f& camera_pose, const Intr& intr, Depth& depth, Normals& normals);
      virtual void raycast(const Affine3f& camera_pose, const Intr& intr, Points& points, Normals& normals);

      void swap(CudaData& data);

      DeviceArray<Point> fetchCloud(DeviceArray<Point>& cloud_buffer) const;
      void fetchNormals(const DeviceArray<Point>& cloud, DeviceArray<pcl::Normal>& normals) const;

      struct Entry
      {
        typedef unsigned short half;

        half tsdf;
        unsigned short weight;

        static float half2float(half value);
        static half float2half(float value);
      };
    private:
      CudaData data_;

      float trunc_dist_;
      int max_weight_;
      Vec3i dims_;
      Vec3f size_;
      Affine3f pose_;

      Vec3i grid_origin_;

      float gradient_delta_factor_;
      float raycast_step_factor_;
    };
  }
}