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
#include <pcl2/kinfu/cuda/tsdf_volume.hpp>
#include <pcl2/kinfu/cuda/projective_icp.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>

namespace pcl
{
  struct PCL_EXPORTS KinFuParams
  {
    static KinFuParams default_params();

    int cols;  //pixels
    int rows;  //pixels
  
    Intr intr;  //Camera parameters

    Vec3i volume_dims; //number of voxels
    Vec3f volume_size; //meters
    Affine3f volume_pose; //meters, inital pose

    float bilateral_sigma_depth;   //meters
    float bilateral_sigma_spatial;   //pixels
    int   bilateral_kernel_size;   //pixels

    float icp_truncate_depth_dist; //meters
    float icp_dist_thres;          //meters
    float icp_angle_thres;         //radians
    std::vector<int> icp_iter_num; //iterations for level index 0,1,..,3
    
    float tsdf_min_camera_movement; //meters, integrate only if exceedes
    float tsdf_trunc_dist;             //meters;
    int tsdf_max_weight;               //frames

    float raycast_step_factor;   // in voxel sizes
    float gradient_delta_factor; // in voxel sizes

    Vec3f light_pose; //meters

  };

  class PCL_EXPORTS KinFu
  {
  public:
    typedef boost::shared_ptr<KinFu> Ptr;

    KinFu(const KinFuParams& params);

    const KinFuParams& params() const;
    KinFuParams& params();

    const cuda::TsdfVolume& tsdf() const;
    cuda::TsdfVolume& tsdf();

    const cuda::ProjectiveICP& icp() const;
    cuda::ProjectiveICP& icp();

    void reset();

    bool operator()(const cuda::Depth& dpeth, const cuda::Image& image);

    void renderImage(cuda::Image& image, int flags = 0);
    void renderImage(cuda::Image& image, const Affine3f& pose, int flags = 0);

    Affine3f getCameraPose (int time = -1) const;
  private:
    void allocate_buffers();

    int frame_counter_;
    KinFuParams params_;

    std::vector<Affine3f> poses_;

    cuda::Dists dists_;

    cuda::ProjectiveICP::DepthPyr depth_pyr_curr_;
    cuda::ProjectiveICP::PointsPyr points_pyr_curr_;
    cuda::ProjectiveICP::NormalsPyr normals_pyr_curr_;

    cuda::ProjectiveICP::DepthPyr depth_pyr_prev_;
    cuda::ProjectiveICP::PointsPyr points_pyr_prev_;
    cuda::ProjectiveICP::NormalsPyr normals_pyr_prev_;

    cuda::Points points_;
    cuda::Normals normals_;
    cuda::Depth depths_;

    boost::shared_ptr<cuda::TsdfVolume> volume_;
    boost::shared_ptr<cuda::ProjectiveICP> icp_;
  };
}
