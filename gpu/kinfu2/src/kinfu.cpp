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
#include "internal.hpp"
#include "pcl/common/angles.h"
#include "pcl/common/time.h"

using namespace std;
using namespace pcl;
using namespace pcl::cuda;

pcl::KinFuParams pcl::KinFuParams::default()
{
  const int iters[] = {10, 9, 0, 0};
  const int levels = sizeof(iters)/sizeof(iters[0]);
  
  KinFuParams p;

  p.cols = 640;  //pixels
  p.rows = 480;  //pixels
  p.intr = Intr(525.f, 525.f, p.cols/2 - 0.5f, p.rows/2 - 0.5f);

  p.volume_dims = Vec3i::Constant(512);  //number of voxels
  p.volume_size = Vec3f::Constant(3.f);  //meters
  p.volume_pose = Eigen::Translation3f(-p.volume_size(0)/2, -p.volume_size(1)/2, std::max(0.f, 1.f - p.volume_size(2)/2));

  p.bilateral_sigma_depth = 0.04;  //meter
  p.bilateral_sigma_spatial = 4.5; //pixels
  p.bilateral_kernel_size = 7;     //pixels

  p.icp_truncate_depth_dist = 0.f;        //meters, disabled
  p.icp_dist_thres = 0.1f;                //meters
  p.icp_angle_thres = pcl::deg2rad(30.f); //radians
  p.icp_iter_num.assign(iters, iters + levels);
  
  p.tsdf_min_camera_movement = 0.f; //meters, disabled
  p.tsdf_trunc_dist = 0.04f; //meters;
  p.tsdf_max_weight = 64;   //frames

  p.raycast_step_factor = 0.75f;  //in voxel sizes
  p.gradient_delta_factor = 0.5f; //in voxel sizes
  
  //p.light_pose = p.volume_pose.translation()/4; //meters
  p.light_pose = Vec3f::Constant(0.f); //meters

  return p;
}

pcl::KinFu::KinFu(const KinFuParams& params) : frame_counter_(0), params_(params)
{
  volume_.reset(new cuda::TsdfVolume(params_.volume_dims));

  volume_->setTruncDist(params_.tsdf_trunc_dist);
  volume_->setMaxWeight(params_.tsdf_max_weight);
  volume_->setSize(params_.volume_size);
  volume_->setPose(params_.volume_pose);
  volume_->setRaycastStepFactor(params_.raycast_step_factor);
  volume_->setGradientDeltaFactor(params_.gradient_delta_factor);

  icp_.reset(new cuda::ProjectiveICP());
  icp_->setDistThreshold(params_.icp_dist_thres);
  icp_->setAngleThreshold(params_.icp_angle_thres);
  icp_->setIterationsNum(params_.icp_iter_num);

  allocate_buffers();
  reset();
}

const pcl::KinFuParams& pcl::KinFu::params() const
{ return params_; }

pcl::KinFuParams& pcl::KinFu::params()
{ return params_; }

const pcl::cuda::TsdfVolume& pcl::KinFu::tsdf() const
{ return *volume_; }

pcl::cuda::TsdfVolume& pcl::KinFu::tsdf()
{ return *volume_; }

const pcl::cuda::ProjectiveICP& pcl::KinFu::icp() const
{ return *icp_; }

pcl::cuda::ProjectiveICP& pcl::KinFu::icp()
{ return *icp_; }

void pcl::KinFu::allocate_buffers()
{
  const int LEVELS = cuda::ProjectiveICP::MAX_PYRAMID_LEVELS;

  int cols = params_.cols;
  int rows = params_.rows;

  dists_.create(rows, cols);

  depth_pyr_curr_.resize(LEVELS);
  normals_pyr_curr_.resize(LEVELS);
  depth_pyr_prev_.resize(LEVELS);
  normals_pyr_prev_.resize(LEVELS);

  points_pyr_curr_.resize(LEVELS);
  points_pyr_prev_.resize(LEVELS);

  for(int i = 0; i < LEVELS; ++i)
  {    
    depth_pyr_curr_[i].create(rows, cols);
    normals_pyr_curr_[i].create(rows, cols);

    depth_pyr_prev_[i].create(rows, cols);
    normals_pyr_prev_[i].create(rows, cols);
    
    points_pyr_curr_[i].create(rows, cols);
    points_pyr_prev_[i].create(rows, cols);

    cols /= 2;
    rows /= 2;
  }

  depths_.create(params_.rows, params_.cols);
  normals_.create(params_.rows, params_.cols);
  points_.create(params_.rows, params_.cols);
}

void pcl::KinFu::reset()
{
  if (frame_counter_) 
    cout << "Reset" << endl;

  frame_counter_ = 0;
  poses_.clear();
  poses_.reserve(30000);
  poses_.push_back(Affine3f::Identity());
  volume_->clear();
}

pcl::Affine3f pcl::KinFu::getCameraPose (int time) const
{
  if (time > (int)poses_.size () || time < 0)
    time = poses_.size () - 1;
  return poses_[time];
}

bool pcl::KinFu::operator()(const pcl::cuda::Depth& depth, const pcl::cuda::Image& image)
{
  const KinFuParams& p = params_;
  const int LEVELS = icp_->getUsedLevelsNum();

  cuda::computeDists(depth, dists_, p.intr);
  cuda::depthBilateralFilter(depth, depth_pyr_curr_[0], p.bilateral_kernel_size, p.bilateral_sigma_spatial, p.bilateral_sigma_depth);

  if (p.icp_truncate_depth_dist > 0)
    pcl::cuda::depthTruncation(depth_pyr_curr_[0], p.icp_truncate_depth_dist);

  for (int i = 1; i < LEVELS; ++i)
    cuda::depthBuildPyramid(depth_pyr_curr_[i-1], depth_pyr_curr_[i], p.bilateral_sigma_depth);

  for (int i = 0; i < LEVELS; ++i)
#if defined USE_DEPTH
    cuda::computeNormalsAndMaskDepth(p.intr, depth_pyr_curr_[i], normals_pyr_curr_[i]);
#else
    cuda::computePointNormals(p.intr(i), depth_pyr_curr_[i], points_pyr_curr_[i], normals_pyr_curr_[i]);
#endif

  cuda::waitAllDefaultStream();

  //can't perform more on first frame
  if (frame_counter_ == 0)
  {
    volume_->integrate(dists_, poses_.back(), p.intr);
#if defined USE_DEPTH
    depth_pyr_curr_.swap(depth_pyr_prev_);
#else
    points_pyr_curr_.swap(points_pyr_prev_);
#endif
    normals_pyr_curr_.swap(normals_pyr_prev_);
    return ++frame_counter_, false;
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////////
  // ICP
  Affine3f affine; // cuur -> prev
  {
    //ScopeTime time("icp");
#if defined USE_DEPTH
    bool ok = icp_->estimateTransform(affine, p.intr, depth_pyr_curr_, normals_pyr_curr_, depth_pyr_prev_, normals_pyr_prev_);
#else
    bool ok = icp_->estimateTransform(affine, p.intr, points_pyr_curr_, normals_pyr_curr_, points_pyr_prev_, normals_pyr_prev_);
#endif
    if (!ok)
      return reset(), false;
  }

  poses_.push_back(poses_.back() * affine); // curr -> global

  ///////////////////////////////////////////////////////////////////////////////////////////
  // Volume integration
  
  // We do not integrate volume if camera does not move.
  float rnorm = cuda::ProjectiveICP::rodrigues2(affine.rotation()).norm();
  float tnorm = affine.translation().norm();
  bool integrate = (rnorm + tnorm)/2 >= p.tsdf_min_camera_movement;
  if (integrate)
  {
   // ScopeTime time("tsdf");
    volume_->integrate(dists_, poses_.back(), p.intr);
  }

  ///////////////////////////////////////////////////////////////////////////////////////////
  // Ray casting
  {
    //ScopeTime time("ray-cast-all");
#if defined USE_DEPTH
    volume_->raycast(poses_.back(), p.intr, depth_pyr_prev_[0], normals_pyr_prev_[0]);
    for (int i = 1; i < LEVELS; ++i)
      resizeDepthNormals(depth_pyr_prev_[i-1], normals_pyr_prev_[i-1], depth_pyr_prev_[i], normals_pyr_prev_[i]);
#else
    volume_->raycast(poses_.back(), p.intr, points_pyr_prev_[0], normals_pyr_prev_[0]);
    for (int i = 1; i < LEVELS; ++i)
      resizePointsNormals(points_pyr_prev_[i-1], normals_pyr_prev_[i-1], points_pyr_prev_[i], normals_pyr_prev_[i]);
#endif
    cuda::waitAllDefaultStream();
  }

  return ++frame_counter_, true;
}

void pcl::KinFu::renderImage(cuda::Image& image, int flag)
{
  const KinFuParams& p = params_;
  image.create(p.rows, flag != 3 ? p.cols : p.cols * 2);
  
#if defined USE_DEPTH
  #define PASS1 depth_pyr_prev_
#else
  #define PASS1 points_pyr_prev_
#endif

  if (flag < 1 || flag > 3)
    cuda::renderImage(PASS1[0], normals_pyr_prev_[0], params_.intr, params_.light_pose, image);
  else if (flag == 2)
    cuda::renderTangentColors(normals_pyr_prev_[0], image);
  else /* if (flag == 3) */
  {
    DeviceArray2D<RGB> i1(p.rows, p.cols, image.ptr(), image.step());
    DeviceArray2D<RGB> i2(p.rows, p.cols, image.ptr() + p.cols, image.step());

    cuda::renderImage(PASS1[0], normals_pyr_prev_[0], params_.intr, params_.light_pose, i1);
    cuda::renderTangentColors(normals_pyr_prev_[0], i2);
  }
#undef PASS1
}


void pcl::KinFu::renderImage(cuda::Image& image, const Affine3f& pose, int flag)
{
  const KinFuParams& p = params_;
  image.create(p.rows, flag != 3 ? p.cols : p.cols * 2);
  depths_.create(p.rows, p.cols);
  normals_.create(p.rows, p.cols);
  points_.create(p.rows, p.cols);

#if defined USE_DEPTH
  #define PASS1 depth_
#else
  #define PASS1 points_
#endif

  volume_->raycast(pose, p.intr, PASS1, normals_);

  if (flag < 1 || flag > 3)
    cuda::renderImage(PASS1, normals_, params_.intr, params_.light_pose, image);
  else if (flag == 2)
    cuda::renderTangentColors(normals_, image);
  else /* if (flag == 3) */
  {
    DeviceArray2D<RGB> i1(p.rows, p.cols, image.ptr(), image.step());
    DeviceArray2D<RGB> i2(p.rows, p.cols, image.ptr() + p.cols, image.step());

    cuda::renderImage(PASS1, normals_, params_.intr, params_.light_pose, i1);
    cuda::renderTangentColors(normals_, i2);
  }
#undef PASS1
}