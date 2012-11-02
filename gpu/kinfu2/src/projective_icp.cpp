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

#include "pcl/common/angles.h"
#include <Eigen/SVD>
#include <Eigen/Cholesky>
#include <Eigen/LU>

using namespace pcl;
using namespace std;
using namespace pcl::cuda;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// ComputeIcpHelper

pcl::device::ComputeIcpHelper::ComputeIcpHelper(float dist_thres, float angle_thres)
{
  min_cosine = cos(angle_thres);
  dist2_thres = dist_thres * dist_thres;
}

void pcl::device::ComputeIcpHelper::setLevelIntr(int level_index, float fx, float fy, float cx, float cy)
{
  int div = 1 << level_index;
  f = make_float2(fx/div, fy/div);
  c = make_float2(cx/div, cy/div);
  finv = make_float2(1.f/f.x, 1.f/f.y);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// ProjectiveICP::StreamHelper

struct pcl::cuda::ProjectiveICP::StreamHelper
{
  typedef device::ComputeIcpHelper::PageLockHelper PageLockHelper;
  typedef Eigen::Matrix<float, 6, 6, Eigen::RowMajor> Mat6f;
  typedef Eigen::Matrix<float, 6, 1> Vec6f;
  
  cudaStream_t stream;
  PageLockHelper locked_buffer;
  
  StreamHelper() { cudaSafeCall( cudaStreamCreate(&stream) ); }
  ~StreamHelper() { cudaSafeCall( cudaStreamDestroy(stream) ); }

  operator float*() { return locked_buffer.data; }
  operator cudaStream_t() { return stream; }

  Mat6f get(Vec6f& b)
  {
    cudaSafeCall( cudaStreamSynchronize(stream) );

    Mat6f A;
    float *data_A = A.data();
    float *data_b = b.data();

    int shift = 0;
    for (int i = 0; i < 6; ++i)   //rows
      for (int j = i; j < 7; ++j) // cols + b
      {
        float value = locked_buffer.data[shift++];
        if (j == 6)               // vector b
          data_b[i] = value;
        else
          data_A[j * 6 + i] = data_A[i * 6 + j] = value;
      }
    return A;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// ProjectiveICP

pcl::cuda::ProjectiveICP::ProjectiveICP() : angle_thres_(pcl::deg2rad(20.f)), dist_thres_(0.1f)
{ 
  const int iters[] = {10, 5, 4, 0};
  std::vector<int> vector_iters(iters, iters + 4);
  setIterationsNum(vector_iters);
  device::ComputeIcpHelper::allocate_buffer(buffer_);

  shelp_.reset( new StreamHelper() );
}

pcl::cuda::ProjectiveICP::~ProjectiveICP() {}

float pcl::cuda::ProjectiveICP::getDistThreshold() const
{ return dist_thres_; }

void pcl::cuda::ProjectiveICP::setDistThreshold(float distance)
{ dist_thres_ = distance; }

float pcl::cuda::ProjectiveICP::getAngleThreshold() const
{ return angle_thres_; }

void pcl::cuda::ProjectiveICP::setAngleThreshold(float angle)
{ angle_thres_ = angle; }

void pcl::cuda::ProjectiveICP::setIterationsNum(const std::vector<int>& iters)
{
  if (iters.size() >= MAX_PYRAMID_LEVELS)
    iters_.assign(iters.begin(), iters.begin() + MAX_PYRAMID_LEVELS);
  else
  {
    iters_ = vector<int>(MAX_PYRAMID_LEVELS, 0);
    copy(iters.begin(), iters.end(),iters_.begin());
  }
}

int pcl::cuda::ProjectiveICP::getUsedLevelsNum() const
{
  int i = MAX_PYRAMID_LEVELS - 1;
  for(; i >= 0 && !iters_[i]; --i);  
  return i + 1;
}

bool pcl::cuda::ProjectiveICP::estimateTransform(Affine3f& affine, const Intr& intr, const DepthPyr& dcurr, const NormalsPyr ncurr, const DepthPyr dprev, const NormalsPyr nprev)
{
  const int LEVELS = getUsedLevelsNum();
  StreamHelper& sh = *shelp_;

  device::ComputeIcpHelper helper(dist_thres_, angle_thres_);
  affine = Affine3f::Identity();

  for(int level_index = LEVELS - 1; level_index >= 0; --level_index)
  {
    const device::Normals& n = (const device::Normals& )nprev[level_index];
    
    helper.rows = (float)n.rows();
    helper.cols = (float)n.cols();
    helper.setLevelIntr(level_index, intr.fx, intr.fy, intr.cx, intr.cy);
    helper.dcurr = dcurr[level_index];
    helper.ncurr = ncurr[level_index];

    for(int iter = 0; iter < iters_[level_index]; ++iter)
    {
      helper.R = eigen_cast<device::Mat3f, Mat3f>(affine.rotation());
      helper.t = eigen_cast<device::Vec3f, Vec3f>(affine.translation());
      
      helper(dprev[level_index], n, buffer_, sh, sh);

      StreamHelper::Vec6f b;
      StreamHelper::Mat6f A  = sh.get(b);

      //checking nullspace
      double det = A.determinant();
      
      if (fabs (det) < 1e-15 || pcl_isnan (det))
      {
        if (pcl_isnan (det)) cout << "qnan" << endl;
        return false;
      }

      Eigen::Matrix<float, 6, 1> result = A.llt ().solve (b);
      //Eigen::Matrix<float, 6, 1> result = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);

      Eigen::AngleAxisf Rx_alpha(result(0), Vec3f::UnitX ());
      Eigen::AngleAxisf Ry_beta (result(1), Vec3f::UnitY ());
      Eigen::AngleAxisf Rz_gamma(result(2), Vec3f::UnitZ ());
      Eigen::Translation3f translation(result.tail<3>());

      Affine3f Tinc = translation * Rz_gamma * Ry_beta * Rx_alpha;
      affine = Tinc * affine;
    }
  }
  return true;
}

bool pcl::cuda::ProjectiveICP::estimateTransform(Affine3f& affine, const Intr& intr, const PointsPyr& vcurr, const NormalsPyr ncurr, const PointsPyr vprev, const NormalsPyr nprev)
{
  const int LEVELS = getUsedLevelsNum();
  StreamHelper& sh = *shelp_;

  device::ComputeIcpHelper helper(dist_thres_, angle_thres_);
  affine = Affine3f::Identity();

  for(int level_index = LEVELS - 1; level_index >= 0; --level_index)
  {
    const device::Normals& n = (const device::Normals& )nprev[level_index];
    const device::Points& v = (const device::Points& )vprev[level_index];
    
    helper.rows = (float)n.rows();
    helper.cols = (float)n.cols();
    helper.setLevelIntr(level_index, intr.fx, intr.fy, intr.cx, intr.cy);
    helper.vcurr = vcurr[level_index];
    helper.ncurr = ncurr[level_index];

    for(int iter = 0; iter < iters_[level_index]; ++iter)
    {
      helper.R = eigen_cast<device::Mat3f, Mat3f>(affine.rotation());
      helper.t = eigen_cast<device::Vec3f, Vec3f>(affine.translation());

      helper(v, n, buffer_, sh, sh);

      StreamHelper::Vec6f b;
      StreamHelper::Mat6f A  = sh.get(b);

      //checking nullspace
      double det = A.determinant();
      
      if (fabs (det) < 1e-15 || pcl_isnan (det))
      {
        if (pcl_isnan (det)) cout << "qnan" << endl;
        return false;
      }

      Eigen::Matrix<float, 6, 1> result = A.llt ().solve (b);
      //Eigen::Matrix<float, 6, 1> result = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);

      Eigen::AngleAxisf Rx_alpha(result(0), Vec3f::UnitX ());
      Eigen::AngleAxisf Ry_beta (result(1), Vec3f::UnitY ());
      Eigen::AngleAxisf Rz_gamma(result(2), Vec3f::UnitZ ());
      Eigen::Translation3f translation(result.tail<3>());

      Affine3f Tinc = translation * Rz_gamma * Ry_beta * Rx_alpha;
      affine = Tinc * affine;
    }
  }
  return true;
}

pcl::Vec3f pcl::cuda::ProjectiveICP::rodrigues2(const Mat3f& matrix)
{
  Eigen::JacobiSVD<Mat3f> svd(matrix, Eigen::ComputeFullV | Eigen::ComputeFullU);
  Mat3f R = svd.matrixU() * svd.matrixV().transpose();

  double rx = R(2, 1) - R(1, 2);
  double ry = R(0, 2) - R(2, 0);
  double rz = R(1, 0) - R(0, 1);

  double s = sqrt((rx*rx + ry*ry + rz*rz)*0.25);
  double c = (R.trace() - 1) * 0.5;
  c = c > 1. ? 1. : c < -1. ? -1. : c;

  double theta = acos(c);

  if( s < 1e-5)
  {
    double t;

    if( c > 0 )
      rx = ry = rz = 0;
    else
    {
      t = (R(0, 0) + 1)*0.5;
      rx = sqrt( std::max(t, 0.0) );
      t = (R(1, 1) + 1)*0.5;
      ry = sqrt( std::max(t, 0.0) ) * (R(0, 1) < 0 ? -1.0 : 1.0);
      t = (R(2, 2) + 1)*0.5;
      rz = sqrt( std::max(t, 0.0) ) * (R(0, 2) < 0 ? -1.0 : 1.0);

      if( fabs(rx) < fabs(ry) && fabs(rx) < fabs(rz) && (R(1, 2) > 0) != (ry*rz > 0) )
        rz = -rz;
      theta /= sqrt(rx*rx + ry*ry + rz*rz);
      rx *= theta;
      ry *= theta;
      rz *= theta;
    }
  }
  else
  {
    double vth = 1/(2*s);
    vth *= theta;
    rx *= vth; ry *= vth; rz *= vth;
  }
  return Eigen::Vector3d(rx, ry, rz).cast<float>();
}
