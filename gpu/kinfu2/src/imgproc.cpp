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

void pcl::cuda::depthBilateralFilter(const Depth& in, Depth& out, int kernel_size, float sigma_spatial, float sigma_depth)
{ 
  out.create(in.rows(), in.cols());
  device::bilateralFilter(in, out, kernel_size, sigma_spatial, sigma_depth); 
}

void pcl::cuda::depthTruncation(Depth& depth, float threshold)
{ device::truncateDepth(depth, threshold); }

void pcl::cuda::depthBuildPyramid(const Depth& depth, Depth& pyramid, float sigma_depth)
{ 
  pyramid.create (depth.rows () / 2, depth.cols () / 2);
  device::depthPyr(depth, pyramid, sigma_depth); 
}

void pcl::cuda::waitAllDefaultStream()
{ cudaSafeCall(cudaDeviceSynchronize() ); }

void pcl::cuda::computeNormalsAndMaskDepth(const Intr& intr, Depth& depth, Normals& normals)
{
  normals.create(depth.rows(), depth.cols());

  device::Reprojector reproj(intr.fx, intr.fy, intr.cx, intr.cy);

  device::Normals& n = (device::Normals&)normals;
  device::computeNormalsAndMaskDepth(reproj, depth, n);
}

void pcl::cuda::computePointNormals(const Intr& intr, const Depth& depth, Points& points, Normals& normals)
{
  points.create(depth.rows(), depth.cols());
  normals.create(depth.rows(), depth.cols());

  device::Reprojector reproj(intr.fx, intr.fy, intr.cx, intr.cy);
  
  device::Points& p = (device::Points&)points;
  device::Normals& n = (device::Normals&)normals;
  device::computePointNormals(reproj, depth, p, n);
}


void pcl::cuda::computeDists(const Depth& depth, Dists& dists, const Intr& intr)
{
  dists.create(depth.rows(), depth.cols());
  device::compute_dists(depth, dists, make_float2(intr.fx, intr.fy), make_float2(intr.cx, intr.cy));
}

void pcl::cuda::resizeDepthNormals(const Depth& depth, const Normals& normals, Depth& depth_out, Normals& normals_out)
{
  depth_out.create (depth.rows()/2, depth.cols()/2);
  normals_out.create (normals.rows()/2, normals.cols()/2);

  device::Normals& nsrc = (device::Normals&)normals;
  device::Normals& ndst = (device::Normals&)normals_out;

  device::resizeDepthNormals(depth, nsrc, depth_out, ndst);
}

void pcl::cuda::resizePointsNormals(const Points& points, const Normals& normals, Points& points_out, Normals& normals_out)
{
  points_out.create (points.rows()/2, points.cols()/2);
  normals_out.create (normals.rows()/2, normals.cols()/2);

  device::Points& pi = (device::Points&)points;
  device::Normals& ni= (device::Normals&)normals;
  
  device::Points& po = (device::Points&)points_out;
  device::Normals& no = (device::Normals&)normals_out;

  device::resizePointsNormals(pi, ni, po, no);
}


void pcl::cuda::renderImage(const Depth& depth, const Normals& normals, const Intr& intr, const Vec3f& light_pose, Image& image)
{
  image.create(depth.rows(), depth.cols());

  const device::Depth& d = (const device::Depth&)depth;
  const device::Normals& n = (const device::Normals&)normals;
  device::Reprojector reproj(intr.fx, intr.fy, intr.cx, intr.fy);
  device::Vec3f light = eigen_cast<device::Vec3f>(light_pose);

  device::Image& i = (device::Image&)image;
  device::renderImage(d, n, reproj, light, i);
  waitAllDefaultStream();
}

void pcl::cuda::renderImage(const Points& points, const Normals& normals, const Intr& intr, const Vec3f& light_pose, Image& image)
{
  image.create(points.rows(), points.cols());

  const device::Points& p = (const device::Points&)points;
  const device::Normals& n = (const device::Normals&)normals;
  device::Reprojector reproj(intr.fx, intr.fy, intr.cx, intr.fy);
  device::Vec3f light = eigen_cast<device::Vec3f>(light_pose);

  device::Image& i = (device::Image&)image;
  device::renderImage(p, n, reproj, light, i);
  waitAllDefaultStream();
}

void pcl::cuda::renderTangentColors(const Normals& normals, Image& image)
{
  image.create(normals.rows(), normals.cols());
  const device::Normals& n = (const device::Normals&)normals;
  device::Image& i = (device::Image&)image;

  device::renderTangentColors(n, i);
  waitAllDefaultStream();
}

