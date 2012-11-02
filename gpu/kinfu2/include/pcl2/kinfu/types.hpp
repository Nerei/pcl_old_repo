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

#include <pcl/point_types.h>
#include <pcl2/kinfu/cuda/device_array.hpp>
#include <Eigen/Geometry>
#include <iosfwd>

namespace pcl
{
  typedef Eigen::Matrix<float, 3, 3, (Eigen::DontAlign|Eigen::RowMajor)>  Mat3f;

  typedef Eigen::Matrix<float, 3, 1, (Eigen::DontAlign)>  Vec3f;
  typedef Eigen::Matrix<int,   3, 1, (Eigen::DontAlign)>  Vec3i;
  typedef Eigen::Matrix<int,   2, 1, (Eigen::DontAlign)>  Vec2i;
  
  typedef Eigen::Transform<float,  3, Eigen::Affine, (Eigen::DontAlign|Eigen::RowMajor)> Affine3f;

  struct PCL_EXPORTS Intr
  {
    float fx, fy, cx, cy;
  
    Intr ();
    Intr (float fx, float fy, float cx, float cy);
    Intr operator()(int level_index) const;
  };

  std::ostream& operator << (std::ostream& os, const Intr& intr);

  namespace cuda
  {
    typedef pcl::PointXYZ Point;
    typedef pcl::PointXYZ Normal;

    struct PixelRGB
    {
      unsigned char r,g,b;
    };

    typedef DeviceMemory CudaData;
    typedef DeviceArray2D<unsigned short> Depth;
    typedef DeviceArray2D<unsigned short> Dists;
    typedef DeviceArray2D<RGB> Image;
    typedef DeviceArray2D<Normal> Normals;
    typedef DeviceArray2D<Point> Points;
  }
}