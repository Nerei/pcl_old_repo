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
    class ProjectiveICP
    {
    public:
      enum { MAX_PYRAMID_LEVELS = 4 };

      typedef std::vector<Depth> DepthPyr;
      typedef std::vector<Points> PointsPyr;
      typedef std::vector<Normals> NormalsPyr;

      ProjectiveICP();
      virtual ~ProjectiveICP();

      float getDistThreshold() const;
      void setDistThreshold(float distance);

      float getAngleThreshold() const;
      void setAngleThreshold(float angle);

      void setIterationsNum(const std::vector<int>& iters);
      int getUsedLevelsNum() const;

      /** The function takes masked depth, i.e. it assumes for performance reasons that 
        * "if depth(y,x) is not zero, then normals(y,x) surely is not qnan" */
      virtual bool estimateTransform(Affine3f& affine, const Intr& intr, const DepthPyr& dcurr, const NormalsPyr ncurr, const DepthPyr dprev, const NormalsPyr nprev);

      virtual bool estimateTransform(Affine3f& affine, const Intr& intr, const PointsPyr& vcurr, const NormalsPyr ncurr, const PointsPyr vprev, const NormalsPyr nprev);

      static Vec3f rodrigues2(const Mat3f& matrix);
    private:
        std::vector<int> iters_;
        float angle_thres_;
        float dist_thres_;
        pcl::gpu::DeviceArray2D<float> buffer_;

        struct StreamHelper;
        boost::shared_ptr<StreamHelper> shelp_;
    };
  }
}