/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2010, Willow Garage, Inc.
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
 * $Id$
 *
 */

#ifndef PCL_COMMON_H_
#define PCL_COMMON_H_

#include <pcl/pcl_base.h>
#include <cfloat>

/**
 * \file common.h
 * Define standard C methods and C++ classes that are common to all methods
 * \ingroup common
 */

/*@{*/
namespace pcl
{
  /** \brief Compute the smallest angle between two vectors in the [ 0, PI ) interval in 3D.
    * \param v1 the first 3D vector (represented as a \a Eigen::Vector4f)
    * \param v2 the second 3D vector (represented as a \a Eigen::Vector4f)
    * \return the angle between v1 and v2
    */
  inline double getAngle3D (const Eigen::Vector4f &v1, const Eigen::Vector4f &v2);

  /** \brief Compute both the mean and the standard deviation of an array of values
    * \param values the array of values
    * \param mean the resultant mean of the distribution
    * \param stddev the resultant standard deviation of the distribution
    */
  inline void getMeanStd (const std::vector<float> &values, double &mean, double &stddev);

  /** \brief Get a set of points residing in a box given its bounds
    * \param cloud the point cloud data message
    * \param min_pt the minimum bounds
    * \param max_pt the maximum bounds
    * \param indices the resultant set of point indices residing in the box
    */
  template <typename PointT> 
  inline void getPointsInBox (const pcl::PointCloud<PointT> &cloud, Eigen::Vector4f &min_pt,
                              Eigen::Vector4f &max_pt, std::vector<int> &indices);

  /** \brief Get the point at maximum distance from a given point and a given pointcloud
    * \param cloud the point cloud data message
    * \param pivot_pt the point from where to compute the distance
    * \param max_pt the point in cloud that is the farthest point away from pivot_pt
    */
  template<typename PointT>
  inline void
  getMaxDistance (const pcl::PointCloud<PointT> &cloud, const Eigen::Vector4f &pivot_pt, Eigen::Vector4f &max_pt);

  /** \brief Get the point at maximum distance from a given point and a given pointcloud
      * \param cloud the point cloud data message
      * \param pivot_pt the point from where to compute the distance
      * \param indices the vector of point indices to use from \a cloud
      * \param max_pt the point in cloud that is the farthest point away from pivot_pt
      */
  template<typename PointT>
  inline void
  getMaxDistance (const pcl::PointCloud<PointT> &cloud, const std::vector<int> &indices, const Eigen::Vector4f &pivot_pt,
                  Eigen::Vector4f &max_pt);

  /** \brief Get the minimum and maximum values on each of the 3 (x-y-z) dimensions in a given pointcloud
    * \param cloud the point cloud data message
    * \param min_pt the resultant minimum bounds
    * \param max_pt the resultant maximum bounds
    */
  template <typename PointT> 
  inline void getMinMax3D (const pcl::PointCloud<PointT> &cloud, PointT &min_pt, PointT &max_pt);
  
  /** \brief Get the minimum and maximum values on each of the 3 (x-y-z) dimensions in a given pointcloud
    * \param cloud the point cloud data message
    * \param min_pt the resultant minimum bounds
    * \param max_pt the resultant maximum bounds
    */
  template <typename PointT> 
  inline void getMinMax3D (const pcl::PointCloud<PointT> &cloud, Eigen::Vector4f &min_pt, Eigen::Vector4f &max_pt);

  /** \brief Get the minimum and maximum values on each of the 3 (x-y-z) dimensions in a given pointcloud
    * \param cloud the point cloud data message
    * \param indices the vector of point indices to use from \a cloud
    * \param min_pt the resultant minimum bounds
    * \param max_pt the resultant maximum bounds
    */
  template <typename PointT> 
  inline void getMinMax3D (const pcl::PointCloud<PointT> &cloud, const std::vector<int> &indices, 
                           Eigen::Vector4f &min_pt, Eigen::Vector4f &max_pt);

  /** \brief Get the minimum and maximum values on each of the 3 (x-y-z) dimensions in a given pointcloud
    * \param cloud the point cloud data message
    * \param indices the vector of point indices to use from \a cloud
    * \param min_pt the resultant minimum bounds
    * \param max_pt the resultant maximum bounds
    */
  template <typename PointT> 
  inline void getMinMax3D (const pcl::PointCloud<PointT> &cloud, const pcl::PointIndices &indices, 
                           Eigen::Vector4f &min_pt, Eigen::Vector4f &max_pt);

  /** \brief Compute the radius of a circumscribed circle for a triangle formed of three points pa, pb, and pc
    * \param pa the first point
    * \param pb the second point
    * \param pc the third point
    * \return the radius of the circumscribed circle
    */
  template <typename PointT> 
  inline double getCircumcircleRadius (const PointT &pa, const PointT &pb, const PointT &pc);

  /** \brief Get the minimum and maximum values on a point histogram
    * \param histogram the point representing a multi-dimensional histogram
    * \param len the length of the histogram
    * \param min_p the resultant minimum 
    * \param max_p the resultant maximum 
    */
  template <typename PointT> 
  inline void getMinMax (const PointT &histogram, int len, float &min_p, float &max_p);

  /** \brief Get the minimum and maximum values on a point histogram
    * \param cloud the cloud containing multi-dimensional histograms
    * \param idx the point index representing the histogram that we need to compute min/max for
    * \param field_name the field name containing the multi-dimensional histogram
    * \param min_p the resultant minimum 
    * \param max_p the resultant maximum 
    */
  inline void 
  getMinMax (const sensor_msgs::PointCloud2 &cloud, int idx, const std::string &field_name, 
             float &min_p, float &max_p)
  {
    min_p = FLT_MAX;
    max_p = -FLT_MAX;

    int field_idx = -1;
    for (size_t d = 0; d < cloud.fields.size (); ++d)
      if (cloud.fields[d].name == field_name)
        field_idx = d;

    if (field_idx == -1)
    {
      ROS_ERROR ("[getMinMax] Invalid field (%s) given!", field_name.c_str ());
      return;
    }

    for (unsigned int i = 0; i < cloud.fields[field_idx].count; ++i)
    {
      float data;
      // TODO: replace float with the real data type
      memcpy (&data, &cloud.data[cloud.fields[field_idx].offset + i * sizeof (float)], sizeof (float));
      min_p = (data > min_p) ? min_p : data; 
      max_p = (data < max_p) ? max_p : data; 
    }
  }

  /** \brief Compute both the mean and the standard deviation of an array of values
    * \param values the array of values
    * \param mean the resultant mean of the distribution
    * \param stddev the resultant standard deviation of the distribution
    */
  inline void
  getMeanStdDev (const std::vector<float> &values, double &mean, double &stddev)
  {
    double sum = 0, sq_sum = 0;

    for (size_t i = 0; i < values.size (); ++i)
    {
      sum += values[i];
      sq_sum += values[i] * values[i];
    }
    mean = sum / values.size ();
    double variance = (double)(sq_sum - sum * sum / values.size ()) / (values.size () - 1);
    stddev = sqrt (variance);
  }

}
/*@}*/
#include "pcl/common/common.hpp"

#endif  //#ifndef PCL_COMMON_H_
