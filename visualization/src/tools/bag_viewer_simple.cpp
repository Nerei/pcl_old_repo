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
 * $Id: pcd_viewer.cpp 34914 2010-12-20 22:50:09Z rusu $
 *
 */
// PCL

#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/ros/conversions.h>
#include <pcl/io/io.h>
#include <iostream>
#include <boost/foreach.hpp>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_visualization/cloud_viewer.h>

void
printHelp (int argc, char **argv)
{
  std::cout << "Syntax is:" << argv[0] << " <file_name>.pcd [cloud topic name = points2]\n";
}

/* ---[ */
int
main (int argc, char** argv)
{
  if (argc < 2)
  {
    printHelp (argc, argv);
    return (-1);
  }

  std::string cloud_topic = "points2";
  if (argc > 2)
  {
    cloud_topic = argv[2];
  }

  rosbag::Bag bag;
  bag.open (argv[1], rosbag::bagmode::Read);

  std::vector<std::string> topics;
  topics.push_back (cloud_topic);

  pcl_visualization::CloudViewer viewer ("Simple Cloud Viewer");

  while (!viewer.wasStopped ()) //loop over bag
  {
    rosbag::View view (bag, rosbag::TopicQuery (topics));

    BOOST_FOREACH(rosbag::MessageInstance const m, view)
    {
      if (viewer.wasStopped ())
      {
        std::cout << "quiting ..." << std::endl;
        return 0;
      }
      {
        pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud = m.instantiate<pcl::PointCloud<pcl::PointXYZRGB> > ();
        if (cloud)
          viewer.showCloud (*cloud);
      }
      {
        pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud = m.instantiate<pcl::PointCloud<pcl::PointXYZ> > ();
        if (cloud)
          viewer.showCloud (*cloud);
      }

    }
  }
  return 0;
}
/* ]--- */
