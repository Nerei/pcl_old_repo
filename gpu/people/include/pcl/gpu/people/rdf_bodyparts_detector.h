/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Willow Garage, Inc.
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
 * @author: Koen Buys, Anatoly Baksheev
 */


#ifndef PCL_GPU_PEOPLE_RDF_BODYPARTS_DETECTOR_H
#define PCL_GPU_PEOPLE_RDF_BODYPARTS_DETECTOR_H

#include <pcl/pcl_exports.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/gpu/people/colormap.h>
#include <pcl/gpu/people/label_blob2.h>
#include <boost/shared_ptr.hpp>
#include <string>
#include <vector>

namespace pcl
{
  namespace device
  {
    class MultiTreeLiveProc;
  }

  namespace gpu
  {
    namespace people
    {
      class PCL_EXPORTS RDFBodyPartsDetector
      {
      public:
        typedef boost::shared_ptr<RDFBodyPartsDetector> Ptr;

        typedef DeviceArray2D<unsigned short> Depth;
        typedef DeviceArray2D<unsigned char> Labels;
        typedef DeviceArray2D<pcl::RGB> Image;

        RDFBodyPartsDetector(const std::vector<std::string>& tree_files, 
            int default_buffer_rows = 480, int default_buffer_cols = 640);

        //RDF & smooth
        void computeLabels(const Depth& depth);

        ////////// in development (dirty ) //////////

        typedef std::vector<std::vector<label_skeleton::Blob2, Eigen::aligned_allocator<label_skeleton::Blob2> > > BlobMatrix;

        void step2_selectBetterName(const PointCloud<PointXYZ>& cloud, int cluster_area_threshold, BlobMatrix& blobs);
        ////////////////////////////////////////////

        //getters
        const Labels& getLabels() const;
        size_t treesNumber() const;

        //utility
        void colorizeLabels(const Labels& labels, Image& color_labels) const;
      private:
        boost::shared_ptr<device::MultiTreeLiveProc> impl_;

        Labels labels_;
        Labels labels_smoothed_;
        DeviceArray<pcl::RGB> color_map_;

        int max_cluster_size_;
        float cluster_tolerance_;

        void optimized_elec4(const PointCloud<pcl::PointXYZ>& cloud, const cv::Mat& src_labels, float tolerance,
                    std::vector<std::vector<PointIndices> > &labeled_clusters,
                    unsigned int min_pts_per_cluster, unsigned int max_pts_per_cluster, unsigned int num_parts,
                    bool brute_force_border, float radius_scale);

      };
    }
  }
}

#endif /* PCL_GPU_PEOPLE_RDF_BODYPARTS_DETECTOR_H */
