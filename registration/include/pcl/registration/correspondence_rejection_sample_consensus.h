/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2011, Willow Garage, Inc.
 *  Copyright (c) 2012-, Open Perception, Inc.
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
 *   * Neither the name of the copyright holder(s) nor the names of its
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
#ifndef PCL_REGISTRATION_CORRESPONDENCE_REJECTION_SAMPLE_CONSENSUS_H_
#define PCL_REGISTRATION_CORRESPONDENCE_REJECTION_SAMPLE_CONSENSUS_H_

#include <pcl/registration/correspondence_rejection.h>

#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_registration.h>
#include <pcl/common/transforms.h>

namespace pcl
{
  namespace registration
  {
    /** \brief CorrespondenceRejectorSampleConsensus implements a correspondence rejection
      * using Random Sample Consensus to identify inliers (and reject outliers)
      * \author Dirk Holz
      * \ingroup registration
      */
    template <typename PointT>
    class CorrespondenceRejectorSampleConsensus: public CorrespondenceRejector
    {
      typedef pcl::PointCloud<PointT> PointCloud;
      typedef typename PointCloud::Ptr PointCloudPtr;
      typedef typename PointCloud::ConstPtr PointCloudConstPtr;

      public:
        using CorrespondenceRejector::input_correspondences_;
        using CorrespondenceRejector::rejection_name_;
        using CorrespondenceRejector::getClassName;

        typedef boost::shared_ptr<CorrespondenceRejectorSampleConsensus> Ptr;
        typedef boost::shared_ptr<const CorrespondenceRejectorSampleConsensus> ConstPtr;

        /** \brief Empty constructor. Sets the inlier threshold to 5cm (0.05m), 
          * and the maximum number of iterations to 1000. 
          */
        CorrespondenceRejectorSampleConsensus () 
          : inlier_threshold_ (0.05)
          , max_iterations_ (1000) // std::numeric_limits<int>::max ()
          , input_ ()
          , input_transformed_ ()
          , target_ ()
          , best_transformation_ ()
          , refine_ (false)
        {
          rejection_name_ = "CorrespondenceRejectorSampleConsensus";
        }

        /** \brief Empty destructor. */
        virtual ~CorrespondenceRejectorSampleConsensus () {}

        /** \brief Get a list of valid correspondences after rejection from the original set of correspondences.
          * \param[in] original_correspondences the set of initial correspondences given
          * \param[out] remaining_correspondences the resultant filtered set of remaining correspondences
          */
        inline void 
        getRemainingCorrespondences (const pcl::Correspondences& original_correspondences, 
                                     pcl::Correspondences& remaining_correspondences);

        /** \brief Provide a source point cloud dataset (must contain XYZ data!)
          * \param[in] cloud a cloud containing XYZ data
          */
        virtual inline void 
        setInputCloud (const PointCloudConstPtr &cloud) 
        { 
          PCL_WARN ("[pcl::registration::%s::setInputCloud] setInputCloud is deprecated. Please use setInputSource instead.\n", getClassName ().c_str ());
          input_ = cloud; 
        }

        /** \brief Get a pointer to the input point cloud dataset target. */
        inline PointCloudConstPtr const 
        getInputCloud () 
        { 
          PCL_WARN ("[pcl::registration::%s::getInputCloud] getInputCloud is deprecated. Please use getInputSource instead.\n", getClassName ().c_str ());
          return (input_); 
        }

        /** \brief Provide a source point cloud dataset (must contain XYZ data!)
          * \param[in] cloud a cloud containing XYZ data
          */
        virtual inline void 
        setInputSource (const PointCloudConstPtr &cloud) 
        { 
          input_ = cloud; 
        }

        /** \brief Get a pointer to the input point cloud dataset target. */
        inline PointCloudConstPtr const 
        getInputSource () { return (input_); }

        /** \brief Provide a target point cloud dataset (must contain XYZ data!)
          * \param[in] cloud a cloud containing XYZ data
          */
        virtual inline void 
        setTargetCloud (const PointCloudConstPtr &cloud) 
        { 
          PCL_WARN ("[pcl::registration::%s::setTargetCloud] setTargetCloud is deprecated. Please use setInputTarget instead.\n", getClassName ().c_str ());
          target_ = cloud; 
        }

        /** \brief Provide a target point cloud dataset (must contain XYZ data!)
          * \param[in] cloud a cloud containing XYZ data
          */
        virtual inline void 
        setInputTarget (const PointCloudConstPtr &cloud) { target_ = cloud; }

        /** \brief Get a pointer to the input point cloud dataset target. */
        inline PointCloudConstPtr const 
        getInputTarget () { return (target_ ); }

        /** \brief Set the maximum distance between corresponding points.
          * Correspondences with distances below the threshold are considered as inliers.
          * \param[in] threshold Distance threshold in the same dimension as source and target data sets.
          */
        inline void 
        setInlierThreshold (double threshold) { inlier_threshold_ = threshold; };

        /** \brief Get the maximum distance between corresponding points.
          * \return Distance threshold in the same dimension as source and target data sets.
          */
        inline double 
        getInlierThreshold() { return inlier_threshold_; };

        /** \brief Set the maximum number of iterations.
          * \param[in] max_iterations Maximum number if iterations to run
          */
        inline void 
        setMaxIterations (int max_iterations) 
        { 
          PCL_WARN ("[pcl::registration::%s::setMaxIterations] setMaxIterations is deprecated. Please use setMaximumIterations instead.\n", getClassName ().c_str ());
          max_iterations_ = std::max (max_iterations, 0); 
        }

        /** \brief Set the maximum number of iterations.
          * \param[in] max_iterations Maximum number if iterations to run
          */
        inline void 
        setMaximumIterations (int max_iterations) { max_iterations_ = std::max (max_iterations, 0); }

        /** \brief Get the maximum number of iterations.
          * \return max_iterations Maximum number if iterations to run
          */
        inline int 
        getMaxIterations () 
        {
          PCL_WARN ("[pcl::registration::%s::getMaxIterations] getMaxIterations is deprecated. Please use getMaximumIterations instead.\n", getClassName ().c_str ());
          return (max_iterations_); 
        }

        /** \brief Get the maximum number of iterations.
          * \return max_iterations Maximum number if iterations to run
          */
        inline int 
        getMaximumIterations () { return (max_iterations_); }

        /** \brief Get the best transformation after RANSAC rejection.
          * \return The homogeneous 4x4 transformation yielding the largest number of inliers.
          */
        inline Eigen::Matrix4f 
        getBestTransformation () { return best_transformation_; };

        /** \brief Provide a simple mechanism to update the internal source cloud
          * using a given transformation. Used in registration loops.
          * \param[in] transform the transform to apply over the source cloud
          */
        virtual bool
        updateSource (const Eigen::Matrix4d &transform)
        {
          if (!input_)
          {
            PCL_ERROR ("[pcl::registration::%s::updateSource] No input XYZ dataset given. Please specify the input source cloud using setInputSource.\n", getClassName ().c_str ());
            return (false);
          }
          input_transformed_.reset (new PointCloud);
          pcl::transformPointCloud<PointT, double> (*input_, *input_transformed_, transform);
          input_ = input_transformed_;
          return (true);
        }

        /** \brief Specify whether the model should be refined internally using the variance of the inliers
          * \param[in] refine true if the model should be refined, false otherwise
          */
        inline void
        setRefineModel (const bool refine)
        {
          refine_ = refine;
        }

        /** \brief Get the internal refine parameter value as set by the user using setRefineModel */
        inline bool
        getRefineModel () const
        {
          return (refine_);
        }
      protected:

        /** \brief Apply the rejection algorithm.
          * \param[out] correspondences the set of resultant correspondences.
          */
        inline void 
        applyRejection (pcl::Correspondences &correspondences)
        {
          getRemainingCorrespondences (*input_correspondences_, correspondences);
        }

        double inlier_threshold_;

        int max_iterations_;

        PointCloudConstPtr input_;
        PointCloudPtr input_transformed_;
        PointCloudConstPtr target_;

        Eigen::Matrix4f best_transformation_;

        bool refine_;
      public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
  }
}

#include <pcl/registration/impl/correspondence_rejection_sample_consensus.hpp>

#endif    // PCL_REGISTRATION_CORRESPONDENCE_REJECTION_SAMPLE_CONSENSUS_H_
