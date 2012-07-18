/*
 * Software License Agreement  (BSD License)
 *
 *  Point Cloud Library  (PCL) - www.pointclouds.org
 *  Copyright  (c) 2012, Jeremie Papon.
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
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES  (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef CLOUD_COMPOSER_ITEM_H_
#define CLOUD_COMPOSER_ITEM_H_


#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/apps/cloud_composer/qt.h>
#include <pcl/apps/cloud_composer/properties_model.h>

enum ITEM_ROLES { 
  PROPERTIES = Qt::UserRole,
  ITEM_ID
};

//This currently isn't used for anything, it will probably be removed
enum ITEM_TYPES { 
  CLOUD_COMPOSER_ITEM = QStandardItem::UserType,
  CLOUD_ITEM,
  NORMALS_ITEM,
  FPFH_ITEM
};

namespace pcl
{
  namespace cloud_composer
  {
    class PropertiesModel;
    
    class PCL_EXPORTS CloudComposerItem : public QStandardItem
    {
      public:
        typedef boost::shared_ptr<pcl::cloud_composer::CloudComposerItem> Ptr;
        typedef boost::shared_ptr<pcl::cloud_composer::CloudComposerItem> ConstPtr;
        
        CloudComposerItem (const QString name = "default item");
        CloudComposerItem (const CloudComposerItem& to_copy);
        virtual ~CloudComposerItem ();
        
        inline virtual int 
        type () const { return CLOUD_COMPOSER_ITEM; }

        /** \brief  Getter function for getting unique internal ID*/
        inline QString
        getID () const { return item_id_; }
        
        virtual CloudComposerItem*
        clone () const;
        
        inline void
        setProperties (PropertiesModel* new_props)
        {
          properties_ = new_props;
        }
        
        inline PropertiesModel* 
        getProperties () const 
        { 
          return properties_;
        }
        
        /** \brief Paint View function - reimpliment in item subclass if it can be displayed in PCLVisualizer*/
        virtual void
        paintView (boost::shared_ptr<pcl::visualization::PCLVisualizer> vis) const;
        
        /** \brief Remove from View function - reimpliment in item subclass if it can be displayed in PCLVisualizer*/
        virtual void
        removeFromView (boost::shared_ptr<pcl::visualization::PCLVisualizer> vis) const;
        
        /** \brief Inspector Display function - reimpliment in item subclass if it can be displayed in Inspector*/
        virtual QWidget*
        paintInspector ();
       
        
        /** \brief Helper function to pull out a cloud from Item */
        bool
        getCloudConstPtr (sensor_msgs::PointCloud2::ConstPtr& const_ptr) const;

        /** \brief The property model calls this when a property changes */
        inline void 
        propertyChanged ()
        {
          emitDataChanged ();
        }
      protected:

        /** \brief Model for storing the properties of the item   */
        PropertiesModel* properties_;
        
        /** \brief Internal ID used when referencing this object, mainly for PCLVisualizer */
        QString item_id_;
    };
    
    /** \brief Templated helper class for converting QVariant to/from pointer classes   */
    template <class T> class VPtr
    {
      public:
        static T* asPtr (QVariant v)
        {
          return (static_cast<T *> (v.value<void *> ()));
        }

        static QVariant asQVariant (T* ptr)
        {
          return (qVariantFromValue (static_cast<void*>(ptr)));
        }
    };
    
  }
}

typedef QList<const pcl::cloud_composer::CloudComposerItem*> ConstItemList;

Q_DECLARE_METATYPE (pcl::cloud_composer::CloudComposerItem);










#endif //CLOUD_COMPOSER_ITEM_H_
