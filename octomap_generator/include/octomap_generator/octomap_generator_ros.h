#ifndef OCTOMAP_GENERATOR_ROS_H
#define OCTOMAP_GENERATOR_ROS_H

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <semantics_octree/semantics_octree.h>
#include <std_srvs/Empty.h>
#include <octomap/ColorOcTree.h>
#include <octomap/Pointcloud.h>
#include <octomap/octomap.h>
#include <memory>
#include <boost/shared_ptr.hpp>
#include <tf/transform_listener.h>
#include <tf/message_filter.h>
#include <message_filters/subscriber.h>
#include <string>
#include <octomap_msgs/Octomap.h>
#include <octomap_generator/octomap_generator.h>

#define COLOR_OCTREE 0
#define SEMANTICS_OCTREE_MAX 1
#define SEMANTICS_OCTREE_BAYESIAN 2

/**
 * \brief ROS wrapper for octomap generator
 * \details Adapted from [RGBD slam](http://wiki.ros.org/rgbdslam) and [octomap server](http://wiki.ros.org/octomap_server)
 * \author Xuan Zhang
 * \data Mai-July 2018
 */
class OctomapGeneratorNode{
  public:
    /**
     * \brief Constructor
     * \param nh The ros node handler to be used in OctomapGenerator
     */
    OctomapGeneratorNode(ros::NodeHandle& nh);
    /// Desturctor
    virtual ~OctomapGeneratorNode();
    /// Reset values to paramters from parameter server
    void reset();
    /**
     * \brief Callback to point cloud topic. Update the octomap and publish it in ROS
     * \param cloud ROS Pointcloud2 message in arbitrary frame (specified in the clouds header)
     */
    void insertCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud);
    /**
     * \brief Save octomap to a file. NOTE: Not tested
     * \param filename The output filename
     */
    bool save(const char* filename) const;

  protected:
    OctomapGeneratorBase* octomap_generator_; ///<Octomap instance pointer
    ros::ServiceServer service_;  ///<ROS service to toggle semantic color display
    bool toggleUseSemanticColor(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response); ///<Function to toggle whether write semantic color or rgb color as when serializing octree
    ros::NodeHandle nh_; ///<ROS handler
    ros::Publisher fullmap_pub_; ///<ROS publisher for octomap message
    message_filters::Subscriber<sensor_msgs::PointCloud2>* pointcloud_sub_; ///<ROS subscriber for pointcloud message
    tf::MessageFilter<sensor_msgs::PointCloud2>* tf_pointcloud_sub_; ///<ROS tf message filter to sychronize the tf and pointcloud messages
    tf::TransformListener tf_listener_; ///<Listener for the transform between the camera and the world coordinates
    std::string world_frame_id_; ///<Id of the world frame
    std::string pointcloud_topic_; ///<Topic name for subscribed pointcloud message
    float max_range_; ///<Max range for points to be inserted into octomap
    float raycast_range_; ///<Max range for points to perform raycasting to free unoccupied space
    float clamping_thres_max_; ///<Upper bound of occupancy probability for a node
    float clamping_thres_min_; ///<Lower bound of occupancy probability for a node
    float resolution_; ///<Resolution of octomap
    float occupancy_thres_; ///<Minimum occupancy probability for a node to be considered as occupied
    float prob_hit_;  ///<Hit probability of sensor
    float prob_miss_; ///<Miss probability of sensor
    int tree_type_; ///<0: color octree, 1: semantic octree using bayesian fusion, 2: semantic octree using max fusion
    octomap_msgs::Octomap map_msg_; ///<ROS octomap message
  };

#endif//OCTOMAP_GENERATOR_ROS
