#include <octomap_generator/octomap_generator_ros.h>
#include <pcl_ros/transforms.h>
#include <pcl_ros/impl/transforms.hpp>
#include <octomap_msgs/conversions.h>
#include <pcl/conversions.h>
#include <cmath>
#include <sstream>
#include <cstring> // For std::memcpy

OctomapGeneratorNode::OctomapGeneratorNode(ros::NodeHandle& nh): nh_(nh)
{
  nh_.getParam("/octomap/tree_type", tree_type_);
  // Initiate octree
  if(tree_type_ == SEMANTICS_OCTREE_BAYESIAN || tree_type_ == SEMANTICS_OCTREE_MAX)
  {
    if(tree_type_ == SEMANTICS_OCTREE_BAYESIAN)
    {
      ROS_INFO("Semantic octomap generator [bayesian fusion]");
      octomap_generator_ = new OctomapGenerator<PCLSemanticsBayesian, SemanticsOctreeBayesian>();
    }
    else
    {
      ROS_INFO("Semantic octomap generator [max fusion]");
      octomap_generator_ = new OctomapGenerator<PCLSemanticsMax, SemanticsOctreeMax>();
    }
    service_ = nh_.advertiseService("toggle_use_semantic_color", &OctomapGeneratorNode::toggleUseSemanticColor, this);
  }
  else
  {
    ROS_INFO("Color octomap generator");
    octomap_generator_ = new OctomapGenerator<PCLColor, ColorOcTree>();
  }
  reset();
  fullmap_pub_ = nh_.advertise<octomap_msgs::Octomap>("octomap_full", 1, true);
  pointcloud_sub_ = new message_filters::Subscriber<sensor_msgs::PointCloud2> (nh_, pointcloud_topic_, 5);
  tf_pointcloud_sub_ = new tf::MessageFilter<sensor_msgs::PointCloud2> (*pointcloud_sub_, tf_listener_, world_frame_id_, 5);
  tf_pointcloud_sub_->registerCallback(boost::bind(&OctomapGeneratorNode::insertCloudCallback, this, _1));
}

OctomapGeneratorNode::~OctomapGeneratorNode() {}
/// Clear octomap and reset values to paramters from parameter server
void OctomapGeneratorNode::reset()
{
  nh_.getParam("/octomap/pointcloud_topic", pointcloud_topic_);
  nh_.getParam("/octomap/world_frame_id", world_frame_id_);
  nh_.getParam("/octomap/resolution", resolution_);
  nh_.getParam("/octomap/max_range", max_range_);
  nh_.getParam("/octomap/raycast_range", raycast_range_);
  nh_.getParam("/octomap/clamping_thres_min", clamping_thres_min_);
  nh_.getParam("/octomap/clamping_thres_max", clamping_thres_max_);
  nh_.getParam("/octomap/occupancy_thres", occupancy_thres_);
  nh_.getParam("/octomap/prob_hit", prob_hit_);
  nh_.getParam("/octomap/prob_miss", prob_miss_);
  nh_.getParam("/tree_type", tree_type_);
  octomap_generator_->setClampingThresMin(clamping_thres_min_);
  octomap_generator_->setClampingThresMax(clamping_thres_max_);
  octomap_generator_->setResolution(resolution_);
  octomap_generator_->setOccupancyThres(occupancy_thres_);
  octomap_generator_->setProbHit(prob_hit_);
  octomap_generator_->setProbMiss(prob_miss_);
  octomap_generator_->setRayCastRange(raycast_range_);
  octomap_generator_->setMaxRange(max_range_);
}

bool OctomapGeneratorNode::toggleUseSemanticColor(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response)
{
  octomap_generator_->setUseSemanticColor(!octomap_generator_->isUseSemanticColor());
  if(octomap_generator_->isUseSemanticColor())
    ROS_INFO("Using semantic color");
  else
    ROS_INFO("Using rgb color");
  if (octomap_msgs::fullMapToMsg(*octomap_generator_->getOctree(), map_msg_))
     fullmap_pub_.publish(map_msg_);
  else
     ROS_ERROR("Error serializing OctoMap");
  return true;
}

void OctomapGeneratorNode::insertCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg)
{
  // Voxel filter to down sample the point cloud
  // Create the filtering object
  pcl::PCLPointCloud2::Ptr cloud (new pcl::PCLPointCloud2 ());
  pcl_conversions::toPCL(*cloud_msg, *cloud);
  // Get tf transform
  tf::StampedTransform sensorToWorldTf;
  try
  {
    tf_listener_.lookupTransform(world_frame_id_, cloud_msg->header.frame_id, cloud_msg->header.stamp, sensorToWorldTf);
  }
  catch(tf::TransformException& ex)
  {
    ROS_ERROR_STREAM( "Transform error of sensor data: " << ex.what() << ", quitting callback");
    return;
  }
  // Transform coordinate
  Eigen::Matrix4f sensorToWorld;
  pcl_ros::transformAsMatrix(sensorToWorldTf, sensorToWorld);
  octomap_generator_->insertPointCloud(cloud, sensorToWorld);
  // Publish octomap
  map_msg_.header.frame_id = world_frame_id_;
  map_msg_.header.stamp = cloud_msg->header.stamp;
  if (octomap_msgs::fullMapToMsg(*octomap_generator_->getOctree(), map_msg_))
     fullmap_pub_.publish(map_msg_);
  else
     ROS_ERROR("Error serializing OctoMap");
}

bool OctomapGeneratorNode::save(const char* filename) const
{
  octomap_generator_->save(filename);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "octomap_generator");
  ros::NodeHandle nh;
  OctomapGeneratorNode octomapGeneratorNode(nh);
  ros::spin();
  std::string save_path;
  nh.getParam("/octomap/save_path", save_path);

  octomapGeneratorNode.save(save_path.c_str());
  ROS_INFO("OctoMap saved.");
  return 0;
}
