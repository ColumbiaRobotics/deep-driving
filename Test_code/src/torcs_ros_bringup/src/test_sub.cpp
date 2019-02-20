// This ROS node helps interface the TORCS/ROS bridge with our
// TensorFlow back-end. This is important since we are essentially
// trying to interface Python and C++ components, and this node
// subscribes to a topic published by a Python Node, and then 
// publishes it to a C++ node.

#include<ros/ros.h>
#include<torcs_msgs/TORCSCtrl.h>
#include<geometry_msgs/Point.h>

ros::Publisher ctrl_pub_;

void cb(const geometry_msgs::Point::ConstPtr& msg)
{
	torcs_msgs::TORCSCtrl torcs_ctrl_out_;
	torcs_ctrl_out_.header.stamp = ros::Time::now();
    torcs_ctrl_out_.clutch = 1;

    // Added by Team Mech
    torcs_ctrl_out_.accel = msg->y;
    torcs_ctrl_out_.brake = msg->z;
    torcs_ctrl_out_.steering = msg->x;
    torcs_ctrl_out_.gear = 1;

    ctrl_pub_.publish(torcs_ctrl_out_);
}; 

int main(int argc, char **argv)
{
	ros::init(argc,argv,"test_node");
	ros::NodeHandle nh;
	

	ros::Subscriber sub = nh.subscribe("car_commands", 1000, cb);
	ctrl_pub_ = nh.advertise<torcs_msgs::TORCSCtrl>("torcs_ros/arb", 1000);
	ros::spin();

	return 0;
}

