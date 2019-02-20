cmake_minimum_required(VERSION 2.8.3)
project(torcs_ros_bringup)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
roscpp
geometry_msgs
torcs_msgs
)

#catkin_package(CATKIN_DEPENDS)
catkin_package(
#  LIBRARIES torcs_ros_drive_ctrl
  CATKIN_DEPENDS 
    geometry_msgs 
    roscpp 
    sensor_msgs 
    std_msgs 
    torcs_msgs
#  DEPENDS system_lib
)

## Add support for C++11, supported in ROS Kinetic and newer
# add_definitions(-std=c++11)

include_directories(${catkin_INCLUDE_DIRS})


add_executable(test_sub src/test_sub.cpp)
target_link_libraries(test_sub ${catkin_LIBRARIES})

## add_dependencies(test_sub roscpp geometry_msgs)


#############
## Install ##
#############

