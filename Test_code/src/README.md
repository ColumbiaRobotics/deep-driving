# TORCS/ROS Bridge

**[Note that all rights for this repositoy belong to Florian Mirus. This repo was edited and updated with our scripts for purposes of the project. Following is some parts of the original README, and this repository can also be found on his [GitHub](https://github.com/fmirus/torcs_ros)]**

## Description of original TORCS/ROS Bridge packages

### torcs_img_publisher

This package publishes the current game image received via shared memory. For this package to work you need [opencv](http://opencv.org/)

### torcs_msgs

This package holds custom message files for torcs, namely ```TORCSCtrl``` and ```TORCSSensors```

### torcs_ros_bringup

This package holds config and launch files to start the whole ROS machinery

### torcs_ros_client

This is a ROS implementation of the original [SRC C++ client](https://sourceforge.net/projects/cig/files/SCR%20Championship/Client%20C%2B%2B/). However, this client only publishes data received from the game and subscribes to ctrl messages. It does not generate driving commands itself.

### torcs_ros_drive_ctrl

This is a separated implementation of the SimpleDriver contained in the original [SRC C++ client](https://sourceforge.net/projects/cig/files/SCR%20Championship/Client%20C%2B%2B/). It subscribes to the sensor messsage published by the ```torcs_ros_client```, generates simple drive commands and publishes them as ```TORCSCtrl``` messages.

## New package - tf_self_drv

This package interfaces with the TORCS/ROS bridge for testing and data collection. Refer the repository README file for more information on how to run the code.
