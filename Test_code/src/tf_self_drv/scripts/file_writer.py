#!/usr/bin/env python

##### This script is used primarily for helping collect data for the
##### training procedure.
##### Using the TORCS/ROS bridge, this script subscribes and collects
##### both the image and corresponding driving commands, and writes
##### a pre-set number of image/command pairs to two .npy files.

import rospy
import numpy
import time
import os
import message_filters
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
from scipy import misc
from torcs_msgs.msg import TORCSCtrl
import matplotlib.pyplot as plt

# data parameters
BUFFER_SIZE = 1000
IMG_SIZE = [480,640]
LOWER_X_CROP = 200
UPPER_X_CROP = 380
LOWER_Y_CROP = 70
UPPER_Y_CROP = 570
CHANNELS = 3

count = 1

# variables not defined inside callback as it will get re-initialized every
# time a new image is received by the ROS node
images = numpy.zeros((BUFFER_SIZE,IMG_SIZE[0],IMG_SIZE[1],CHANNELS))
labels = [None] * BUFFER_SIZE

# track name used when saving file
track = 'E-Road(old)'

def callback(image, cmd):
	
	global count
	print("Number of images collected: {}".format(count))
	
	bridge = CvBridge()
	cv_image = bridge.imgmsg_to_cv2(image, "rgb8")
	cropped_img = cv_image[LOWER_X_CROP:UPPER_X_CROP+1, LOWER_Y_CROP:UPPER_Y_CROP+1]

	imagePub.publish(bridge.cv2_to_imgmsg(cropped_img))

	# appending image to buffer as a numpy matrix
	temp = numpy.asarray(cv_image)
	images[count,:,:,:] = temp

	# appending labels to label buffer
	labels[count] = [cmd.accel, cmd.brake, cmd.clutch, cmd.gear, cmd.steering, cmd.focus, cmd.meta]
	
	# counter variable to check how many images we have collected
	count+=1
	if(count > BUFFER_SIZE):
		# deactivate ROS subscribers
		imageSub.unregister()
		cmdSub.unregister()

		# call function to write data to .npy file
		writeToFile(images,labels)

def writeToFile(images, labels):

 	# written to make code robust to list of images as input as well
 	masterArray = numpy.asarray(images)

 	uniqueTimeStamp = str(time.time())
 	
 	# writing numpy variables as .npy files
 	root = os.path.abspath(os.sep) # to get the path to the home directory
 	numpy.save(os.path.join(root,'/media/brian/Data/NPY_Files/imageArray_'+track+'_'+uniqueTimeStamp),masterArray)

 	labelsArray = numpy.asarray(labels)
 	numpy.save(os.path.join(root,'/media/brian/Data/NPY_Files/labelsArray_'+track+'_'+uniqueTimeStamp),labelsArray)

 	# printed when files written
 	rospy.loginfo("Files written at time {}".format(uniqueTimeStamp))

if __name__ == '__main__':

	# initializing node
	rospy.init_node('FileWriter_Node', anonymous = False)
	
	# subscribing to the messages publised by the TORCS/ROS bridge
	imageSub = message_filters.Subscriber('torcs_ros/pov_image', Image)
	cmdSub = message_filters.Subscriber('torcs_ros/ctrl_cmd', TORCSCtrl)

	# creating a time-sync policy to sync image and car control commands
	sync = message_filters.ApproximateTimeSynchronizer([imageSub, cmdSub], 10, 0.2)
	imagePub = rospy.Publisher('/recorded_image',Image,queue_size=10)

	sync.registerCallback(callback)
	rospy.spin()
	