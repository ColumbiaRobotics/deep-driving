#!/usr/bin/env python2

##### Code that reads the images sent from TORCS Simulator to
##### then send it to the TensorFlow back-end.
##### The model sends back a prediction of car commands using
##### a custom message defintion of TORCSROSCtrl.
##### This message is then routed back to the TORCS Simulator.

import numpy
import rospy
import tensorflow as tf
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
import os
from scipy import misc
from torcs_msgs.msg import TORCSCtrl
import model

# To get the path of the saved model
root = os.getcwd()
tf.reset_default_graph()
sess = tf.Session()

# Restoring model - change to desired .meta file if new model is trained
saver = tf.train.import_meta_graph(root + '/src/tf_self_drv/model/best_model.meta')
saver.restore(sess, tf.train.latest_checkpoint(root + '/src/tf_self_drv/model/'))
graph = tf.get_default_graph()

Reading model description from tensorflow graph
idx = 0
tf_input = graph.get_operations()[idx].name+':0'
x = graph.get_tensor_by_name(tf_input)
tf_input = graph.get_operations()[2].name+':0'  
kp = graph.get_tensor_by_name(tf_input)
y = graph.get_tensor_by_name("fc_layer_3/Add:0")

# Cropping the raw image to focus only on the regions of itnerest
LOWER_X_CROP = 200
UPPER_X_CROP = 380
LOWER_Y_CROP = 70
UPPER_Y_CROP = 570

def callback(data):
	
	bridge = CvBridge()
	cv_image = bridge.imgmsg_to_cv2(data, "rgb8")
	cropped_img = cv_image[LOWER_X_CROP:UPPER_X_CROP+1, LOWER_Y_CROP:UPPER_Y_CROP+1]
	image = numpy.asarray(cropped_img)

	steerCmd = testModel(image)
	
	# printing the predicted steering value
	print(steerCmd)
	publishCommands(steerCmd)

def testModel(image):
	   
	# Make prediciton
	
	image2 = misc.imresize(image,(66,200,3))

	# converting 3-D image to 4-D for dimensionality purposes
	null_mat = numpy.ones((1,66,200,3))
	null_mat[0,:,:,:] = image2.astype(numpy.float32)
	y_out = sess.run(y, feed_dict={x: null_mat, kp: 1.0})
	
	pred = y_out
	return (pred-numpy.pi/2)/numpy.pi

def publishCommands(command):
	msg = TORCSCtrl()
	msg.header.stamp = rospy.Time.now()
	msg.steering = command
	pub.publish(msg)

if __name__ == '__main__':

	# Initializing ROS node
	rospy.init_node('Prediction_Node', anonymous = True)

	# Setting up node to publish commands to TORCS
	pub = rospy.Publisher('torcs_ros/arb', TORCSCtrl, queue_size = 100)

	# Subscribes to images from the ROS/TORCS bridge
	rospy.Subscriber('/torcs_ros/pov_image', Image, callback)
	rospy.spin()
