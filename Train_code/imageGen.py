#!/usr/bin/env/ python
# This script is used primarily for Data Augmentation. Code inspired from Assignment 2.

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import rotate
from scipy import misc


class ImageGenerator(object):

	def __init__(self, x, y,img_size = 32):

		self.x = x
		self.y = y
		self.num_of_samples,self.height,self.width,self.channels = np.shape(self.x)
		self.num_pixels_trans = 0
		self.degree_of_rot = 0
		self.is_horizontal_flip = False
		self.is_vertical_flip = False
		self.is_add_noise = False

	def next_batch_gen(self, batch_size, shuffle=True):
		
		num_of_samples = self.num_of_samples
		x = self.x
		y = self.y
		total_batch_count = num_of_samples // batch_size
		batch_count = 0 # counter for batches
		mask = np.arange(num_of_samples)
		while True:
			if (batch_count<total_batch_count):
				batch_count += 1
				yield(x[(batch_count-1)*batch_size:batch_count*batch_size,:],y[(batch_count-1)*batch_size:batch_count*batch_size])
			else:                
				np.random.shuffle(mask)
				x = x[mask]
				y = y[mask]
				batch_count = 0


	def show(self):
		images = []
		for i in range(16):
			
			images.append(self.x[i,:,:,:])

		f, axarr = plt.subplots(4, 4, figsize=(12,12))
		for i in range(4):
			for j in range(4):
				img = images[4*i+j]
				axarr[i][j].imshow(img)

	def translate(self, shift_height, shift_width):

		#Translate self.x according to the mode specified

		x = self.x

		x = np.roll(x,shift_height,axis=2)
		x = np.roll(x,shift_width,axis=1)

		self.x = x
		self.num_pixels_trans = shift_height + shift_width

	def rotate(self, angle=0.0):
		
        #Rotate self.x by the angles (in degree) given.
        
		x = self.x
		x = rotate(x,angle,axes=(1,2),reshape = False)
		self.x = x
		self.degree_of_rot = angle

	def flip(self, mode='h'):
		
		#Flip self.x according to the mode specified
		
		x = self.x
		if(mode=='h'):
			self.is_horizontal_flip = True
			x = np.flip(x,1)
		elif(mode=='v'):
			self.is_vertical_flip = True
			x = np.flip(x,2)
		elif(mode =='hv'):
			self.is_horizontal_flip = True
			self.is_vertical_flip = True
			x = np.flip(x,1)
			x = np.flip(x,2)

		self.x = x


	def add_noise(self, portion, amplitude):
		
		#Add random integer noise to self.x.
		
		self.is_add_noise = True
		x = self.x
		height = self.height
		width = self.width
		channels = self.channels

		num_of_samples = self.num_of_samples

		noisy_samples_size = int(portion*num_of_samples)

		noise = np.random.normal(0,amplitude,(noisy_samples_size,height,width,channels))

		mask = np.arange(num_of_samples)
		np.random.shuffle(mask)


		x[mask[:noisy_samples_size]] = x[mask[:noisy_samples_size]] + noise

		self.x = x
