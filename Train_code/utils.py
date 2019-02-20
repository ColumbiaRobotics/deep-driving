#!/usr/bin/env python
# This code is primarily used to download, unzip and pre-process the data before training on the modified AlexNet.

import _pickle as pickle
import os
import zipfile
import glob
import urllib.request as url
import numpy as np
import scipy.misc
from PIL import Image

import csv

# download our data set from OneDrive link - ensure that folder "Data" does not exist!
def download_data():
    if not os.path.exists('./Data/all_data.tar.gz'):
        os.mkdir('./Data')
        print('Start downloading data...')
        url.urlretrieve("https://onedrive.live.com/download?cid=71B264C5094F6C26&resid=71B264C5094F6C26%2152523&authkey=AE4nxs-SvK2oec0",
                        "./Data/all_data.tar.gz")
        print('Download complete.')
    else:
        if os.path.exists('./Data/add_data.tar.gz'):
            print('Image files already exist')

# Unzip the data
def load_data():
    if not os.path.exists('./Data/all_data.tar,gz'):
        download_data()
    else:
        print("Zip file already exists!")


    root_dir = os.getcwd()
    os.chdir('./Data')

    # extract images

    if not os.path.exists('./Cropped'):
        print("Extracting...")
        os.system("tar -xvzf all_data.tar.gz")


    print("Data extracted!")
    os.chdir(root_dir)

# Pre-process the data to make it ready for training
def process_crop_data():

    root = os.getcwd()
    crop_data_dir = root + '/Data/Cropped/Train'
    crop_data_names = os.listdir(crop_data_dir)
    
    crop_label_dir = root + '/Data/Cropped/Labels'
    crop_label_names = os.listdir(crop_label_dir)
    
    crop_data_names.sort()
    
    crop_label_names.sort()
    
    cropData = np.array([],dtype=np.float32).reshape(0,90,250,3)
    
    cropLabel = np.array([],dtype=np.float32).reshape(0,7)
    
    print("Loading Cropped Data...")
    for name in crop_data_names:
        print('Loading '+name+' Images...')
        name = crop_data_dir + '/' +name
        cropData = np.concatenate([cropData,np.load(name)],axis=0)
        
        
    print('Loading Cropped Labels...')
    for name in crop_label_names:
        print('Loading '+name+' Labels...')
        name = crop_label_dir+ '/' +name
        cropLabel = np.concatenate([cropLabel,np.load(name)],axis=0)
        
    
    print("Data loaded!")

    # return list of image data and corresponding labels
    return [cropData, cropLabel]
