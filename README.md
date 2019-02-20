# AUTOMATED DRIVING USING DEEP LEARNING

Project repository of **Team MECH** for **ECBM4040 - Neural Networks and Deep Learning** taught by Dr. Zoran Kostic.

All rights reserved - Department of Electrical Engineering, Columbia University in the City of New York.

__Project Members:__ 

+ Brian Jin **bj2364**

+ Rayal Raj Prasad **rpn2108**

+ Rajinder Singh-Moon **rs3226**

This repository contains all files necessary to run the automated driving script, including training and testing.
Attached is also a project report written in a conference-format in PDF for a more theoretical understanding.

## DEPENDENCIES
The simulated project is run on the front-end using TORCS, utilizes ROS as the run-time system, and TensorFlow as
the deep learning framework of choice. Following are the dependencies to run the code:

1. TensorFlow
2. OpenCV (cv2)
3. TORCS 1.3.7 Competition
4. ROS Kinetic Kame
5. ROS/OpenCV Bridge

Note that the entire project has only been tested on Ubuntu 16.0.4 (Xenial) using Python3. Behavior on Python2 is undefined.

Following are the instructions to install said dependencies on the machine (which is a native/virtual Ubuntu environment):

- TensorFlow

Visit [official TensorFlow download page](https://www.tensorflow.org/install/install_linux).

- OpenCV

Visit [OpenCV's documentation for installation](https://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html).

- TORCS 1.3.7 Competition

Since we are going to utilize memory sharing, TORCS provides modules specifically for the SCR championship that we will use.
Visit [Florian Mirus' repository](https://github.com/fmirus/torcs-1.3.7) for the competition version with installation instructions.

- ROS Kinetic Kame

Visit [ROS Kinetic installation instructions](http://wiki.ros.org/kinetic/Installation).

- ROS/OpenCV Bridge

Refer the vision_opencv package [repository](http://wiki.ros.org/cv_bridge). It can be easily installed using pip.

Once the dependencies have been installed, the code can be run succesfully.

## TRAINING CODE

(Note that this step is completely optional, and the user can proceed directly to testing to use the pre-existing model)

To run the training code, we will make use of Jupyter Notebooks. Recommended method of installing is using [Anaconda](https://conda.io/docs/user-guide/install/linux.html).

To start the training process, run jupyter notebooks in the *Train_code* folder as follows.

```shell
cd .../Test_code
jupyter notebook
```

Once this is done, open the file *TeamMECH_nb.ipynb* through the Jupyter kernel. Detailed instructions to run the rest of the code can be found directly on the notebook file.

### DATA COLLECTION

(Note that this step is completely optional, and the user can proceed directly to training/testing to use the pre-existing data set)

This section only comes into play if the user wants to record new data for the model training. Once the data has been collected, the files are written in *.NPY* format which are readable by the training scripts. Once the data files are written, please refer to the next sub-section for instructions as to where to move the data files to and how to run the training on this data.

The data collection pipeline uses ROS and scripts that are located in the *Test_code* folder. First, we must build a ROS workspace that has all these files.

In the *Test_code* folder, copy the *src* folder and place it in a workspace_directory of your choice. To create this directory, simply run the following.

```shell
cd .../
mkdir workspace_directory
```

Placing the *src* folder within this directory, we must now proceed to build the workspace.

```shell
cd .../workspace_directory
catkin_make
```

If performed correctly, the directory should now have created "devel" and "build" folders. Verify this with the following lines of code.

```shell
cd .../workspace_directory
ls
```

Now, we start a ROS Core session. Note that each of the next few steps involve running a new terminal window.

```shell
roscore
```

```shell
cd .../workspace_directory
source devel/setup.bash
rosrun tf_self_drv file_writer.py
```

Now we have a ROS Node waiting to record training data. Note that *file_writer.py* can be modified to store as many training images as the user defines, and this can be done at any resolution desired. Now, the involved part here is to now modify the testing robot to not take commands from the TensorFlow back-end, but instead use the controller that leverages sensor data to automate training data collection. More detail on this implementation in the supplementary PDF.

Navigate to *.../workspace_directory/src/torcs_ros_drive_ctrl/src/* and open *torcs_drive_ctrl_node.cpp* using a text-editor of your choice. In this file, change lines 79 and 130 to use the controller predicted values by deleting underscore as follows:

```cpp
torcs_ctrl_out_.steering = steer;
```

Note that every time a C++ node has been modified, the workspace will have to be built again.

```shell
cd .../workspace_directory
catkin_make
```

Once this is done, we can run the data collector robot. First, we need to run TORCS.

```shell
torcs
```

Click on New Game -> Practice Game, and the loading screen should now be stuck and is waiting on the SCR server module. 

```shell
cd .../workspace_directory
source devel/setup.bash
roslaunch torcs_ros_bringup torcs_ros.launch
```

The *file_writer.py* ROS node will collect images and car control commands as labels until the buffer is reached (which is defaulted to 10,000) and once this is complete, it will write two timestamped *.NPY* files which contain the images and corresponding labels respectively. Note that the publish rate for the images has been set at 5 Hz to avoid highly redundant data sets.

Once the user has collected the *.NPY* file, the model can now be trained. Note that multiple *.NPY* files can be concatenated and used for training, and instructions on that will be specified in the following sub-section. Finally, before proceeding to training and then testing, we will need to revert changes in the *torcs_drive_ctrl_node.cpp* file that was previously modified. Lines 79 and 130 will need to be changed to the following line.

```cpp
torcs_ctrl_out_.steering = steer_;
```

And again, don't forget to build the workspace!

```shell
cd .../workspace_directory
catkin_make
```

### TRAINING MODEL


## TESTING CODE

This step assumes that all the dependencies have been installed (including TORCS competition version). Open a terminal window using Ctrl + Alt + t. Before this, a ROS workspace directory will have to be created in any arbitrary location of your choice.

 Note that the next three steps can be skipped if the data collection modules have been run as that installation also includes directions to install these scripts. Also, since data collection would have involved instantiating a workspace directory, we can utilize that workspace instead of the one created above.

```shell
cd .../
mkdir workspace_directory
```

Open the *Test_Code* folder and copy the *src* directory into a workspace directory. Once this is done, the workspace will have to be built, which is
the equivalent of code being compiled.

```shell
cd workspace_directory
catkin_make
```

If performed correctly, the directory should now have created "devel" and "build" folders. Verify this with the following lines of code.

```shell
cd .../workspace_directory
ls
```

To test the model, first we must initialize a ROS session. Note that each code block represents a separate terminal.

```shell
roscore
```

Once we start the ROS Core, we next start our TORCS session in a new terminal.

```shell
torcs
```

Click on New Game -> Practice Game, and the loading screen should now be stuck and is waiting on the SCR server module. Before that,
we need to first set up the testing scripts for the model, and also a Python/C++ interface.

```shell
cd .../workspace_directory
source devel/setup.bash
rosrun tf_self_drv predict.py
```

```shell
cd .../workspace_directory
source devel/setup.bash
rosrun torcs_ros_bringup test_sub
```

Once this is done, we can complete the TORCS/ROS bridge.

```shell
cd .../workspace_directory
source devel/setup.bash
roslaunch torcs_ros_bringup torcs_ros.launch
```

Once done, RViz should open up with a visualizer of TORCS, and the TORCS window itself should have started the race. The car will now start driving
autonomously. Note that the view of the car should not be changed as the net is trained exactly on what the screen shows and works only in the first-person view.

Note that all this assumes that you are trying to run the simulator on a pre-trained model. To run it on a newly trained model, first ensure that the following 4 files exist:

1. my_model_name.meta
2. my_model_name.index
3. my_model_name.data-00000-of-00001
4. checkpoint

To use this model, move all the files into the folder *.../workspace_directory/src/tf_self_drv/model/*. Once this is done, the *predict.py* script has to be modified to now restore this new model file. This can easily be done by changing the string value of the restored model - refer script comments. Once modified, the above code can be run by following the same steps.