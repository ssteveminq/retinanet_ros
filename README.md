
# Method1. Anaconda 
# Make virtual environment with python 3.8
'''
conda create -n py38 python=3.8 anaconda
'''

Activate your virtual environment
'''
source activate py38
'''

# Method2. Python virtual environment

TODo:



## Compile

make a new catkin workspaces 

'''
mkdir py38_ws
cd py38_ws
mkdir src
cd ..
catkin build
'''


Go to src foler and clone the retinanet_ros package
'''
cd py38_ws/src
git clone https://github.com/ssteveminq/retinanet_ros.git
cd retinanet_ros/doc
'''


Install required packages using pip install

'''
pip install -r requirement.txt
'''


Use pre-trained model to detect tire

Go To the following link: https://drive.google.com/drive/folders/1_XYbRO9vCr21UbtI8nFXSPIuvH-Imk4T?usp=sharing

Download the zip file and extract it into a folder called ~/runs/tire-detector/2021-04-22T11.25.25

'''
git clone https://github.com/ssteveminq/retinanet_ros.git
cd retinanet_ros/doc
pip install -r requirement.txt
''''

build darknet_ros_msgs package
'''
git clone git@github.com:leggedrobotics/darknet_ros.git
cd darknet_ros/darknet_ros
touch CATKIN_IGNORE
cd ../darknet_
catkin build darknet_ros_msgs
'''






