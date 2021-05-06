
# Anaconda 
# Make virtual environment with python 3.8
'''
conda create -n py38 python=3.8 anaconda
source activate py38
''''
make new workspaces

'''
mkdir py38_ws
cd py38_ws
mkdir src
cd ..
catkin build
'''

clone the retinanet_ros
'''
git clone https://github.com/ssteveminq/retinanet_ros.git
cd retinanet_ros/doc
pip install -r requirement.txt
''''

Use pre-trained model to detect tire

Go To the following link: https://drive.google.com/drive/folders/1_XYbRO9vCr21UbtI8nFXSPIuvH-Imk4T?usp=sharing

Download the zip file and extract it into a folder called ~/runs/tire-detector/2021-04-22T11.25.25



