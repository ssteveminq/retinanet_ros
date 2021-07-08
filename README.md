# Retinanet_ros

Object recognition algorithm 

## Make virtual environment with python 3.8
### Method1. Anaconda 

- Here, I used "py38" as the name of virtual environment

```
conda create -n py38 python=3.8 anaconda
```

- Activate your virtual environment
```
source activate py38
```

### Method2. Python virtual environment

(Not confirmed yet)
```
python3 -m venv ~/.envs/py38
source ~/.envs/py38/bin/activate
python3 -m pip install --upgrade pip
```


## Dependencies

- Make a new catkin workspace and clone the retinanet_ros package

```
mkdir py38_ws
cd py38_ws
mkdir src
git clone https://github.com/ssteveminq/retinanet_ros.git
cd retinanet_ros/doc
```

- Install required packages using pip install ( should use pip install virtual environment)

- You should confirm that you are using correct pip

```
which pip
```
output should be like "/home/$user_name/anaconda3/envs/$environment_name/bin/pip"

```
pip install -r requirements.txt  (**Anaconda**)
python3 -m pip install -r requirements.txt  (**python virtual environement**)
```


## Compile

```
source /opt/ros/melodic/setup.bash
cd py38_ws/
catkin build
```

- clone and build darknet_ros_msgs package 
```
git clone git@github.com:leggedrobotics/darknet_ros.git
cd darknet_ros/darknet_ros
touch CATKIN_IGNORE
cd ..
catkin build darknet_ros_msgs
```

- build retinanet_ros package
```
catkin build retinanet_ros
```

## Run

- Download pre-trained model to detect tire.

- (Tire) Go To the following link: https://drive.google.com/file/d/16DihIWDBkDreBPA_yqBTu9E3Gawq0QJ4/view?usp=sharing


- (Barrel) Go to the following link: https://drive.google.com/drive/folders/1G01Ag1YkqYPmBxdPZ7Sv2JJvzCNyGDax?usp=sharing

- Download the zip file and extract it into a folder called **~/runs/tire-detector/2021-04-22T11.25.25**

- run the test code 
```
source /py38_ws/devel/setup.bash
rosrun retinanet_ros test.py
```

- You might have to change the topic name for image topic.


## Contributer
- Alexander Witt [`@alexwitt23`](https://github.com/alexwitt23)
- Minkyu Kim
- Ryan Gupta



