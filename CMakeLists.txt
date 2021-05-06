cmake_minimum_required(VERSION 2.8.3)
project(retinanet_ros)

find_package(catkin
    REQUIRED
    COMPONENTS
    roslib
    rospy
    std_msgs
    message_runtime
    message_generation
    actionlib
    actionlib_msgs
    darknet_ros_msgs
    )

catkin_python_setup()

generate_messages(
   DEPENDENCIES
   std_msgs
   actionlib_msgs
 )

catkin_package(
  CATKIN_DEPENDS actionlib_msgs
)

install(DIRECTORY scripts/
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  USE_SOURCE_PERMISSIONS
)

