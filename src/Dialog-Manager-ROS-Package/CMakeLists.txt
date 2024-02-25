cmake_minimum_required(VERSION 2.8.3)
project(dialog_manager)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  message_generation
)

add_service_files(FILES
  HandleSuggestion.srv
  HandleFrame.srv
  TTSRequest.srv
)

add_message_files(FILES
  FrameMessage.msg
)

catkin_install_python(PROGRAMS
  src/dialog_manager_script.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Generate added messages and services
generate_messages(
  DEPENDENCIES
  std_msgs
)

## Declare a catkin package
catkin_package(
  CATKIN_DEPENDS
)

## Build talker and listener
include_directories(include ${catkin_INCLUDE_DIRS})