#!/bin/bash

cd src
catkin_init_workspace
cd ..
catkin_make
# source devel/setup.bash ## doesn't seem to work here. call this manually
echo "Initialization is done"
echo "Remember to run: source devel/setup.bash"
