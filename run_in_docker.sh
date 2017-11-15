#!/bin/bash
CAM_INDEX=$1
xhost +local:
nvidia-docker run -it --rm --device="/dev/video0" --device="/dev/video1" -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY kylehounslow/pyyolov2:latest /bin/bash -c "git pull origin master; /usr/bin/python /home/pyyolov2/python/pyyolov2.py 0 $CAM_INDEX"


