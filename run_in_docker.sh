#!/bin/bash
GPU_DEVICE=$2
VID_INPUT=$1
xhost +local:
nvidia-docker run -it --rm --device="/dev/video1" -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY kylehounslow/pyyolov2:latest /usr/bin/python /home/pyyolov2/python/pyyolov2.py 0 "$VID_INPUT"

