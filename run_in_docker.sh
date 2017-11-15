nvidia-docker run -it --rm --device=/dev/video0 -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -v $PWD:/home kylehounslow/pyyolov2:latest /usr/bin/python /home/python/pyyolov2.py

