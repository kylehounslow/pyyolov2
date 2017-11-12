FROM dymat/opencv:cuda
# get latest pyyolov2 code
RUN git clone
RUN wget https://pjreddie.com/media/files/yolo.weights
# add pyyolov2.so and libcudnn.so.5 to library path
RUN LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/python/lib"