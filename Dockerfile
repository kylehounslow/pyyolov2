FROM dymat/opencv:cuda
# get latest pyyolov2 code and yolo model weights
RUN git clone https://github.com/kylehounslow/pyyolov2.git /home
RUN mkdir -p /home/pyyolov2/model
RUN wget https://pjreddie.com/media/files/yolo.weights /home/pyyolov2/model
# add pyyolov2.so and libcudnn.so.5 to library path
RUN LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/python/lib"