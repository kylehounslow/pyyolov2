FROM dymat/opencv:cuda
# get latest pyyolov2 code and yolo model weights
RUN apt-get update
RUN apt-get install -y git
RUN git clone https://github.com/kylehounslow/pyyolov2.git /home
RUN mkdir -p /home/pyyolov2/model
RUN cd /home/pyyolov2/model
run apt-get install -y wget
RUN wget https://pjreddie.com/media/files/yolo.weights
# add pyyolov2.so and libcudnn.so.5 to library path
RUN LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/python/lib"