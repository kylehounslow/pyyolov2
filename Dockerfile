FROM dymat/opencv:cuda
# get latest pyyolov2 code and yolo model weights
RUN apt-get update
RUN apt-get install -y git
RUN git clone https://github.com/kylehounslow/pyyolov2.git /home/pyyolov2
# download the model
run apt-get install -y wget
RUN mkdir -p /home/model
RUN wget --no-check-certificate -P /home/model https://pjreddie.com/media/files/yolo.weights
# add pyyolov2.so and libcudnn.so.5 to library path
#RUN cp /home/pyyolov2/python/lib/* /usr/local/nvidia/lib64
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/python/lib"

