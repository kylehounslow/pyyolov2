"""
PyYoloV2
A Python wrapper for DarkNet YOLO V2 object detector
"""
from __future__ import print_function
import ctypes
import os
import numpy as np
import cv2

__author__ = "Kyle Hounslow"


class BBox_C_Struct(ctypes.Structure):
    _fields_ = [("x1", ctypes.c_float),
                ("y1", ctypes.c_float),
                ("x2", ctypes.c_float),
                ("y2", ctypes.c_float),
                ("confidence", ctypes.c_float),
                ("cls", ctypes.c_int)]


class Detection(object):
    def __init__(self, bbox, confidence, class_name, color=(0, 255, 0)):
        self.bbox = bbox
        self.x1, self.y1, self.x2, self.y2 = self.bbox
        self.p1 = (self.x1, self.y1)
        self.p2 = (self.x2, self.y2)
        self.confidence = confidence
        self.class_name = class_name
        self.color = color


class PyYoloV2(object):
    def __init__(self, gpu_index=0):
        cwd = os.path.dirname(__file__)
        sys.path.append('lib/libcudnn.so.5')
        self.c_prog = ctypes.cdll.LoadLibrary(os.path.join(cwd, 'lib/libpyyolov2.so'))
        self.weights = os.path.join(cwd, '../model/yolo.weights')
        self.cfg = os.path.join(cwd, '../cfg/yolo.cfg')
        self.datacfg = os.path.join(cwd, '../cfg/coco.data')
        self.class_names_path = os.path.join(cwd, '../data/coco.names')

        # get class names by index
        with open(self.class_names_path) as f:
            class_names_raw = f.read()
        self.class_names = class_names_raw.split('\n')
        self.class_colors = []
        for cls in self.class_names:
            self.class_colors.append(tuple(np.random.randint(0, 255, 3)))

        self.c_prog.set_gpu_index(gpu_index)  # must set cuda device BEFORE loading net.
        self.load_net()

    def load_net(self):

        self.c_prog.load_net(self.cfg, self.weights)
        self.net_w = self.c_prog.get_net_width()
        self.net_h = self.c_prog.get_net_height()
        self.max_num_bboxes = self.c_prog.get_max_num_bboxes()

    def detect(self, img, thresh=0.5):
        img_net = cv2.resize(img.copy(), (self.net_w, self.net_h))
        img_net = cv2.cvtColor(img_net, cv2.COLOR_BGR2RGB)
        self.scale_factor = (
            float(img.shape[1]) / self.net_w,
            float(img.shape[0]) / self.net_h)  # used to scale bbox back to original image scale

        h, w, ch = img_net.shape
        c_img = img_net.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        all_bboxes_c = (self.max_num_bboxes * BBox_C_Struct)()
        self.c_prog.detect_objs(c_img, w, h, ch, ctypes.byref(all_bboxes_c))
        detections = []
        for bbox_c in all_bboxes_c:
            if bbox_c.confidence > thresh:
                #             print bbox_c.x1,bbox_c.y1,bbox_c.x2,bbox_c.y2
                det = Detection([int(bbox_c.x1 * self.scale_factor[0]), int(bbox_c.y1 * self.scale_factor[1]),
                                 int(bbox_c.x2 * self.scale_factor[0]), int(bbox_c.y2 * self.scale_factor[1])],
                                bbox_c.confidence,
                                str(self.class_names[bbox_c.cls]), self.class_colors[bbox_c.cls])
                detections.append(det)
        return detections


def rtsp_test():
    rtsp_addr = 'rtsp://admin:Istuary1127@10.0.80.7/cam/realmonitor?channel=1&subtype=1'
    #     rtsp_addr = '/home/kyle/Videos/lasse/ejner_hessel.mp4'
    rtsp_addr = '/home/kyle/Videos/lasse/lars_hein.mp4'
    c_prog = ctypes.cdll.LoadLibrary('lib/libpyyolov2.so')
    c_prog.rtsp_test(rtsp_addr)


# rtsp_test()
def on_change(num):
    pass


def demo(gpu_index=0, cam_index=0):
    threshold = 40
    vc = cv2.VideoCapture()
    vc.open(cam_index)
    yolo = PyYoloV2(gpu_index=gpu_index)
    cv2.namedWindow('PYYOLOV2')

    cv2.createTrackbar('treshold', 'PYYOLOV2', threshold, 100, on_change)
    vc.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:

        # img = cv2.imread('/home/kyle/Pictures/cantest/54c4155c4793.jpg')
        _, img = vc.read()
        threshold = cv2.getTrackbarPos('treshold', 'PYYOLOV2')
        detections = yolo.detect(img=img, thresh=float(threshold) / 100)
        for det in detections:
            cv2.rectangle(img, det.p1, det.p2, det.color, 4)
            cv2.putText(img, det.class_name.upper(), (det.x1, det.y1 - 5), 1, 1.4, det.color, 2)

        img = cv2.resize(img, (1920, 1080))  # resize bigger for larger demo screen

        cv2.imshow('PYYOLOV2', img)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break


if __name__ == '__main__':
    import sys

    args = sys.argv
    gpu_index = 0
    cam_index = 0
    if len(sys.argv) > 1:
        gpu_index = int(sys.argv[1])
        print('gpu_index={}'.format(gpu_index))
    if len(sys.argv) > 2:
        cam_index = int(sys.argv[2])
        print('cam_index={}'.format(cam_index))
    if len(sys.argv) > 3:
        output_video = sys.argv[3]
        print('argv[3]={}'.format(sys.argv[3]))
    demo(gpu_index=gpu_index, cam_index=cam_index)
