import dlib
import setting
import face_recognition
import cv2
from imutils.video import FPS
import imutils
# from utils import *
import numpy as np
import os
import time
from imutils.video import WebcamVideoStream
from faced import FaceDetector
from faced.utils import annotate_image
from mtcnn import MTCNN

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"


class FACE_DETECT(object):
    def __init__(self, conf_thres=0.9):
        self.mtcnn_net = MTCNN()
        self.conf_thres = conf_thres
        self.face_cascade = cv2.CascadeClassifier('../pretrained_models/haarcascade_frontalface_default.xml')
        self.dlib_hog_detector = dlib.get_frontal_face_detector()
        self.face_detector = FaceDetector()
        self.thresh = 0.8

    def faced_detect(self, frame):  # faced_detector
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bboxes = self.face_detector.predict(rgb_img, self.thresh)
        faces = list(map(lambda x: [x[0] - x[2] / 2, x[1] - x[3] / 2, x[0] + x[2] / 2, x[1] + x[3] / 2], bboxes))
        return faces

    def detect_mtcnn(self, frame, conf=False):  # mtcnn
        faces = np.array(self.mtcnn_net.detect_faces(frame))
        faces = list(filter(lambda x: x['confidence'] > self.conf_thres, faces))
        faces_locs = list(map(lambda x: [x['box'][0], x['box'][1] + x['box'][3] * 18 / 100, x['box'][0] + x['box'][2],
                                         x['box'][1] + x['box'][3] * 96 / 100], faces))
        confidences = list(map(lambda x: x['confidence'], faces))
        return (confidences, faces_locs) if conf else faces_locs

    def detect_haar(self, frame):  # detect_haar
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))
        faces = list(map(lambda x: [x[0], x[1], x[0] + x[2], x[1] + x[3]], faces))
        return faces

    @staticmethod
    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, number_of_times_to_upsample=0, model="hog")
        faces = list(map(lambda x: [x[3], x[0], x[1], x[2]], boxes))
        return faces

