# from src.BaseModel.detector import FACE_DETECT
# from src.BaseModel.recognizer import FaceRecognizer
from src.Data_Processing.face_aligner import FaceAligner
from imutils.video import WebcamVideoStream
import face_recognition
import configparser
import numpy as np
from imutils.video import FPS
import imutils
import cv2
import os
import sys
import logging

cfg = configparser.ConfigParser()
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


class CameraURL(FACE_DETECT):

    def __init__(self):
        camera_url = self.cfg.getint('DEFAULT', 'camera_url')
        _method_detect = self.cfg.get('MODEL', 'model_name')
        _thresh = self.cfg.getint('PARAMETER', 'thresh')
        frame_count = self.cfg.getint('PARAMETER', 'frame_count')
        frame_skip = self.cfg.getint('PARAMETER', 'frame_skip')
        number_skip = self.cfg.getint('PARAMETER', 'number_skip')
        self.list_image = 0

    def face_detect(self):
        logging.warning("USE CAMERA:{}".format(self.camera_url))
        cap = WebcamVideoStream(self.camera_url)
        cap.start()
        fps = FPS().start()
        while True:
            list_face = []
            logging.debug("LOAD CAMERA_URL CREATE FRAME")
            _, frame = cap.read()
            frame = imutils.resize(frame, width=640)
            logging.WARNING("LOAD METHOD DETECT FACE:{}".format(self._method_detect))
            if self._method_detect is 'face_recognition':
                face_locations = face_recognition.face_locations(frame)
                if face_locations:
                    face_locations = [max(face_locations, key=lambda x: (x[2] - x[0]) * (x[1] - x[3]))]
                    og_img_size = np.asarray(frame.shape)
                    resized_img_size = np.asarray(frame.shape)
                    w_ratio = int(og_img_size[1] / resized_img_size[1])
                    h_ratio = int(og_img_size[0] / resized_img_size[0])
                    for loc in face_locations:
                        top, right, bottom, left = loc
                        bb = np.absolute(np.array(loc).astype(np.int32))
                        bb_og = np.copy(bb)
                        bb_og[0] = left * w_ratio
                        bb_og[1] = top * h_ratio
                        bb_og[2] = right * w_ratio
                        bb_og[3] = bottom * h_ratio
                        align_padding = 45
                        print('img', frame.shape)
                        try:
                            aligned_face = FaceAligner.align(frame, bb_og, align_padding)
                            list_face.append(aligned_face)
                            if len(list_face) == self.number_skip:
                                list_face = []
                        except:
                            logging.error("None Image In List")

        else:
            detector = self.detect_hog()
