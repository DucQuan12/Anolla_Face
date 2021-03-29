from __future__ import unicode_literals
# from src.BaseModel.model_Face import
# from .src.BaseModel.FaceDetect.detector import FACE_DETECT
# from .src.BaseModel.FaceVerify.recognizer import FaceRecognizer
from .src.search.search_feature import Search
from utils.camera_url import CameraURL
import threading
import configparser
import logging
import time
import cv2
import os

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
cfg = configparser.ConfigParser()
cfg.read('config.ini')


class Face_id(object):
    def __init__(self):
        model_url = self.cfg.get('MODEL', 'model_url')

    def lience(self):
        return False

    def _search_feature(self):
        self.feeature = model.predict()
        top_n = Search.search_feature()
        return top_n

    def run(self):
        logging.info("Start App")
        if self.lience() is True:
            start = time.time()
            logging.warning("Run Camera Url")
            list_object = CameraURL.run()
            

        else:
            logging.info("App no activate")
