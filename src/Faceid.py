from __future__ import unicode_literals
# from src.BaseModel.model_Face import
# from .src.BaseModel.FaceDetect.detector import FACE_DETECT
# from .src.BaseModel.FaceVerify.recognizer import FaceRecognizer
from .src.search.search_feature import Search
from cfg import Config
from .grpc import grpc
from utils.camera_url import CameraURL
import threading
import logging
import time
import cv2
import os

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
cfg = Config.config()

class FaceApp(object):
    def __init__(self):
        __model_url = self.cfg.get('MODEL', 'model_url')
        __index = self.Search.list_index()

    def __lience(self):
        return False

    def __search_feature(self, common=True):
        # self.feeature = model.predict()
        top_final = Search.search_feature()
        if common:
            return top_final
        else:
            return top_final[0]

    def run(self):
        logging.info("Start App")
        if self.lience() is True:
            start = time.time()
            logging.warning("Run Camera Url")
            list_object = CameraURL.run()
            logging.info("list index in elasticsearch: {}".format(self.index))

        else:
            logging.info("App no activate")

    def __str__(self):
        return self.__class__.__name__

