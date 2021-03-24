from __future__ import unicode_literals
# from src.BaseModel.model_Face import
from src.BaseModel.detector import FACE_DETECT
from src.BaseModel.recognizer import FaceRecognizer
from utils.camera_url import CameraURL
import threading
import configparser
import logging
import cv2
import os

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
cfg = configparser.ConfigParser()
cfg.read('config.ini')


class Face_id(object):
    def __init__(self):
        model_url = self.cfg.get('MODEL', 'model_url')

    def run(self):
        logging.info("Start App")
        CameraURL.run()



