from __future__ import unicode_literals
#from src.BaseModel.model_Face import
from src.BaseModel.detector import FACE_DETECT
import configparser
import logging
import cv2
import os


logging.basicConfig()


class Face_id(object):
    def __init__(self):
        cfg = configparser.ConfigParser()
        cfg.read('config.ini')
        camera = cfg.get('Default', 'camera_url')
        model = cfg.get('Model', 'model_url')

    def convert_image(self):
        pass

    def view(self):
        pass

    def run(self):
        pass
