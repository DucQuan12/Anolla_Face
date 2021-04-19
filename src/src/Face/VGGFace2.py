import keras
from keras_vggface.vggface import VGGFace
import os


class ModelFace(object):
    def __init__(self):
        pass
    @staticmethod
    def load_model():
        vgg_face = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
        return vgg_face
