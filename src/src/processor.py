from Face.VGGFace2 import ModelFace
from keras_vggface.utils import preprocess_input
import numpy as np
import cv2


class ModelProcessor(object):
    def __init__(self):
        self.vgg_model = ModelFace.load_model()
        
    def predict(self, image):
        img = cv2.resize(image, (224, 224))
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        samples = np.expand_dims(img, axis=0)
        samples = preprocess_input(samples, version=2)
        embeddings = self.vgg_model.predict(samples)
        return embeddings

