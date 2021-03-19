import cv2

class ImgProcess(object):
    def __init__(self, *args):
        pass
        
    def resize(self, img):
        return cv2.resize(img, (224, 224))

    def cvtColor(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def process(self, img):
        img = self.resize(img)
        img = self.cvtColor(img)
        return img