import cv2
import dlib
import os
import numpy as np


class FaceAligner:
    def __init__(self, predictor_path, face_size=224):
        predictor_path = predictor_path
        if os.path.isfile(predictor_path):
            pass

        predictor = dlib.shape_predictor(predictor_path)
        
        padding = 45
        desiredFaceWidth = 224 + padding*2
        fa = FaceAligner(predictor, desiredFaceWidth=self.desiredFaceWidth)

    def align(self, image, bb, padding=45):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # left, top, right, bottom = bb
        rect = dlib.rectangle(int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))
        face_aligned = self.fa.align(image, gray, rect)
        face_aligned = face_aligned[self.padding: self.desiredFaceWidth-self.padding, self.padding: self.desiredFaceWidth-self.padding]
        return face_aligned
