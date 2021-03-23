import dlib
import setting
import face_recognition
import cv2
from imutils.video import FPS
import imutils
# from utils import *
import numpy as np
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
import time
from imutils.video import WebcamVideoStream
from faced import FaceDetector
from faced.utils import annotate_image
from mtcnn import MTCNN


class faced_detector(object):
    def __init__(self, thresh = 0.8):
        face_detector = FaceDetector()

    def detect(self, frame):
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bboxes = self.face_detector.predict(rgb_img, self.thresh)
        faces = list(map(lambda x: [x[0]-x[2]/2, x[1]-x[3]/2, x[0]+x[2]/2, x[1]+x[3]/2], bboxes))
        return faces

class mtcnn_detector(object):
    def __init__(self, conf_thres=0.9):
        self.mtcnn_net = MTCNN()
        self.conf_thres = conf_thres
        self.face_cascade = cv2.CascadeClassifier('../pretrained_models/haarcascade_frontalface_default.xml')
        self.dlib_hog_detector = dlib.get_frontal_face_detector()

    def detect_mtcnn(self, frame, conf = False): #mtcnn
        faces = np.array(self.mtcnn_net.detect_faces(frame))
        faces = list(filter(lambda x: x['confidence'] > self.conf_thres, faces))
        faces_locs = list(map(lambda x: [x['box'][0], x['box'][1] + x['box'][3] * 18/100, x['box'][0]+x['box'][2], x['box'][1]+x['box'][3] * 96/100], faces))
        confidences = list(map(lambda x: x['confidence'], faces))
        return (confidences, faces_locs) if conf else faces_locs

    def detect_haar(self, frame): #detect_haar
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(50,50))
        faces = list(map(lambda x: [x[0], x[1], x[0]+x[2], x[1]+x[3]], faces))
        return faces

    def detect_hog(self, frame):
        # dets = self.dlib_hog_detector(frame, 0)
        dets, scores, idx = self.dlib_hog_detector.run(frame, 0, 0)
        faces = list(map(lambda x: [x.left(), x.top(), x.right(), x.bottom()], dets))
        return faces

    @staticmethod
    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, number_of_times_to_upsample=0, model="hog")
        faces = list(map(lambda x: [x[3], x[0], x[1], x[2]], boxes))
        return faces



if __name__ == '__main__':
    frame_count = 0
    frames_skip = 1

    detect_methods = {'1': mtcnn_detector, '2': haar_cascade_detector, '3': hog_detector,
                    '4': FR_hog_detector, '5': faced_detector, '6': DNNDector}

    detector = detect_methods['1'](0.9)  
    
    cap = WebcamVideoStream(src=0)
    # cap.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    # cap.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    cap.start()
    
    fps = FPS().start()
    while(True):

        # Capture frame-by-frame
        frame = cap.read()
        print('size', frame.shape)
        frame = imutils.resize(frame, width=600)
    	# frame = cv2.resize(frame, (300, 300))



        if frame_count % frames_skip == 0:
            print('processing') 

            # detect just a region of the original frame
            # locations = [100, 100, frame.shape[1] -100, frame.shape[0]]
            # locations = [0, 0, frame.shape[1], frame.shape[0]]
            # sub_frame = extract_sub_region(frame, locations)
            sub_frame = frame
            
            faces_bbs = detector.detect(sub_frame)
            # filter out small faces
            # faces_bbs = filter_smaller_bbs(faces_bbs, 120)

            if len(faces_bbs) > 0:
                # uncomment to get only the biggest face
                # faces_bbs = [max(faces_bbs, key = lambda x: (x[2]-x[0])*(x[3]-x[1]))]
                for box in faces_bbs:
                    left, top, right, bottom = box
                    pt1 = (int(left), int(top))
                    pt2 = (int(right), int(bottom))
                    # Draw bounding boxes
                    cv2.rectangle(sub_frame, pt1, pt2, (0, 255 ,0), 2)


        frame_count += 1
        cv2.imshow('frame', frame)
        fps.update()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    fps.stop()

    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
