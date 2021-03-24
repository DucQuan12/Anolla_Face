# USAGE
# python recognize_faces_video.py --encodings encodings.pickle
# python recognize_faces_video.py --encodings encodings.pickle --output output/jurassic_park_trailer_output.avi --display 0

# import the necessary packages
from imutils.video import VideoStream, FileVideoStream
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras.preprocessing import image as img_keras
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
import nmslib
import numpy as np
import pathlib
import sys
import os
# project_name = 'faceid-medium'
# project_path = str(pathlib.Path().absolute())
# sys.path.append(project_path)
from searcher import search_nmslib_index
# from .detector import *
from keras.models import load_model
# folder_path = str(pathlib.Path().absolute()).split(project_name)[0] + project_name + '/'
# sys.path.append(folder_path)


class DistancesVoting():
    def __init__(self, index_path, data_embedded, dist_threshold, k_nearests):
        self.index = nmslib.init(method='hnsw', space='cosinesimil', data_type=nmslib.DataType.DENSE_VECTOR)
        self.index.loadIndex(index_path, load_data=True)
        self.dist_threshold = dist_threshold
        self.data_labels = data_embedded['ids']
        # self.unknown_id = len(np.unique(self.data_labels))
        self.unknown_id = 0
        self.k_nearests = k_nearests

    def get_final_predictions(self, predicted_labels, dists):
        # a=np.zeros((self.unknown_id + 1))
        a =  np.zeros(np.max(self.data_labels) + 1) 

        for i,label in enumerate(predicted_labels):
                # a[label]+=1.0/(i+1)
                a[label]+=1.0/(i+dists[i])
                # a[label]+=1.0/(dists[i])
        scores=sorted(a[a>0.0])

        ranked_labels=sorted(range(len(a)), key=a.__getitem__)
        print('ranked_labels', predicted_labels)
        # prop = list(map(lambda x: x/sum(scores), scores))
        # prop = prop[::-1][0]

        candidates =ranked_labels[-len(scores):]
        candidates =candidates[::-1]
        # print('props', props)

        # use mean distance as probability
        temp_index = np.where(np.array(predicted_labels) == candidates[0])[0]
        predicted_dists = np.array(dists)[temp_index]
        prop = 1- np.mean(predicted_dists)/0.54
        final_id = candidates[0]
        print('candidates', candidates)
        if prop < 0.5:
            final_id = self.unknown_id

        return final_id, prop

    def vote_distances_2(self, distances, f_ids, Y_train):
        D = np.array(distances)
        I = np.array(f_ids) 
        predictions = []
        dist_list = []
        for k in range(len(I)):
                la = int(Y_train[I[k]])
                if 1==1:
                    dis = D[k]
                    dist_list.append(dis)
                    if dis > self.dist_threshold :
                        predictions.append(self.unknown_id)
                        # predictions.append(self.unknown_id)
                    else:
                        predictions.append(la)
        # print('predictions', predictions)
        prediction, prop = self.get_final_predictions(predictions, dist_list)
        # print('final:', predictions)
        return prediction, prop

    def predict(self, embeddings):

        query_results = search_nmslib_index(self.index, embeddings, self.k_nearests)
        print('query_results', query_results)
        labels = []
        props = []
        for i, result in enumerate(query_results):
            distances = result[1]
            f_ids = result[0]
            name, prop = self.vote_distances_2(distances, f_ids, self.data_labels)
            labels.append(name)
            props.append(prop)
        print('labels', labels)
        print('props', props)
        return labels, props

    def predict_simple(self, embeddings):
        labels, distances = self.index.knnQuery(embeddings, k=10)
        print('labels', labels)
        print('distances', distances)

        props = [1- distances[0]]

        if props[0] < 0.5:
            labels = [self.unknown_id]

        return labels, props


class FaceRecognizer():
    def __init__(self, index_path= '', embedding_path= '',  dist_threshold=0.24):
        # EMBEDDING_PATH = setting.base_url + 'deep/embeddings/data_embedded_test.pickle'
        # INDEX_PATH = setting.base_url + 'deep/search_index/test_index'
        EMBEDDING_PATH = project_path + '/app/data_embedded.pickle'
        print('EMBEDDING_PATH', EMBEDDING_PATH)
        INDEX_PATH = project_path + '/app/nms_index'
        print('INDEX_PATH', INDEX_PATH)
        
        if os.path.isfile(EMBEDDING_PATH):
            DATA = pickle.loads(open(EMBEDDING_PATH, "rb").read())
            if os.path.isfile(INDEX_PATH):
                self.predictor = DistancesVoting(INDEX_PATH, DATA, dist_threshold, 7)
                self.unknown_id = self.predictor.unknown_id
            else:
                print ("MISSING INDEX_PATH !")
        else:
            print ("MISSING EMBEDDING_PATH !")
        
        # self.unknown_id = self.predictor.unknown_id
        self.vgg_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
        # print('vgg_model', self.vgg_model)

        # self.FN_model = load_model(folder_path + 'app/static/pretrain_models/facenet_keras.h5')

    def get_embedding_vggface(self, face_img):
        face_img = cv2.resize(face_img, (224, 224))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # convert into an array of samples
        face_img = img_keras.img_to_array(face_img)
        samples = np.expand_dims(face_img, axis=0)
        samples = preprocess_input(samples, version=2)
        # perform prediction
        print('samples', samples.shape)
        emb = self.vgg_model.predict(samples)
        return emb


    def get_embedding_FaceNet(self, face_pixels):
        face_pixels = cv2.resize(face_pixels, (160, 160))
        face_pixels = face_pixels.astype('float32')
        # standardize pixel values across channels (global)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        # face_pixels = normalize(face_pixels)
        # transform face into one sample
        samples = np.expand_dims(face_pixels, axis=0)
        embedding = self.FN_model.predict(samples)
        # in_encoder = Normalizer(norm='l2')
        # embedding = in_encoder.transform(embedding)
        return embedding

    def recognize(self, face_img, face_location=None):
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        embedding = self.get_embedding_vggface(face_img)[0]

        # if face_location is not None:
        #     embedding = face_recognition.face_encodings(face_img, face_location)[0]
        # else:
        #     embedding = face_recognition.face_encodings(face_img)[0]

        # embedding = get_embedding_FaceNet(face_img)[0]

        # Face recognition
        ids, props = self.predictor.predict(np.array([embedding]))
        id = ids[0] if ids[0] != self.unknown_id else 0

        return id, props[0]



if __name__ == '__main__':
    from detector import *

    frame_count = 0
    frames_skip = 1

    FACE_RECOGNIZER = FaceRecognizer(dist_threshold=0.28)

    DETECT_METHODS = {'1': mtcnn_detector, '2': haar_cascade_detector, '3': hog_detector, '4': FR_hog_detector, '5': faced_detector, '6': DNNDector}    

    detector = DETECT_METHODS[str(6)]()  
    
    cap = WebcamVideoStream(src=0)
    # cap.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.start()
    
    fps = FPS().start()
    while(True):

        # Capture frame-by-frame
        frame = cap.read()
        print('size', frame.shape)
        # frame = imutils.resize(frame, width=400)


        if frame_count % frames_skip == 0:
            print('processing') 

            # detect just a region of the original frame

            # locations = [100, 100, frame.shape[1] -100, frame.shape[0]]
            locations = [0, 0, frame.shape[1], frame.shape[0]]
            # sub_frame = extract_sub_region(frame, locations)
            sub_frame = frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            faces_bbs = detector.detect(sub_frame)
            # filter out small faces
            # faces_bbs = filter_smaller_bbs(faces_bbs, 120)

            if len(faces_bbs) > 0:
                main_face = max(faces_bbs, key = lambda x: (x[2]-x[0])*(x[3]-x[1]))
                boxes = [(int(t), int(r), int(b), int(l)) for (l, t, r, b) in faces_bbs]
                # t = time.time()
                # encodings = face_recognition.face_encodings(rgb, boxes)
                # print(f'---{time.time() - t} ----')
                for element in faces_bbs:
                    
                    left, top, right, bottom = np.array(main_face, dtype='int32')
                    pt1 = (int(left), int(top))
                    pt2 = (int(right), int(bottom))
                    face = sub_frame[top:bottom, left:right].copy()
                    print(face.shape)
                    print(face.dtype)
                    print(type(face))
                    emp = FACE_RECOGNIZER.get_embedding_vggface(face)[0]
                    print(emp)
                    
                    id, prop = FACE_RECOGNIZER.recognize(face)
                    

                    id = str(id) if id != 0 else 'Unknown'

                    cv2.rectangle(sub_frame, pt1, pt2, (0, 255 ,0), 2)
                    cv2.putText(sub_frame, id, (left+5,top+20), cv2.FONT_HERSHEY_SIMPLEX,
                                                0.5, (0, 225, 0), 2)


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
