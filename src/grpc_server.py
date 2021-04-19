from concurrent import futures
import server_pb2
import server_pb2_grpc
import base64
import time
import cv2
import numpy as np
from cfg import Config
from utils.utils import util
from NeatLogger import Log

cfg = Config.config()
log = Log()
logger = log.get_logger()

class ShowVideoStream(object):
    img = None
    thread = futures.ThreadPoolExecutor(max_workers=1)

    def start(self):
        self.thread.submit(self.ShowWindow)

    def set(self, img):
        self.img = img

    def ShowWindow(self):
        while True:
            if self.img is not None:
                cv2.imshow('dst Image', self.img)
                k = cv2.waitKey(1)
                if k == 27:
                    break


class Greeter(server_pb2_grpc.FaceServiceServicer):
    def __init__(self):
        self.show = ShowVideoStream()

    def getStream(self, request_iterator, context):
        timer = 0

        for req in request_iterator:
            timer = time.clock()
            b64d = base64.b64decode(req.datas)
            d_buf = np.frombuffer(b64d, dtype=np.uint8)
            logger.info(d_buf)
            dst = cv2.imdecode(d_buf, cv2.IMREAD_COLOR)
            self.show.start()
            self.show.set(dst)
            yield server_pb2.Reply(id=1)
