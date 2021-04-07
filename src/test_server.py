from concurrent import futures
import grpc
import time
import os
import cv2
import sys
import server_pb2
import server_pb2_grpc
import base64
import numpy as np
from cfg import Config
from utils.utils import util

cfg = Config.config()


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

    # ==========
    def __init__(self):
        pass

    # ==========
    def getStream(self, request_iterator, context):
        timer = 0

        for req in request_iterator:
            print('process time = ' + str(time.clock() - timer))
            timer = time.clock()

            # decode from base64
            b64d = base64.b64decode(req.datas)
            # print("base64 decode size : ", sys.getsizeof(b64d))

            # base64 buffer to uint8
            dBuf = np.frombuffer(b64d, dtype=np.uint8)
            # print("buffer size : ", sys.getsizeof(dBuf))

            # decode to cv2
            dst = cv2.imdecode(dBuf, cv2.IMREAD_COLOR)
            # print("dst size : ", sys.getsizeof(dst))

            # set pixels
            show.set(dst)

            # success
            yield server_pb2.Reply(id=1)


show = ShowVideoStream()


# ============================================================
# functions
# ============================================================
def serve():
    print('===== server start =====')
    _key_crt = cfg.get('DEFAULT', 'key_crt')
    _key_server = cfg.get('DEFAULT', 'key_server')

    private_key = util.read_key(_key_server)
    certificate = util.read_key(_key_crt)
    server_certifical = grpc.ssl_server_credentials(((private_key, certificate,),))
    print(server_certifical)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    server_pb2_grpc.add_FaceServiceServicer_to_server(Greeter(), server)
    server.add_insecure_port('[::]:50070')
    server.start()
    try:
        while True:
            time.sleep(0)

    except KeyboardInterrupt:
        server.stop(0)

# ============================================================
# main
# ============================================================
if __name__ == '__main__':
    show.start()
    serve()

# ============================================================
# after the App exit
# ============================================================
cv2.destroyAllWindows()
