from concurrent import futures
import threading
import logging
import time
import grpc
import server_pb2
import server_pb2_grpc

logging.basicConfig()


class Listener(server_pb2_grpc.FaceServiceServicer):

    def __init__(self):
        self.counter = 0
        self.last_time = time.time()

    def __str__(self):
        return self.__class__.__name__

    def ping(self, request, context):
        self.counter = 1
        if self.counter > 10000:
            self.last_time = time.time()
            self.counter = 0
        return server_pb2.Pong(count=request.count)

def serve():
    server = grpc.serve(futures.ThreadPoolExecutor(max_workers=2))
    server_pb2_grpc.add_FaceServiceServicer_to_server(Listener(), server)
    server.add_insecure_port("[::]:50070")
    server.start()
    try:
        while True:
            logging.info("Server running: theadcount %i"%(threading.activeCount()))
            time.sleep(5)
    except KeyboardInterrupt:
        logging.error("KeyboardInterrupt")
        server.stop()

