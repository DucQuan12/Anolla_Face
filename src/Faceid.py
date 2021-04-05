from __future__ import unicode_literals

import threading
import logging
import signal
import grpc
import time

from cfg import Config
from .utils.utils import util
from .grpc import server_pb2, server_pb2_grpc
from .src.search.search_feature import Search
from concurrent import futures

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
cfg = Config.config()


class FaceApp(object):
    def __init__(self):
        self.__model_url = cfg.get('MODEL', 'model_url')
        self.__key_crt = cfg.get('DEFAULT', 'key_crt')
        self.__key_server = cfg.get('DEFAULT', 'key_server')
        self.__index = Search.list_index()
        self.__util = util()
        self.__port = cfg.get('DEFAULT', 'port')

    def __lience(self):
        return False

    def serve(self):
        private_key = self.__util.read_key(self.__key_server)
        certificate = self.__util.read_key(self.__key_crt)

        server_certifical = grpc.ssl_server_credentials(((private_key, certificate, ), ))

        server = grpc.serve(futures.ThreadPoolExecutor(max_workers=2))
        server_pb2_grpc.add_FaceServiceServicer_to_server(server)
        server.add_insecure_port("[0.0.0.0]:"+self.__port)
        server.start()

        try:
            while True:
                logging.info("Server running: theadcount %i" % (threading.activeCount()))
                time.sleep(5)
        except KeyboardInterrupt:
            logging.error("KeyboardInterrupt")
            server.stop()

    def __search_feature(self, common=True):
        # self.feeature = model.predict()
        top_final = Search.search_feature()
        if common:
            return top_final
        else:
            return top_final[0]

    def run(self):
        logging.info("Start App")
        if self.lience() is True:
            start = time.time()
            logging.warning("Run Camera Url")
            logging.info("list index in elasticsearch: {}".format(self.index))

        else:
            logging.info("App no activate")

    def __str__(self):
        return self.__class__.__name__
