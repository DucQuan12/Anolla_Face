from __future__ import unicode_literals

import threading
import logging
import grpc
import time

from cfg import Config
from utils.utils import util
import server_pb2_grpc
import server_pb2
from src.Base.search_feature import Search
from grpc_server import ShowVideoStream, Greeter
from concurrent import futures
from NeatLogger import Log

cfg = Config.config()
log = Log()
logger = log.get_logger()


class FaceApp(object):
    def __init__(self):
        self.__key_crt = cfg.get('DEFAULT', 'key_crt')
        self.__key_server = cfg.get('DEFAULT', 'key_server')
        self.__util = util()
        self.__port = cfg.get('DEFAULT', 'port')

    def lience(self):
        return True

    def serve(self):
        logger.info('===== server start =====')
        _key_crt = cfg.get('DEFAULT', 'key_crt')
        _key_server = cfg.get('DEFAULT', 'key_server')

        private_key = util.read_key(_key_server)
        certificate = util.read_key(_key_crt)
        server_certifical = grpc.ssl_server_credentials(((private_key, certificate,),))
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        server_pb2_grpc.add_FaceServiceServicer_to_server(Greeter, server)
        server.add_insecure_port('[::]:50070')
        server.start()
        try:
            while True:
                time.sleep(0)
        except KeyboardInterrupt:
            server.stop(0)

    def _search_feature(self, common=True):
        top_final = Search.search_feature()
        if common:
            return top_final
        else:
            return top_final[0]

    @serve
    def run(self, func):
        if self.lience is True:
            self.func
        else:
            logger.error("App no active")

    def __str__(self):
        return self.__class__.__name__
