import time
import os
import sys


class util(object):

    @staticmethod
    def read_key(pathfile):
        with open(pathfile, 'rb') as f:
            key_sever = f.read()
        return key_sever
