import cfg
import configparser

class Config(object):
    @staticmethod
    def config():
        cfg = configparser.ConfigParser()
        cfg.read('config.ini')
        return cfg
