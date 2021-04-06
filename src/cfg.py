import cfg
import configparser

class Config(object):
    @staticmethod
    def config():
        cfg = configparser.ConfigParser()
        return cfg.read("config.ini")
