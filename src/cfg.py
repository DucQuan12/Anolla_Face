import cfg
import configparser

class Config(object):
    def config(self):
        cfg = configparser.ConfigParser()
        return cfg.read("config.ini")
