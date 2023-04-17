import logging
from logging import handlers


class Logger:
    __instance = None

    def __new__(cls, *args, **kwargs):
        if not cls.__instance:
            cls.__instance = object.__new__(cls, *args, **kwargs)
        return cls.__instance

    def __init__(self):
        formater = logging.Formatter(
            '[%(asctime)s]-[%(levelname)s]-[%(filename)s]-[%(funcName)s:%(lineno)d] : %(message)s')
        self.logger = logging.getLogger('main')
        self.logger.setLevel(logging.INFO)

        self.console = logging.StreamHandler()
        self.console.setLevel(logging.DEBUG)
        self.console.setFormatter(formater)
        self.logger.addHandler(self.console)

        # self.fileLogger = handlers.RotatingFileHandler("./Easysacle.log", maxBytes=5242880, backupCount=3)
        # self.fileLogger.setFormatter(formater)
        # self.logger.addHandler(self.fileLogger)

    def get_logger(self):
        return self.logger

my_logger = Logger().get_logger()


if __name__ == '__main__':
    logger = Logger().get_logger()
    logger.debug('debug')
    logger.info('info')
    logger.warning('warning')