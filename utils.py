import logging
from logging.handlers import RotatingFileHandler

log_format = '%(asctime)s - %(name)s - %(levelname)s - ' \
             '%(funcName)s(%(lineno)d)- %(message)s'


def set_up_logging():
    log_file = 'deeplens.log'
    handlers = [get_console_handler(), get_log_file_handler(log_file)]
    logging.basicConfig(level=logging.INFO, format=log_format,
                        handlers=handlers)
    logging.info("started logging to: " + log_file)


def get_log_file_handler(log_file):
    fh = RotatingFileHandler(log_file, mode='a', maxBytes=5 * 1024 * 1024,
                             backupCount=3, encoding=None, delay=0)
    # create formatter
    formatter = logging.Formatter(log_format)
    # add formatter
    fh.setFormatter(formatter)
    return fh


def get_console_handler():
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    # create formatter
    formatter = logging.Formatter(log_format)
    # add formatter
    ch.setFormatter(formatter)
    return ch


def get_logger(name=__name__):
    # create logger
    logger = logging.getLogger(name)
    return logger
