import logging
import sys

def set_up_script_logger(logfile:str):
    # Configure the root logger so stuff gets printed
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.CRITICAL)
    root_logger.handlers = [
        logging.StreamHandler(sys.stdout),
        logging.StreamHandler(sys.stderr),
    ]
    root_logger.handlers[0].setLevel(logging.INFO)
    root_logger.handlers[1].setLevel(logging.CRITICAL)
    if logfile is not None:
        root_logger.addHandler(logging.FileHandler(logfile, mode="w"))
        root_logger.handlers[-1].setLevel(logging.INFO)
    return root_logger
