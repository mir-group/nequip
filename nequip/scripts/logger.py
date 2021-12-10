import logging
import sys


def set_up_script_logger(logfile: str, verbose: str = "INFO"):
    # Configure the root logger so stuff gets printed
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.CRITICAL)
    root_logger.handlers = [
        logging.StreamHandler(sys.stderr),
        logging.StreamHandler(sys.stdout),
    ]
    level = getattr(logging, verbose.upper())
    root_logger.handlers[0].setLevel(level)
    root_logger.handlers[1].setLevel(logging.CRITICAL)
    if logfile is not None:
        root_logger.addHandler(logging.FileHandler(logfile, mode="w"))
        root_logger.handlers[-1].setLevel(level)
    return root_logger
