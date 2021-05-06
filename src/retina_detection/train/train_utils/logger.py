""" A class which can be used to log information during a program's execution. """

import pathlib
import logging


class Log:
    def __init__(self, log_file: pathlib.Path) -> None:
        log_file.parent.mkdir(exist_ok=True, parents=True)
        fmt_str = "[%(asctime)s] %(levelname)s: %(message)s"
        # set up logging to file - see previous section for more details
        logging.basicConfig(
            level=logging.DEBUG,
            format=fmt_str,
            datefmt="%m/%d/%Y %H:%M:%S",
            filename=log_file,
            filemode="w",
        )
        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        # set a format which is simpler for console use
        formatter = logging.Formatter(fmt_str, datefmt="%m/%d/%Y %H:%M:%S")
        # tell the handler to use this format
        console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger("").addHandler(console)
        # to write metrics to tensorboard

    def info(self, message: str) -> None:
        logging.info(message)

    def warning(self, message: str) -> None:
        logging.warning(message)

    def error(self, message: str) -> None:
        logging.error(message)
