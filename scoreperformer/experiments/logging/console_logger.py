""" Console and File Logger. """
from typing import Optional

from loguru import logger


def setup_logger(file: Optional[str] = None, **extra):
    import sys

    time = "<green>{time:YYYY-MM-DD HH:mm:ss}</green>"
    level = "<level>{level:<7}</level>"
    process = "<level>{extra[process]}</level>"
    node_info = "<level>{extra[node_info]}</level>"
    module = "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>"
    message = "<level>{message}</level>"

    # formatter = f"{time} {level} {module} - {message_level:6s} {message}"
    # formatter = f"{time} {level} - {message_level:6s} {message}"
    # formatter = f"{time} {level} {module} - {message}"
    formatter = f"{time} {level} - {message}"
    handlers = [dict(sink=sys.stdout, format=formatter, enqueue=True)]
    if file is not None:
        handlers.append(dict(sink=file, format=formatter, enqueue=True))

    logger.configure(
        handlers=handlers,
        extra=extra
    )

    return logger
