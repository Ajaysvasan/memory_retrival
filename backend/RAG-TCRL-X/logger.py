import logging
import sys
from pathlib import Path
from datetime import datetime


class Logger:
    """Centralized logging with structured output"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self.logger = logging.getLogger("RAG_TCRL_X")
        self.logger.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)

        # File handler
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = (
            log_dir / f"rag_tcrl_x_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
        )
        file_handler.setFormatter(file_formatter)

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def get_logger(self, name: str = None):
        if name:
            return logging.getLogger(f"RAG_TCRL_X.{name}")
        return self.logger

    @staticmethod
    def info(msg: str, **kwargs):
        Logger().logger.info(msg, **kwargs)

    @staticmethod
    def debug(msg: str, **kwargs):
        Logger().logger.debug(msg, **kwargs)

    @staticmethod
    def warning(msg: str, **kwargs):
        Logger().logger.warning(msg, **kwargs)

    @staticmethod
    def error(msg: str, **kwargs):
        Logger().logger.error(msg, **kwargs)

    @staticmethod
    def critical(msg: str, **kwargs):
        Logger().logger.critical(msg, **kwargs)
