import os
import sys

from dotenv import load_dotenv
from loguru import logger


load_dotenv()
logger.remove()


class LogFactory:
    @staticmethod
    def get_logger(name: str):
        """
        Provides a logger with the given name and a preconfigured formatter.

        The log level can be configured per module (=name of the logger). For example, to configure the log level for
        the llm.openai.openai_llm module the environment variable LOG_LEVEL_MYGPT_LLM_OPENAI_OPENAI_LLM=DEBUG overrides
        the root log level.

        See https://loguru.readthedocs.io/en/0.7.2/api/logger.html#loguru._logger.Logger.add
        """
        root_log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
        logger_var_name = name.replace(".", "_").upper()
        log_level = os.environ.get(
            f"LOG_LEVEL_{logger_var_name}", root_log_level
        ).upper()
        root_log_format = os.environ.get(
            "LOG_FORMAT",
            "{time} <light-blue>[{thread.name} | {module}]</light-blue> <level>{level}</level> {message}",
        )
        log_format = os.environ.get(f"LOG_FORMAT_{logger_var_name}", root_log_format)
        root_log_colorize = os.environ.get("LOG_COLORIZE", "true").lower()
        log_colorize = os.environ.get(
            f"LOG_COLORIZE_{logger_var_name}", root_log_colorize
        ).lower()
        logger.add(
            sys.stdout,
            colorize=log_colorize == "true",
            format=log_format,
            level=log_level,
            backtrace=True,
            diagnose=True if log_level in ["DEBUG", "TRACE"] else False,
            filter=name,
        )
        return logger
