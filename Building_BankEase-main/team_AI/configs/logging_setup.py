# -*- coding: utf-8 -*-
# """
# logging_setup.py
# Created on June 02, 2025
# @ Author: Mazhar
# """

import logging
import os
import sys
from typing import Any

import yaml
from colorama import Fore, Style, init

# Initialize colorama for colored logging
init(autoreset=True)

# Load the YAML file
with open(file="./configs/log_config.yaml", mode="r", encoding="utf-8") as file:
    config: dict[str, Any] = yaml.safe_load(stream=file)

# Access the variables from config.yaml file
LOG_FILE = config["LOGGING"]["LOG_FILE"]
LOG_LEVEL = config["LOGGING"]["LOG_LEVEL"]
LOG_FORMAT = config["LOGGING"]["LOG_FORMAT"]

# Convert LOG_LEVEL to logging module constants
LOG_LEVEL = getattr(logging, LOG_LEVEL.upper(), logging.INFO)

# Define Log Size Limit (1KB = 1024 Bytes)
LOG_SIZE_LIMIT = config["LOGGING"][
    "LOG_SIZE_LIMIT"
]  # Set log file size limit (Default is 1MB)


# Function to check log file size and clear it if needed
def check_log_file_size() -> None:
    """Clears the log file if it exceeds the specified limit."""
    if (
        os.path.exists(path=LOG_FILE)
        and os.path.getsize(filename=LOG_FILE) > LOG_SIZE_LIMIT
    ):
        with open(file=LOG_FILE, mode="w", encoding="utf-8") as log_file:
            log_file.write("")  # Clear the file content
        print(f"🧹 Log file '{LOG_FILE}' exceeded {LOG_SIZE_LIMIT} bytes, cleared.")


# Custom formatter to include colors
class ColoredFormatter(logging.Formatter):
    def format(self, record) -> str:
        # Add colors based on the log level
        if record.levelno == logging.INFO:
            record.msg = f"{Fore.GREEN}{record.msg}{Style.RESET_ALL}"
        elif record.levelno == logging.DEBUG:
            record.msg = f"{Fore.CYAN}{record.msg}{Style.RESET_ALL}"
        elif record.levelno == logging.WARNING:
            record.msg = f"{Fore.YELLOW}{record.msg}{Style.RESET_ALL}"
        elif record.levelno == logging.ERROR:
            record.msg = f"{Fore.RED}{record.msg}{Style.RESET_ALL}"
        elif record.levelno == logging.CRITICAL:
            record.msg = f"{Fore.MAGENTA}{record.msg}{Style.RESET_ALL}"
        return super().format(record=record)


# Set up logging with UTF-8 encoding for file handler
formatter = ColoredFormatter(fmt=LOG_FORMAT)

# Console Handler (Fix encoding issue on Windows)
console_handler = logging.StreamHandler(stream=sys.stdout)
console_handler.setFormatter(fmt=formatter)

# Ensure console output uses UTF-8 encoding
# if hasattr(console_handler.stream, "reconfigure"):
#     console_handler.stream.reconfigure(encoding="utf-8")

# File Handler (Ensure UTF-8 encoding)
file_handler = logging.FileHandler(filename=LOG_FILE, encoding="utf-8")
file_handler.setFormatter(fmt=formatter)

# Check and clear the log file if it exceeds the limit
check_log_file_size()

logging.basicConfig(
    level=LOG_LEVEL,
    handlers=[console_handler, file_handler],
)

# Create a logger object
logger: logging.Logger = logging.getLogger()

# Ensure the logger is only configured once
if not logger.hasHandlers():
    logger.addHandler(hdlr=console_handler)
    logger.addHandler(hdlr=file_handler)


def get_logger() -> logging.Logger:
    """Return the configured logger."""
    return logger
