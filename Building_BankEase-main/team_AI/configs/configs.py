# -*- coding: utf-8 -*-
# """
# configs.py
# Created on June 02, 2025
# @ Author: Mazhar
# """

import logging
from typing import Any

import yaml

# Import the logger setup
from configs.logging_setup import get_logger

# Get the logger
logger: logging.Logger = get_logger()

PLAID_SBX_CONFIGS = "./configs/plaid_sbx_configs.yaml"

cfgs_plaid_sbx: dict[str, Any] = {}
try:
    with open(file=PLAID_SBX_CONFIGS, mode="r") as file:
        cfgs_plaid_sbx = yaml.safe_load(stream=file)
except FileNotFoundError:
    logger.error(msg=f"⚠️ {PLAID_SBX_CONFIGS} not found.")
    cfgs_plaid_sbx = {}
finally:
    if not cfgs_plaid_sbx:
        logger.error(msg="No configuration found in the YAML file.")
        exit(code=1)
