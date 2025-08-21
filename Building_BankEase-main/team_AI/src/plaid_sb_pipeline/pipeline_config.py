# -*- coding: utf-8 -*-
"""
pipeline_config.py
Created on June 16, 2025
@author: Mazhar

Configuration loader for the Plaid pipeline.
"""
import logging
import os

import yaml

# --- Logger ---
logger = logging.getLogger(__name__)

# --- Constants ---
CONFIG_FILE_PATH = os.path.join(
    os.getcwd(), "configs", "plaid_sb_pipeline_configs.yaml"
)


def load_pipeline_config():
    """
    Loads the main pipeline configuration from the YAML file.
    """
    logger.info(f"Attempting to load configuration from: {CONFIG_FILE_PATH}")
    if not os.path.exists(CONFIG_FILE_PATH):
        logger.critical(f"Configuration file not found at {CONFIG_FILE_PATH}. Halting.")
        return None
    try:
        with open(CONFIG_FILE_PATH, "r") as f:
            config = yaml.safe_load(f)
        logger.info("✅ Configuration file loaded successfully.")
        return config
    except yaml.YAMLError as e:
        logger.critical(f"Error parsing YAML configuration file: {e}")
        return None
    except Exception as e:
        logger.critical(
            f"An unexpected error occurred while loading the configuration: {e}"
        )
        return None
