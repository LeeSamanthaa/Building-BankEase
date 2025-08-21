# -*- coding: utf-8 -*-
# """
# utils/list_of_dicts_to_csv.py
# Created on June 03, 2025
# @ Author: Mazhar
# """

import logging
import os
from typing import Any, Dict, List, Optional
import json  # Ensure List, Dict are imported

import pandas as pd


def save_list_of_dicts_to_json(
    data_list: List[Dict[str, Any]],
    json_filepath: Optional[str],  # Allow None to skip saving
    logger: logging.Logger,
) -> bool:
    """
    Saves a list of dictionaries to a single JSON file.
    The JSON file will contain a JSON array as its top-level structure.

    Args:
        data_list (List[Dict[str, Any]]): List of dictionaries to save.
        json_filepath (Optional[str]): Full path to the output JSON file.
                                     If None, the function will log and return False.
        logger (logging.Logger): Logger instance.

    Returns:
        bool: True if saving was successful, False otherwise.
    """
    if not json_filepath:
        logger.warning("JSON file path not provided. Data will not be saved to JSON.")
        return False  # Or True if "not saving" is considered a success by the caller

    if not data_list:  # Handle empty list case
        logger.info(
            f"Data list is empty. Writing an empty list to JSON: {json_filepath}"
        )
        # Fallthrough to write an empty list "[]" to the file

    logger.info(
        f"Attempting to save a list of {len(data_list)} dictionaries to JSON: {json_filepath}"
    )
    try:
        with open(json_filepath, "w", encoding="utf-8") as f_json:
            json.dump(data_list, f_json, ensure_ascii=False, indent=2)
        logger.info(f"Successfully saved list of dictionaries to JSON: {json_filepath}")
        return True
    except IOError as e:
        logger.error(
            f"IOError writing list of dictionaries to JSON file {json_filepath}: {e}",
            exc_info=True,
        )
    except TypeError as e:  # If data_list contains non-JSON serializable items
        logger.error(
            f"TypeError - data in list not JSON serializable for {json_filepath}: {e}",
            exc_info=True,
        )
    except Exception as e:  # Catch any other unexpected errors
        logger.error(
            f"An unexpected error occurred while saving list to JSON {json_filepath}: {e}",
            exc_info=True,
        )
    return False


def save_list_of_dicts_to_csv(
    data_list: List[Dict[str, Any]],
    csv_filepath: str,
    logger: logging.Logger,
) -> bool:
    """
    Converts a list of flat dictionaries to a Pandas DataFrame and saves it as a CSV.
    Overwrites the CSV file if it exists.

    Args:
        data_list (List[Dict[str, Any]]): The list of dictionaries to convert.
                                         Each dictionary represents a row.
        csv_filepath (str): Path to the output CSV file.
        logger (logging.Logger): Logger instance.

    Returns:
        bool: True if saving was successful, False otherwise.
    """
    if not csv_filepath:  # Check if path is provided
        logger.error(msg="CSV filepath not provided for saving.")
        return False

    logger.info(msg=f"Attempting to save data to CSV: {csv_filepath}")

    if not data_list:
        logger.info(msg="The provided data list is empty. Creating an empty CSV file.")
        try:
            df_empty = pd.DataFrame(data=data_list)  # Will create an empty DataFrame
            df_empty.to_csv(
                path_or_buf=csv_filepath,
                index=False,
                mode="w",  # Overwrite
                encoding="utf-8",
            )
            logger.info(msg=f"Empty CSV file '{csv_filepath}' created/overwritten.")
            return True
        except Exception as e:
            logger.error(
                msg=f"Error creating an empty CSV {csv_filepath}: {e}", exc_info=True
            )
            return False

    try:
        df = pd.DataFrame(data=data_list)
        logger.info(
            msg=f"Successfully created DataFrame for CSV with {len(df)} rows and {len(df.columns)} columns."
        )
        df.to_csv(
            path_or_buf=csv_filepath, index=False, mode="w", encoding="utf-8"
        )  # Overwrite
        logger.info(msg=f"Successfully saved DataFrame to CSV: {csv_filepath}")
        return True
    except Exception as e:
        logger.error(
            msg=f"An error occurred during DataFrame creation or CSV writing: {e}",
            exc_info=True,
        )
        return False
