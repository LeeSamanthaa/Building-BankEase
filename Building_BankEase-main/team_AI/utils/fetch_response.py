# -*- coding: utf-8 -*-
# """
# fetch_response.py
# Created on June 02, 2025
# @ Author: Mazhar
# """

import logging
from typing import Any, Optional

import requests


# --- Fetch Plaid Data ---
def fetch_response(
    api_url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    logger: logging.Logger,
) -> Optional[requests.Response]:
    """
    Makes a POST request to the specified API URL and returns the Response object.
    Handles HTTP errors and connection issues.

    Args:
        api_url (str): The API endpoint URL.
        headers (dict[str, str]): HTTP headers for the request.
        payload (dict[str, Any]): The JSON payload for the request.
        logger (logging.Logger): Logger instance for logging messages.

    Returns:
        Optional[requests.Response]: The requests.Response object if the request
                                     was successful (status 2xx), otherwise None.
    """
    try:
        # logger.info(msg=f"Sending POST request to: {api_url}")
        # Consider redacting sensitive parts of the payload if logging it
        # logger.debug(f"Payload: {payload}")

        response: requests.Response = requests.post(
            url=api_url, headers=headers, json=payload
        )
        response.raise_for_status()  # Raise an HTTPError for bad responses (4XX or 5XX)

        # logger.info(msg=f"Request successful. Status Code: {response.status_code}")
        return response

    except requests.exceptions.HTTPError as http_err:
        logger.error(msg=f"HTTP error occurred: {http_err}")
        # The 'response' object is available in http_err.response
        if http_err.response is not None:
            logger.error(
                msg=f"Response content for HTTPError: {http_err.response.text}"
            )
        return None
    except requests.exceptions.ConnectionError as conn_err:
        logger.error(msg=f"Connection error occurred: {conn_err}")
        return None
    except requests.exceptions.Timeout as timeout_err:
        logger.error(msg=f"Timeout error occurred: {timeout_err}")
        return None
    except (
        requests.exceptions.RequestException
    ) as req_err:  # Catch other requests-related errors
        logger.error(msg=f"A requests library error occurred: {req_err}")
        return None
    except Exception as e:  # Catch any other unexpected errors
        logger.error(
            msg=f"An unexpected general error occurred during fetch_plaid_data: {e}",
            exc_info=True,
        )
        return None
