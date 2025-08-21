# -*- coding: utf-8 -*-
"""
src/plaid_sb_pipeline/fetchers.py
Created on June 15, 2025
@author: Mazhar

Contains functions to fetch data from the Plaid API.
"""
import os
import sys
import time
from datetime import date
from typing import Any, Dict, List, Optional

import plaid
from plaid.api import plaid_api
from plaid.model.country_code import CountryCode
from plaid.model.institutions_get_request import InstitutionsGetRequest
from plaid.model.institutions_get_request_options import InstitutionsGetRequestOptions
from plaid.model.item_public_token_exchange_request import (
    ItemPublicTokenExchangeRequest,
)
from plaid.model.products import Products
from plaid.model.sandbox_public_token_create_request import (
    SandboxPublicTokenCreateRequest,
)
from plaid.model.sandbox_public_token_create_request_options import (
    SandboxPublicTokenCreateRequestOptions,
)

# --- Path Setup ---
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
sys.path.insert(0, PROJECT_ROOT)

from configs.logging_setup import get_logger
from utils.fetch_response import fetch_response

logger = get_logger()

# This needs to be adapted from your existing config structure.
# For now, I will hardcode the URL.
# A better approach would be to have this in the config.
BASE_URL = "https://sandbox.plaid.com"


def get_plaid_client(client_id: str, secret: str) -> Optional[plaid_api.PlaidApi]:
    """Initializes and returns a Plaid API client."""
    host = plaid.Environment.Sandbox
    configuration = plaid.Configuration(
        host=host, api_key={"clientId": client_id, "secret": secret}
    )
    api_client = plaid.ApiClient(configuration)
    return plaid_api.PlaidApi(api_client)


def get_institutions(
    client_id: str,
    secret: str,
    country_codes: List[str],
    count: int,
    products: Optional[List[str]],
    offset: int,
    delay: int,
) -> List[Dict]:
    """
    Fetches a list of institutions from Plaid, filtered by country and products.
    """
    client = get_plaid_client(client_id, secret)
    if not client:
        return []

    try:
        plaid_country_codes = [CountryCode(c.upper()) for c in country_codes]

        request_params = {
            "client_id": client_id,
            "secret": secret,
            "country_codes": plaid_country_codes,
            "count": count,
            "offset": offset,
        }

        if products:
            plaid_products = [Products(p) for p in products]
            options = InstitutionsGetRequestOptions(products=plaid_products)
            request_params["options"] = options

        request = InstitutionsGetRequest(**request_params)
        response = client.institutions_get(request)

        logger.info(
            f"Successfully fetched {len(response['institutions'])} institutions for countries: {country_codes}"
        )

        # Add a delay to respect rate limits
        time.sleep(delay)

        return [
            {
                "institution_id": inst.institution_id,
                "name": inst.name,
                "products": [p.value for p in inst.products],
            }
            for inst in response["institutions"]
        ]
    except plaid.ApiException as e:
        logger.error(f"Error fetching institutions: {e.body}")
        return []


def create_item(
    client_id: str,
    secret: str,
    institution_id: str,
    initial_products: List[str],
) -> Optional[Dict[str, str]]:
    """
    Creates a new Item (a customer's connection) for a given institution.
    """
    client = get_plaid_client(client_id, secret)
    if not client:
        return None

    try:
        plaid_products = [Products(p) for p in initial_products]

        # Create a sandbox public token
        pt_request = SandboxPublicTokenCreateRequest(
            client_id=client_id,
            secret=secret,
            institution_id=institution_id,
            initial_products=plaid_products,
            options=SandboxPublicTokenCreateRequestOptions(
                webhook="https://www.generic-webhook-receiver.com/webhook"
            ),
        )
        pt_response = client.sandbox_public_token_create(pt_request)
        public_token = pt_response["public_token"]

        # Exchange public token for an access token
        exchange_request = ItemPublicTokenExchangeRequest(public_token=public_token)
        exchange_response = client.item_public_token_exchange(exchange_request)

        logger.info(
            f"Successfully created Item with ID: {exchange_response['item_id']}"
        )

        return {
            "access_token": exchange_response["access_token"],
            "item_id": exchange_response["item_id"],
        }
    except plaid.ApiException as e:
        logger.error(f"Error creating item for institution {institution_id}: {e.body}")
        return None


# ---------------------------------------------------------------------------- #
#                               Product Fetchers                               #
# ---------------------------------------------------------------------------- #


def _fetch_data(access_token, client_id, secret, fetch_function_name):
    """Generic data fetching helper."""
    client = get_plaid_client(client_id, secret)
    if not client:
        return None
    try:
        fetch_function = getattr(client, fetch_function_name)
        request = {"access_token": access_token}
        response = fetch_function(request)
        return response.to_dict()
    except plaid.ApiException as e:
        logger.error(f"Error in {fetch_function_name}: {e.body}")
        return None


def fetch_auth(access_token, client_id, secret):
    return _fetch_data(access_token, client_id, secret, "auth_get")


def fetch_identity(access_token, client_id, secret):
    return _fetch_data(access_token, client_id, secret, "identity_get")


def fetch_balance(access_token, client_id, secret):
    return _fetch_data(access_token, client_id, secret, "accounts_balance_get")


def fetch_liabilities(access_token, client_id, secret):
    from plaid.model.liabilities_get_request import LiabilitiesGetRequest

    client = get_plaid_client(client_id, secret)
    if not client:
        return None
    try:
        request = LiabilitiesGetRequest(access_token=access_token)
        response = client.liabilities_get(request)
        return response.to_dict()
    except plaid.ApiException as e:
        logger.error(f"Error fetching liabilities: {e.body}")
        return None


def fetch_investments(access_token, client_id, secret):
    from plaid.model.investments_holdings_get_request import (
        InvestmentsHoldingsGetRequest,
    )

    client = get_plaid_client(client_id, secret)
    if not client:
        return None
    try:
        request = InvestmentsHoldingsGetRequest(access_token=access_token)
        response = client.investments_holdings_get(request)
        return response.to_dict()
    except plaid.ApiException as e:
        logger.error(f"Error fetching investments: {e.body}")
        return None


def fetch_transactions(access_token, client_id, secret):
    from plaid.model.transactions_get_request import TransactionsGetRequest
    from plaid.model.transactions_get_request_options import (
        TransactionsGetRequestOptions,
    )

    client = get_plaid_client(client_id, secret)
    if not client:
        return None
    try:
        request = TransactionsGetRequest(
            access_token=access_token,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            options=TransactionsGetRequestOptions(count=100),
        )
        response = client.transactions_get(request)
        return response.to_dict()
    except plaid.ApiException as e:
        logger.error(f"Error fetching transactions: {e.body}")
        return None
