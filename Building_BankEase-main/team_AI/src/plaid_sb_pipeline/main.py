# -*- coding: utf-8 -*-
"""
main.py
Created on June 16, 2025
@author: Mazhar

Main entry point for the Country-Wise Plaid Data Pipeline.
"""
import ast
import logging
import os
import shutil
import sys

import pandas as pd
from dotenv import load_dotenv

# --- Path Setup ---
# Assumes the script is run from the 'team_AI' directory.
PROJECT_ROOT = os.getcwd()
sys.path.insert(0, PROJECT_ROOT)

import src.plaid_sb_pipeline.fetchers as fetchers
from configs.logging_setup import get_logger
from src.plaid_sb_pipeline.data_processors import flatten_and_save_csv, save_raw_json
from src.plaid_sb_pipeline.fetchers import create_item, get_institutions
from src.plaid_sb_pipeline.pipeline_config import load_pipeline_config

# --- Logger ---
logger = get_logger()


def run_pipeline():
    """
    Executes the full data generation and fetching pipeline for each country.
    """
    logger.info("=================================================")
    logger.info("🚀 Starting the Plaid Data Pipeline...")
    logger.info("=================================================")

    # 1. Load Pipeline Configuration
    config = load_pipeline_config()
    if not config:
        return

    # Load environment variables for Plaid credentials
    env_path = os.path.join(PROJECT_ROOT, ".env")
    load_dotenv(dotenv_path=env_path)
    CLIENT_ID = os.getenv("PLAID_CLIENT_ID")
    SECRET = os.getenv("PLAID_CLIENT_SECRET")
    if not CLIENT_ID or not SECRET:
        logger.critical("PLAID_CLIENT_ID or PLAID_SECRET not found. Halting.")
        return

    base_output_dir = os.path.join(PROJECT_ROOT, config["output_data_dir"])
    os.makedirs(base_output_dir, exist_ok=True)
    logger.info(f"Base output directory is: {base_output_dir}")

    # --- Main Loop: Process each country from the config file ---
    for country_code in config["institutions"]["country_codes"]:
        logger.info(f"\n{'='*20} PROCESSING COUNTRY: {country_code} {'='*20}")

        country_output_dir = os.path.join(base_output_dir, country_code)

        # Clean up previous data for THIS country only
        if os.path.exists(country_output_dir):
            logger.info(
                f"Deleting existing directory for {country_code}: {country_output_dir}"
            )
            shutil.rmtree(country_output_dir)
        os.makedirs(country_output_dir)
        logger.info(f"Created fresh output directory: {country_output_dir}")

        # 2. Fetch Institutions for the current country
        institutions_config = config["institutions"]
        items_config = config["items"]

        # Get advanced settings with defaults
        api_settings = config.get("api_settings", {})
        offset = api_settings.get("institutions_offset", 0)
        delay = api_settings.get("request_delay_seconds", 1)

        # Determine which products to use for the current country
        country_specific_products = institutions_config.get("country_products", {})
        products_to_use = country_specific_products.get(
            country_code, items_config["initial_products"]
        )

        institutions = get_institutions(
            client_id=CLIENT_ID,
            secret=SECRET,
            country_codes=[country_code],
            count=institutions_config["count_per_country"],
            products=products_to_use,
            offset=offset,
            delay=delay,
        )
        if not institutions:
            continue  # Skip to next country if none found

        # Save institutions to CSV
        inst_df = pd.DataFrame(institutions)
        inst_csv_path = os.path.join(country_output_dir, "institutions.csv")
        inst_df.to_csv(inst_csv_path, index=False)
        logger.info(f"Saved {len(inst_df)} institutions to {inst_csv_path}")

        # 3. Create Customer Items for these institutions
        all_items_for_processing = []
        all_items_for_csv = []
        for inst in institutions:
            # Determine the products to use for creating an item for this specific institution.
            # Use the intersection of desired initial products and the products supported by the institution.
            desired_initial_products = items_config.get("initial_products", [])
            supported_products = inst.get("products", [])
            products_for_item_creation = [
                p for p in desired_initial_products if p in supported_products
            ]

            # We must have at least one product to create an item.
            # If there's no overlap, we cannot proceed with this institution.
            if not products_for_item_creation:
                logger.warning(
                    f"Cannot create item for institution '{inst['name']}' ({inst['institution_id']}) "
                    f"as it does not support any of the desired initial products. "
                    f"Supported: {supported_products}. Desired: {desired_initial_products}. "
                    "Skipping item creation for this institution."
                )
                continue

            for i in range(items_config["customers_per_institution"]):
                new_item = create_item(
                    client_id=CLIENT_ID,
                    secret=SECRET,
                    institution_id=inst["institution_id"],
                    initial_products=products_for_item_creation,
                )
                if new_item:
                    # For processing, keep access_token
                    item_for_processing = {
                        **new_item,
                        "institution_name": inst["name"],
                        "institution_id": inst["institution_id"],
                    }
                    all_items_for_processing.append(item_for_processing)

                    # For CSV, exclude access_token
                    item_for_csv = {
                        "item_id": new_item["item_id"],
                        "institution_id": inst["institution_id"],
                        "institution_name": inst["name"],
                    }
                    all_items_for_csv.append(item_for_csv)

        if not all_items_for_processing:
            continue

        # Save items to CSV
        items_df = pd.DataFrame(all_items_for_csv)
        items_csv_path = os.path.join(country_output_dir, "items.csv")
        items_df.to_csv(items_csv_path, index=False)
        logger.info(f"Saved {len(items_df)} items to {items_csv_path}")

        # 4. Fetch and Save Product Data for each item
        fetcher_map = {
            "auth": fetchers.fetch_auth,
            "balance": fetchers.fetch_balance,
            "identity": fetchers.fetch_identity,
            "investments": fetchers.fetch_investments,
            "liabilities": fetchers.fetch_liabilities,
            "transactions": fetchers.fetch_transactions,
        }
        for item in all_items_for_processing:
            logger.info(f"--- Fetching data for Item: {item['item_id']} ---")
            for product in config["products_to_fetch"]:
                fetcher_func = fetcher_map.get(product)
                if not fetcher_func:
                    continue

                product_data = fetcher_func(item["access_token"], CLIENT_ID, SECRET)
                if product_data:
                    save_raw_json(
                        product_data,
                        country_output_dir,
                        item["institution_id"],
                        item["item_id"],
                        product,
                    )
                    flatten_and_save_csv(
                        product_data, country_output_dir, product, item
                    )

    logger.info(f"\n{'='*20} ✅ PIPELINE FINISHED {'='*20}")


if __name__ == "__main__":
    run_pipeline()
