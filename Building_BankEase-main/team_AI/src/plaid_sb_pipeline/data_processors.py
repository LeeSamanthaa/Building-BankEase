# -*- coding: utf-8 -*-
"""
data_processors.py
Created on June 16, 2025
@author: Mazhar

Functions for processing and saving Plaid data.
"""
import json
import logging
import os
from datetime import date
from typing import Any, Dict, List

import pandas as pd

# --- Logger ---
logger = logging.getLogger(__name__)


# --- Custom JSON Encoder for Date objects ---
class DateEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, date):
            return o.isoformat()
        return super().default(o)


# --- Raw Data Saver ---
def save_raw_json(
    data: Dict, output_dir: str, institution_id: str, item_id: str, product: str
):
    """Saves the raw JSON response from Plaid to a file."""
    try:
        # Define the path for the raw data
        raw_dir = os.path.join(output_dir, "raw_json", product)
        os.makedirs(raw_dir, exist_ok=True)

        # Sanitize IDs for use in filenames
        safe_inst_id = "".join(
            c for c in institution_id if c.isalnum() or c in ("_", "-")
        ).rstrip()
        safe_item_id = "".join(
            c for c in item_id if c.isalnum() or c in ("_", "-")
        ).rstrip()

        filename = f"{safe_inst_id}_{safe_item_id}.json"
        filepath = os.path.join(raw_dir, filename)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=4, cls=DateEncoder)
        logger.debug(f"Saved raw {product} JSON to {filepath}")

    except Exception as e:
        logger.error(
            f"Failed to save raw JSON for {product} for item {item_id}. Error: {e}"
        )


# --- Data Flattening and CSV Saving ---
def flatten_and_save_csv(
    all_data: List[Dict[str, Any]],
    output_dir: str,
    product: str,
    item_info: Dict[str, str],
):
    """
    Flattens the product data and saves it to a consolidated CSV file.
    Appends to the file if it already exists.
    """
    try:
        df = pd.DataFrame()
        # Ensure all_data is a list
        if not isinstance(all_data, list):
            all_data = [all_data]

        if product == "transactions":
            records = []
            for data in all_data:
                for t in data.get("transactions", []):
                    record = t.copy()
                    record["item_id"] = item_info.get("item_id")
                    record["institution_id"] = item_info.get("institution_id")
                    records.append(record)
            df = pd.json_normalize(records)
        elif product == "auth":
            records = []
            for data in all_data:
                for a in data.get("accounts", []):
                    record = {
                        "account_id": a["account_id"],
                        "mask": a["mask"],
                        "name": a["name"],
                        "subtype": a["subtype"],
                        "type": a["type"],
                        "ach_account": a.get("numbers", {})
                        .get("ach", [{}])[0]
                        .get("account"),
                        "ach_routing": a.get("numbers", {})
                        .get("ach", [{}])[0]
                        .get("routing"),
                    }
                    record["item_id"] = item_info.get("item_id")
                    record["institution_id"] = item_info.get("institution_id")
                    records.append(record)
            df = pd.DataFrame(records)
        elif product == "identity":
            records = []
            for data in all_data:
                for acc in data.get("accounts", []):
                    for owner in acc.get("owners", []):
                        record = {
                            "account_id": acc.get("account_id"),
                            "mask": acc.get("mask"),
                            "name": acc.get("name"),
                            "subtype": acc.get("subtype"),
                            "type": acc.get("type"),
                            "owner_name": ", ".join(owner.get("names", [])),
                            "owner_phone": ", ".join(
                                p.get("data") for p in owner.get("phone_numbers", [])
                            ),
                            "owner_email": ", ".join(
                                e.get("data") for e in owner.get("emails", [])
                            ),
                            "owner_address": ", ".join(
                                a.get("data", {}).get("street", "")
                                for a in owner.get("addresses", [])
                            ),
                        }
                        record["item_id"] = item_info.get("item_id")
                        record["institution_id"] = item_info.get("institution_id")
                        records.append(record)
            df = pd.DataFrame(records)
        elif product == "investments":
            records = []
            for data in all_data:
                holdings = data.get("holdings", [])
                securities = {s["security_id"]: s for s in data.get("securities", [])}
                for h in holdings:
                    security = securities.get(h["security_id"], {})
                    record = {
                        "account_id": h.get("account_id"),
                        "cost_basis": h.get("cost_basis"),
                        "quantity": h.get("quantity"),
                        "value": h.get("institution_value"),
                        "security_name": security.get("name"),
                        "security_ticker": security.get("ticker_symbol"),
                        "security_type": security.get("type"),
                    }
                    record["item_id"] = item_info.get("item_id")
                    record["institution_id"] = item_info.get("institution_id")
                    records.append(record)
            df = pd.DataFrame(records)
        elif product == "liabilities":
            records = []
            for data in all_data:
                for l in (
                    data.get("liabilities", {}).get("credit", [])
                    + data.get("liabilities", {}).get("mortgage", [])
                    + data.get("liabilities", {}).get("student", [])
                ):
                    record = l.copy()
                    record["item_id"] = item_info.get("item_id")
                    record["institution_id"] = item_info.get("institution_id")
                    records.append(record)
            df = pd.json_normalize(records)

        if not df.empty:
            # Save to CSV
            csv_path = os.path.join(output_dir, f"{product}.csv")
            df.to_csv(
                csv_path, mode="a", header=not os.path.exists(csv_path), index=False
            )
            logger.debug(f"Saved/Appended {len(df)} records to {csv_path}")

    except Exception as e:
        logger.error(f"Failed to flatten/save CSV for {product}. Error: {e}")
