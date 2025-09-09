"""
payment_channel.py

Generates a synthetic list of e-commerce payment channels for the Eurozone,
covering both global and local payment methods and providers.

Outputs:
    - payment_channel.csv: One row per payment channel, with ID, method, provider, type, and ISO country code.

Business Rules & Methodology:
    - Each payment channel is uniquely identified with an ID (country-specific where applicable).
    - The 'country_code' field uses ISO-3166-1 alpha-2 codes. "GLOBAL" is used for international methods.
    - Payment methods and providers reflect both major global and Eurozone-specific solutions, based on real-world data.
    - Payment types are classified as Card, Wallet, BankTransfer, or Other.
    - Adding/removing payment methods is modular—update the 'raw_channels' list as needed.
    - The script is fully reproducible and maintainable for downstream analytics or simulation.

Usage:
    Place this script in your working directory and run:
        python payment_channel.py
    Output CSV will be written as "payment_channel.csv".
"""

import pandas as pd
import pycountry
from itertools import count
from typing import List, Dict

def generate_id(country_code: str, counters: Dict[str, count]) -> str:
    """Generate a unique Payment Channel ID per country."""
    n = next(counters[country_code])
    return f"PC_{country_code}_{n:03d}"

def is_valid_iso(iso: str) -> bool:
    """Return True if ISO code is valid or 'GLOBAL'."""
    return iso == "GLOBAL" or pycountry.countries.get(alpha_2=iso) is not None

def classify_payment_type(method: str) -> str:
    """Classify the payment type based on method name."""
    method = method.lower()
    if "card" in method or "debit" in method:
        return "Card"
    elif "wallet" in method or "pay" in method:
        return "Wallet"
    elif "transfer" in method or "bank" in method or "sepa" in method or "iban" in method:
        return "BankTransfer"
    else:
        return "Other"

def create_payment_channels() -> List[Dict]:
    """
    Generate payment channel records, both global and country-specific.

    Returns:
        List[Dict]: List of payment channel dicts.
    """
    raw_channels = [
        {"country_code": "GLOBAL", "payment_method": "Credit Card", "company": "Visa"},
        {"country_code": "GLOBAL", "payment_method": "Credit Card", "company": "MasterCard"},
        {"country_code": "GLOBAL", "payment_method": "Credit Card", "company": "American Express"},
        {"country_code": "GLOBAL", "payment_method": "Credit Card", "company": "Discover"},
        {"country_code": "GLOBAL", "payment_method": "PayPal", "company": "PayPal"},
        {"country_code": "GLOBAL", "payment_method": "Apple Pay", "company": "Apple"},
        {"country_code": "GLOBAL", "payment_method": "Google Pay", "company": "Google"},
        {"country_code": "GLOBAL", "payment_method": "Bank Transfer", "company": "SWIFT"},
        # Spain (ES)
        {"country_code": "ES", "payment_method": "Bizum", "company": "Bizum"},
        {"country_code": "ES", "payment_method": "Credit Card", "company": "Redsys"},
        {"country_code": "ES", "payment_method": "Bank Transfer", "company": "BBVA"},
        {"country_code": "ES", "payment_method": "Bank Transfer", "company": "Santander"},
        # Italy (IT)
        {"country_code": "IT", "payment_method": "MyBank", "company": "MyBank"},
        {"country_code": "IT", "payment_method": "PostePay", "company": "Poste Italiane"},
        {"country_code": "IT", "payment_method": "Bank Transfer", "company": "UniCredit"},
        # France (FR)
        {"country_code": "FR", "payment_method": "Carte Bancaire", "company": "Cartes Bancaires"},
        {"country_code": "FR", "payment_method": "SEPA Direct Debit", "company": "SEPA Core"},
        {"country_code": "FR", "payment_method": "Paylib", "company": "Paylib"},
        # Germany (DE)
        {"country_code": "DE", "payment_method": "Giropay", "company": "Giropay"},
        {"country_code": "DE", "payment_method": "SOFORT", "company": "Klarna"},
        {"country_code": "DE", "payment_method": "Direct Debit", "company": "SEPA"},
        {"country_code": "DE", "payment_method": "Paydirekt", "company": "Paydirekt"},
        # Belgium (BE)
        {"country_code": "BE", "payment_method": "Bancontact", "company": "Bancontact"},
        {"country_code": "BE", "payment_method": "Payconiq", "company": "Payconiq"},
        # Netherlands (NL)
        {"country_code": "NL", "payment_method": "iDEAL", "company": "iDEAL"},
        {"country_code": "NL", "payment_method": "Bank Transfer", "company": "ING"},
        # Austria (AT)
        {"country_code": "AT", "payment_method": "EPS", "company": "EPS"},
        {"country_code": "AT", "payment_method": "Online Banking", "company": "Local Banks"},
        # Croatia (HR)
        {"country_code": "HR", "payment_method": "Debit Card", "company": "Visa Debit"},
        {"country_code": "HR", "payment_method": "Debit Card", "company": "Maestro"},
        # Cyprus (CY)
        {"country_code": "CY", "payment_method": "JCC Payment Gateway", "company": "JCC"},
        {"country_code": "CY", "payment_method": "Bank Transfer", "company": "Local Banks"},
        # Estonia (EE)
        {"country_code": "EE", "payment_method": "Bank Link", "company": "Swedbank"},
        {"country_code": "EE", "payment_method": "Bank Link", "company": "SEB Pank"},
        # Finland (FI)
        {"country_code": "FI", "payment_method": "MobilePay", "company": "MobilePay"},
        {"country_code": "FI", "payment_method": "Online Banking", "company": "Osuuspankki"},
        # Greece (GR)
        {"country_code": "GR", "payment_method": "Viva Wallet", "company": "Viva Wallet"},
        {"country_code": "GR", "payment_method": "Bank Transfer", "company": "Piraeus Bank"},
        # Ireland (IE)
        {"country_code": "IE", "payment_method": "Debit Card", "company": "Visa Debit"},
        {"country_code": "IE", "payment_method": "Bank Transfer", "company": "AIB"},
        # Latvia (LV)
        {"country_code": "LV", "payment_method": "Bank Link", "company": "Swedbank"},
        {"country_code": "LV", "payment_method": "Bank Link", "company": "SEB banka"},
        # Lithuania (LT)
        {"country_code": "LT", "payment_method": "Paysera", "company": "Paysera"},
        {"country_code": "LT", "payment_method": "Bank Link", "company": "Swedbank"},
        # Luxembourg (LU)
        {"country_code": "LU", "payment_method": "Payconiq", "company": "Payconiq"},
        {"country_code": "LU", "payment_method": "Bank Transfer", "company": "BCEE"},
        # Malta (MT)
        {"country_code": "MT", "payment_method": "Bank Transfer", "company": "Bank of Valletta"},
        {"country_code": "MT", "payment_method": "Credit Card", "company": "Local Banks"},
        # Portugal (PT)
        {"country_code": "PT", "payment_method": "Multibanco", "company": "Multibanco"},
        {"country_code": "PT", "payment_method": "MB Way", "company": "MB Way"},
        # Slovakia (SK)
        {"country_code": "SK", "payment_method": "Bank Transfer", "company": "Tatra banka"},
        {"country_code": "SK", "payment_method": "Online Banking", "company": "Slovenská sporitelna"},
        # Slovenia (SI)
        {"country_code": "SI", "payment_method": "Bankart", "company": "Bankart"},
        {"country_code": "SI", "payment_method": "Bank Transfer", "company": "NLB"}
    ]

    unique_isos = set(row["country_code"] for row in raw_channels)
    counters = {iso: count(1) for iso in unique_isos}
    payment_channels = []

    for row in raw_channels:
        country_code = row["country_code"]
        name = row["payment_method"]
        provider = row["company"]
        payment_type = classify_payment_type(name)

        if not is_valid_iso(country_code):
            raise ValueError(f"Invalid ISO code: {country_code}")

        pc_id = generate_id(country_code, counters)
        payment_channels.append({
            "payment_channel_id": pc_id,
            "channel_name": name,
            "provider": provider,
            "country_code": country_code,
            "payment_type": payment_type
        })

    return payment_channels

def main():
    channels = create_payment_channels()
    fieldnames = ["payment_channel_id", "channel_name", "provider", "country_code", "payment_type"]
    df = pd.DataFrame(channels, columns=fieldnames)
    df.to_csv("payment_channel.csv", index=False)
    print(f"payment_channel.csv created ({len(channels)} records).")

if __name__ == "__main__":
    main()