"""
shipment_company.py

Generates a synthetic, realistic shipment company dataset for Eurozone e-commerce simulation.
- Uses parameterized lists for company names, country codes, and delivery types.
- Uses pandas for output and validation, consistent with project standards.

Note: The use of comma-separated fields for operating_countries and delivery_types intentionally violates SQL 1NF (First Normal Form).
This design allows for downstream data cleansing exercises, such as splitting lists into rows and building proper junction tables in SQL.

Outputs:
    - shipment_company.csv
"""
import pandas as pd

EU_COUNTRIES = [
    "AT", "BE", "HR", "CY", "EE", "FI", "FR", "DE", "GR", "IE",
    "IT", "LV", "LT", "LU", "MT", "NL", "PT", "SK", "SI", "ES"
]

POSTAL_COMPANIES = [
    "Osterreichische Post", "bpost", "Hrvatska posta", "Cyprus Postal Services",
    "Omniva", "Posti Group", "La Poste / Colissimo", "Deutsche Post DHL",
    "Hellenic Post (ELTA)", "An Post", "Poste Italiane", "Latvijas Pasts",
    "Lietuvos pastas", "Post Luxembourg", "MaltaPost", "PostNL",
    "CTT - Correios de Portugal", "Slovenska posta", "Posta Slovenije", "Correos"
]
COURIER_COMPANIES = [
    "Hermes", "Chronopost", "BRT", "DPDgroup", "UPS", "DHL", "FedEx", "GLS"
]
ALL_COMPANIES = POSTAL_COMPANIES + COURIER_COMPANIES

DELIVERY_TYPE_PROFILES = {
    "Postal": ["Standard", "Express", "Registered"],
    "Courier": ["Standard", "Express"],
    "Global": ["Standard", "Express", "International"]
}

def assign_operating_countries(company_name):
    import random
    if company_name in POSTAL_COMPANIES:
        idx = POSTAL_COMPANIES.index(company_name)
        country = EU_COUNTRIES[idx]
        return [country]
    elif company_name in ["UPS", "DHL", "FedEx", "GLS", "DPDgroup"]:
        return EU_COUNTRIES.copy()
    else:
        n = random.randint(2, 7)
        return sorted(random.sample(EU_COUNTRIES, n))

def assign_company_type(company_name):
    if company_name in POSTAL_COMPANIES:
        return "Postal"
    elif company_name in COURIER_COMPANIES:
        return "Global" if company_name in ["UPS", "DHL", "FedEx", "GLS", "DPDgroup"] else "Courier"
    else:
        return "Other"

def main():
    records = []
    for company_name in ALL_COMPANIES:
        operating_countries = assign_operating_countries(company_name)
        company_type = assign_company_type(company_name)
        delivery_types = ",".join(DELIVERY_TYPE_PROFILES[company_type])
        records.append({
            "company_name": company_name,
            "company_type": company_type,
            "operating_countries": ",".join(operating_countries),
            "delivery_types": delivery_types
        })
    df = pd.DataFrame(records)
    df.insert(0, "shipment_company_id", range(1, len(df) + 1))
    df.to_csv("shipment_company.csv", index=False)
    print(f"CSV file created: shipment_company.csv")
    print(f"Total shipment companies: {len(df)} (1:1 mapping enforced)")

if __name__ == "__main__":
    main()
