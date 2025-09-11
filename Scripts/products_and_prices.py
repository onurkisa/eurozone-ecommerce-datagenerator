"""
products_and_prices.py

Generates a synthetic product catalog (`products.csv`) and country-adjusted product pricing (`product_prices.csv`)
for a Eurozone e-commerce simulation project.

Outputs:
    - products.csv: One row per product (Eurozone-wide base price, attributes).
    - product_prices.csv: One row per product per country, with local price adjusted for economic indicators.


Business Rules & Methodology:

    1. Product Catalog Generation (products.csv)
        - Each product has a unique product_id.
        - Product names, brands, and all text fields are ASCII-normalized for compatibility.
        - Categories, subcategories, and brands reflect a Eurozone e-commerce assortment.
        - The unit_price for each product is calculated as cost plus a category-specific margin, plus minor Gaussian noise,
          simulating realistic within-country e-commerce retail pricing.
        - This "unit_price" represents a Eurozone base price and is used as the starting point for country-specific adjustments.

    2. Country-Adjusted Product Prices (product_prices.csv)
        - For each product and country, a local price ("local_price") is computed by adjusting the base price
          using the country's economic indicators from eco_sitch.csv (see below).
        - The blended adjustment formula is:
            blended = 1 + (econ['pps_norm'] - 0.5) * 2
            local_multiplier = clamp(blended, 0.85, 1.2)
            local_price = unit_price * local_multiplier
        - This multiplier ensures prices reflect local purchasing power while preventing extreme deviations
          that could lead to unrealistic price gaps or arbitrage opportunities.
        - This approach is grounded in real-world cross-country pricing: it balances local consumer affordability
          and market pricing structure (PPS_norm), while clamping prevents unrealistic price gaps or arbitrage risk.
        - All prices are rounded to 2 decimals and effective_date is set to "#year-01-01" for this simulation.

    3. Inputs:
        - eco_sitch.csv: From Phase 1. Used to compute country-specific multipliers for price adjustment.

    4. Usage:
        Place this script in your working directory alongside eco_sitch.csv.
        Run:
            python products_and_prices.py
        Output CSVs will be written to the current directory.
"""

import random
from faker import Faker
import math
import unidecode
import pandas as pd

ISO_MAP = {
    'AT': 'Austria', 'BE': 'Belgium', 'HR': 'Croatia', 'CY': 'Cyprus',
    'EE': 'Estonia', 'FI': 'Finland', 'FR': 'France', 'DE': 'Germany',
    'GR': 'Greece', 'IE': 'Ireland', 'IT': 'Italy', 'LV': 'Latvia',
    'LT': 'Lithuania', 'LU': 'Luxembourg', 'MT': 'Malta', 'NL': 'Netherlands',
    'PT': 'Portugal', 'SK': 'Slovakia', 'SI': 'Slovenia', 'ES': 'Spain'
}
REVERSE_ISO_MAP = {v: k for k, v in ISO_MAP.items()}

locale_map = {
    "DE": "de_DE", "FR": "fr_FR", "ES": "es_ES", "IT": "it_IT", "NL": "nl_NL",
    "BE": "fr_BE", "FI": "fi_FI", "AT": "de_AT", "PT": "pt_PT", "IE": "en_IE",
    "SI": "sl_SI", "SK": "sk_SK", "LT": "lt_LT", "LV": "lv_LV", "EE": "et_EE",
    "GR": "el_GR", "LU": "fr_FR", "MT": "en_GB", "CY": "en_GB"
}

def get_safe_locale(country_code):
    """Return a valid Faker locale string for the given Eurozone country code."""
    from faker.config import AVAILABLE_LOCALES
    suggested_locale = locale_map.get(country_code, 'en_GB')
    return suggested_locale if suggested_locale in AVAILABLE_LOCALES else 'en_GB'

def to_ascii(text):
    """Convert any unicode string to ASCII using unidecode."""
    if text is None:
        return None
    return unidecode.unidecode(str(text))

SEED = 42
random.seed(SEED)
Faker.seed(SEED)
valid_locales = list(set(locale_map.values()))
faker = Faker(valid_locales)
faker.seed_instance(SEED)

CATEGORIES = {
    "Electronics": ["Smartphones", "Laptops", "Tablets", "Headphones", "Wearables", "Cameras", "TVs", "Gaming Consoles", "Drones", "Smart Home Devices"],
    "Fashion": ["Men's Clothing", "Women's Clothing", "Shoes", "Accessories", "Bags", "Watches", "Jewelry", "Eyewear", "Sportswear", "Underwear"],
    "Home & Garden": ["Furniture", "Kitchen", "Home Decor", "Bedding", "Lighting", "Garden Tools", "Storage", "Bathroom", "Cleaning Supplies", "Cookware"],
    "Health & Beauty": ["Skincare", "Makeup", "Hair Care", "Fragrances", "Personal Care", "Oral Care", "Health Supplements", "Medical Devices", "Fitness", "Sun Care"],
    "Sports & Outdoors": ["Fitness Equipment", "Outdoor Gear", "Cycling", "Camping", "Sportswear", "Footwear", "Winter Sports", "Fishing", "Water Sports", "Team Sports"],
    "Office Equipment": ["Office Electronics", "Stationery", "Paper", "Printers", "Office Furniture", "Organizers", "Writing Instruments", "Calendars", "Folders", "Office Decor"]
}

EUROPEAN_BRANDS = {
    "Electronics": ["Samsung", "Philips", "Siemens", "Nokia", "Bosch", "Grundig", "Thomson", "Beko", "AEG", "Miele", "Braun"],
    "Fashion": ["H&M", "Zara", "Mango", "Adidas", "Puma", "Benetton", "Desigual", "Burberry", "Lacoste", "Hugo Boss"],
    "Home & Garden": ["IKEA", "Villeroy & Boch", "Le Creuset", "Alessi", "Fiskars", "Bosch Garden", "Gardena", "Kärcher", "Leifheit", "WMF"],
    "Health & Beauty": ["L'Oréal", "Nivea", "Yves Rocher", "Vichy", "Bioderma", "Kiko Milano", "The Body Shop", "Rituals", "Kneipp", "Essence"],
    "Sports & Outdoors": ["Decathlon", "Salomon", "Head", "Fischer", "Rossignol", "Intersport", "Jack Wolfskin", "Mammut", "Fjällräven", "Vaude"],
    "Office Equipment": ["Canon", "HP", "Epson", "Brother", "Xerox", "Ricoh", "Pelikan", "Stabilo", "Lamy", "Moleskine", "Leuchtturm1917", "Rhodia"]
}

def generate_brands_for_category(category):
    """Generate a list of 50 brand names for a given category, using both real and synthetic brand logic."""
    category_specific = {
        "Electronics": {"country": "DE", "suffixes": ["AG", "GmbH", "Group"]},
        "Fashion": {"country": "ES", "suffixes": ["SL", "SA", "Moda"]},
        "Home & Garden": {"country": "FR", "suffixes": ["SAS", "SARL", "Le"]},
        "Health & Beauty": {"country": "IT", "suffixes": ["SRL", "SPA", "Bella"]},
        "Sports & Outdoors": {"country": "GB", "suffixes": ["Ltd", "PLC", "Outdoors"]},
        "Office Equipment": {"country": "DE", "suffixes": ["GmbH", "AG", "Solutions"]}
    }
    brands = EUROPEAN_BRANDS.get(category, []).copy()
    config = category_specific.get(category, {})
    country_code = config.get('country', 'GB')
    if len(brands) < 50:
        local_faker = Faker(get_safe_locale(country_code))
        local_faker.seed_instance(SEED)
        for _ in range(50 - len(brands)):
            name = to_ascii(local_faker.last_name())
            suffix = random.choice(config.get('suffixes', ['Ltd']))
            brands.append(f"{name} {suffix}")
    return random.sample(brands, min(50, len(brands)))

def generate_product_name(category, subcategory, brand):
    """Generate a plausible product name using category, subcategory, and brand."""
    naming_rules = {
        "Electronics": [
            lambda: f"{brand} {random.choice(['Pro','Max','Ultra'])} {subcategory} {random.randint(2023,2025)}",
            lambda: f"{subcategory} {random.choice(['Series','Edition','X'])} - {brand}"
        ],
        "Fashion": [
            lambda: f"{brand} {to_ascii(faker.color_name())} {subcategory.replace(chr(39)+'s', '')}",
            lambda: f"{subcategory} by {brand.split()[0]} {random.choice(['Collection','Line','Edition'])}"
        ],
        "Home & Garden": [
            lambda: f"{brand} {subcategory} {random.choice(['Set','Kit','Bundle'])}",
            lambda: f"{subcategory} {random.choice(['System','Pro','Smart'])} - {brand}"
        ],
        "Health & Beauty": [
            lambda: f"{brand} {random.choice(['Professional','Clinical','Expert'])} {subcategory}",
            lambda: f"{subcategory} {random.choice(['Care','Therapy','Solution'])} - {brand}"
        ]
    }
    formats = naming_rules.get(category, [lambda: f"{brand} {subcategory}"])
    product_name = random.choice(formats)()
    return to_ascii(product_name)

def calculate_pricing(category):
    """Calculate cost, unit price, and margin for a product based on its category."""
    margin_profiles = {
        "Electronics": (0.15, 0.25),
        "Fashion": (0.40, 0.70),
        "Home & Garden": (0.25, 0.45),
        "Health & Beauty": (0.50, 0.80),
        "Sports & Outdoors": (0.30, 0.60),
        "Office Equipment": (0.20, 0.40)
    }
    min_margin, max_margin = margin_profiles.get(category, (0.20, 0.50))
    cost = round(random.uniform(5, 500), 2)
    margin = random.uniform(min_margin, max_margin)
    price = cost * (1 + margin + random.gauss(0, 0.03))
    return cost, round(price, 2), f"{round(margin*100, 1)}%"

def generate_products(total_products=10000):
    """Generate a synthetic product catalog."""
    products = []
    product_ids = set()
    for product_id in range(1, total_products + 1):
        category = random.choice(list(CATEGORIES.keys()))
        subcategory = random.choice(CATEGORIES[category])
        brand = to_ascii(random.choice(generate_brands_for_category(category)))
        cost_price, unit_price, profit_margin = calculate_pricing(category)
        product_name = generate_product_name(category, subcategory, brand)
        rating = max(1.0, min(5.0, random.normalvariate(4.2, 0.4)))
        reviews = int(abs(random.gauss(300, 150)))
        if product_id in product_ids:
            continue
        product_ids.add(product_id)
        products.append([
            product_id,
            product_name,
            category,
            subcategory,
            brand,
            unit_price,
            cost_price,
            round(rating, 1),
            max(0, reviews)
        ])
    columns = [
        "product_id", "product_name", "category", "sub_category", "brand",
        "unit_price", "cost_price", "rating", "review_count"
    ]
    return pd.DataFrame(products, columns=columns)

def validate_products(products_df):
    """Validate that product_ids are unique and all fields are not null."""
    assert products_df['product_id'].is_unique, "Product IDs are not unique."
    assert products_df.notnull().all().all(), "Null values found in product DataFrame."
    print("Product DataFrame validations passed.")

def clamp(val, lower, upper):
    """Clamp a value between lower and upper bounds."""
    return min(max(val, lower), upper)

def generate_product_prices(products_df, eco_sitch_df):
    """
    For each product and country (and year), generate a local price using a blended economic indicator.
    Returns a DataFrame with columns: product_id, country_code, local_price, price_type, effective_date
    """
    eco = eco_sitch_df.copy()
    eco['country_code'] = eco['country'].map(REVERSE_ISO_MAP)
    rows = []
    for _, prod in products_df.iterrows():
        for _, econ in eco.iterrows():
            blended = 1 + (econ['pps_norm'] - 0.5) * 2
            multiplier = clamp(blended, 0.85, 1.2)
            local_price = round(prod['unit_price'] * multiplier, 2)
            rows.append({
                'product_id': int(prod['product_id']),
                'country_code': econ['country_code'],
                'local_price': local_price,
                'price_type': "Retail",
                'effective_date': f"{int(econ['year'])}-01-01"
            })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    print("Generating product data...")
    products_df = generate_products(10000)
    validate_products(products_df)
    products_df.to_csv('products.csv', index=False)
    print(f"products.csv written with {len(products_df)} products.")

    print("Generating country-adjusted product prices...")
    eco_sitch_df = pd.read_csv('/content/drive/MyDrive/etrade/eco_sitch.csv')
    product_prices_df = generate_product_prices(products_df, eco_sitch_df)
    product_prices_df.to_csv('product_prices.csv', index=False)
    print(f"product_prices.csv written with {len(product_prices_df)} rows.")
