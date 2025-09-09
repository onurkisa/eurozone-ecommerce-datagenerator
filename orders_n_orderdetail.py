"""
orders_n_orderdetail.py

Generates realistic, synthetic e-commerce order and order_detail tables for Eurozone cities,
leveraging Phase 1 outputs for rigorous business logic, allocation, and economic realism.
The script is engineered for scalability, memory safety, and high integrity, supporting millions of rows via batch processing.

Inputs:
    - customers.csv: List of all customers.
    - customer_addresses.csv: Primary and secondary addresses for customers.
    - products.csv: Master product catalog (global).
    - product_prices.csv: Country-specific prices for products.
    - shipment_company.csv: Shipment/delivery providers.
    - city_level_allocation.csv: Phase 1 output for city-level customer and order allocations.
    - monthly_orders_df.csv: Phase 1 output for monthly order targets per city/country.
    - eco_sitch.csv: Normalized purchasing power standard (pps_norm) per country (Phase 1 output).

Outputs:
    - orders.csv: Orders table (header and schema described below).
    - order_detail.csv: Order details (basket items), 1:N with orders.
    - Logging output (order_and_detail_generation.log)

Usage:
    python orders_n_orderdetail.py

Business Rules & Logic:
    - Order volumes per city/country/month: Strictly governed by monthly_orders_df from Phase 1. Each city/country/month receives exactly the prescribed number of orders.
    - Customer allocation: Each order is assigned to a valid customer with a primary address in the target city/country. The customer-to-(city,country) mapping is derived from customer_addresses.csv (using the first address per customer).
    - Basket (order_detail) generation: Basket size, composition, and product quality are determined by the country's economic index (pps_norm), as described below. All basket contents reference valid product IDs and country-specific prices.
    - Basket behavior by economic index: The `adjust_basket_behavior` function divides countries into three economic tiers based on pps_norm:
        - High/medium-high (pps_norm >= 0.5): Larger baskets (up to 6 items), higher probability of premium products, higher quantity per product, and greater price tolerance.
        - Lower-medium (0.3 < pps_norm < 0.5): Moderate basket size (up to 4 items), moderate premium product probability, and moderate price/quantity.
        - Lowest (pps_norm <= 0.3): Smallest baskets (up to 2 items), lowest premium product probability, lowest quantity, and highest price sensitivity.
      These parameters control: max basket size, probability of premium products, allowable price deviation, minimum item price ratio, and quantity multipliers.
    - Order-to-order_detail (1:N) relationship: Each order may have multiple order_detail rows (one per unique product in the basket). This reflects real-world scenarios where a single order contains multiple items. 
        In the data warehouse star schema (Phase 3), order_detail will serve as the sales_fact table, with each row representing a unique product purchased within an order (fact row). This design enables granular sales analysis by product, customer, time, etc.
    - Returns and cancellations**: Both are simulated using configurable probabilities. Cancelled orders are always marked as returned, while completed orders may be randomly returned based on RETURN_PROBABILITY. Return reasons are sampled from a predefined list.
    - All foreign keys (products, shipment companies, addresses, etc.) are referenced from dimension tables to ensure referential integrity.
    - Product pricing: For each product in a basket, the unit price is always the country-specific value from product_prices.csv. The global price in products.csv is never used for transaction data.
    - Sales amount and total price: For each order_detail, sales_amount = (unit_price x quantity) - discount_amount (discounts are currently zero, but logic is extensible). The order's total_price is the sum of its order_details' sales_amounts.
    - City/customer/order allocation, basket generation, and pricing: All are strictly governed by Phase 1 outputs, ensuring that simulated data matches prescribed demand, allocation, and economic structure.
    - Batching and scalability: Data is generated in large but manageable batches (see ORDER_DETAILS_BATCH_SIZE and ORDERS_BATCH_SIZE), making it suitable for memory-constrained environments.
    - Script is highly modular, memory-efficient, and outputs data in manageable batches using pandas.
"""

import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta
import uuid
import logging
import psutil
from tqdm import tqdm
import gc
from pathlib import Path
import warnings
import multiprocessing as mp
import concurrent.futures

warnings.filterwarnings('ignore')

### - Settings - ###

# Specify all years present in the input data here (e.g., [2023, 2024])
YEARS = [2023, 2024]

CSV_DATA_DIR = '/content/drive/MyDrive/etrade/outputs/'
CUSTOMERS_CSV_PATH = os.path.join(CSV_DATA_DIR, 'customers.csv')
ADDRESS_CSV_PATH = os.path.join(CSV_DATA_DIR, 'customer_addresses.csv')
CITY_LEVEL_CSV_PATH = os.path.join(CSV_DATA_DIR, 'city_level_allocation.csv')
MONTHLY_ORDERS_CSV_PATH = os.path.join(CSV_DATA_DIR, 'monthly_orders_df.csv')
ECO_SITCH_CSV_PATH = os.path.join(CSV_DATA_DIR, 'eco_sitch.csv') 
LOG_FILE = os.path.join(CSV_DATA_DIR, 'order_and_detail_generation.log')
OUTPUT_DIR = '/content/drive/MyDrive/etrade/outputs/testing/'

NUM_ORDERS_TARGET = 5_000_000
ORDERS_BATCH_SIZE = 500_000
ORDER_DETAILS_BATCH_SIZE = 1_000_000
# Each order can have multiple order_detail rows (1:N relationship).
# This is crucial for the data warehouse/star schema design: order_detail will become the sales_fact table in Phase 3,
# with each row representing a unique product purchased (fact row) within an order. This enables detailed sales analytics.

RETURN_PROBABILITY = 0.05
RETURN_REASONS = ["Damaged item", "Wrong item received", "Size/fit issue", "Changed mind", "Better price elsewhere", "Late delivery"]

TEMP_COUNTRY_DATA_DIR = 'temp_country_data_orders_details'

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Memory Monitoring ---
def get_memory_usage() -> float:
    """Returns current process memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)

def log_memory_usage(stage: str) -> None:
    """Log memory usage at a given stage."""
    memory_gb = get_memory_usage()
    logger.info(f"{stage} - Memory Usage: {memory_gb:.2f} GB")

# --- Seed for reproducibility---
random.seed(42)
np.random.seed(42)

def load_ids_from_csv(file_path: str, id_column_name: str):
    """Load IDs from a CSV file column.
    Args:
        file_path (str): Path to the CSV file.
        id_column_name (str): Name of the ID column.
    Returns:
        np.ndarray: Array of IDs.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: '{file_path}' not found!")
    df = pd.read_csv(file_path, usecols=[id_column_name], dtype={id_column_name: 'string'})
    return df[id_column_name].dropna().to_numpy(dtype=object)

def create_customer_country_city_map(customers_csv_path: str, addresses_csv_path: str):
    """Build nested map of customers by country and city.
    Args:
        customers_csv_path (str): Path to customers.csv.
        addresses_csv_path (str): Path to customer_addresses.csv.
    Returns:
        dict: Nested map[country_code][city] = list of (customer_id, registration_datetime).
    """
    if not os.path.exists(customers_csv_path):
        raise FileNotFoundError(f"'{customers_csv_path}' not found!")
    if not os.path.exists(addresses_csv_path):
        raise FileNotFoundError(f"'{addresses_csv_path}' not found!")

    logger.info("Building customer country/city map using merged address info (first address per customer)...")
    df_customer = pd.read_csv(
        customers_csv_path,
        usecols=['customer_id','registration_datetime'],
        dtype={'customer_id': 'string'}
    )
    df_customer['registration_datetime'] = pd.to_datetime(df_customer['registration_datetime'], errors='coerce')
    df_address = pd.read_csv(
        addresses_csv_path, usecols=['customer_id','country_code','city'],
        dtype={'customer_id':'string','country_code':'category','city':'category'})
    df_customers = pd.merge(
        df_customer,
        df_address[['customer_id','country_code','city']],
        on = 'customer_id',
        how='inner'
    )
    customer_country_city_map = {}
    for _, row in df_customers.iterrows():
        country, city = row['country_code'], row['city']
        customer_id, reg_dt = row['customer_id'], row['registration_datetime']
        if pd.isna(customer_id) or pd.isna(country) or pd.isna(city):
            continue
        customer_country_city_map \
            .setdefault(country, {}) \
            .setdefault(city, []) \
            .append((customer_id, reg_dt))
    for country, cities in customer_country_city_map.items():
        for city, lst in cities.items():
            customer_country_city_map[country][city] = lst
    logger.info("Customer country/city mapping ready!")
    return customer_country_city_map

#### Create helper functions ####

def load_eurostat_data(file_path: str = ECO_SITCH_CSV_PATH):
    """Load Eurostat economic index data from CSV.
    Args:
        file_path (str): Path to eco_sitch.csv.
    Returns:
        dict: Mapping of (country_code, year) to pps_norm.
    """
    try:
        eco_sitch = pd.read_csv(file_path)
        ISO_MAP = {'AT': 'Austria', 'BE': 'Belgium', 'HR': 'Croatia', 'CY': 'Cyprus',
                   'EE': 'Estonia', 'FI': 'Finland', 'FR': 'France', 'DE': 'Germany',
                   'GR': 'Greece', 'IE': 'Ireland', 'IT': 'Italy', 'LV': 'Latvia',
                   'LT': 'Lithuania', 'LU': 'Luxembourg', 'MT': 'Malta', 'NL': 'Netherlands',
                   'PT': 'Portugal', 'SK': 'Slovakia', 'SI': 'Slovenia', 'ES': 'Spain'}
        REVERSE_ISO_MAP = {v: k for k, v in ISO_MAP.items()}
        eco_sitch['country_code'] = eco_sitch['country'].map(REVERSE_ISO_MAP)
        eco_sitch['country_code'] = eco_sitch['country_code'].astype('category')
        eco_sitch['pps_norm'] = pd.to_numeric(eco_sitch['pps_norm']).astype('float32')
        if 'year' not in eco_sitch.columns:
            eco_sitch['year'] = YEARS[0]  # fallback if year missing
        logger.info(f"eco_sitch loaded from '{file_path}'")
        # return mapping of (country_code, year) -> pps_norm
        return {(row['country_code'], int(row['year'])): row['pps_norm'] for _, row in eco_sitch.iterrows()}
    except FileNotFoundError: 
        logger.error(f"eco_sitch not found: {file_path}. Please ensure the file is in the correct location.")
        raise
    except Exception as e:
        logger.error(f"Error loading Eurostat data: {e}")
        raise

def load_distribution_data(city_level_file: str = CITY_LEVEL_CSV_PATH, monthly_orders_file: str = MONTHLY_ORDERS_CSV_PATH):
    """Load city-level allocation and monthly orders data from CSVs.
    Args:
        city_level_file (str): Path to city_level_allocation.csv.
        monthly_orders_file (str): Path to monthly_orders_df.csv.
    Returns:
        tuple: (city_level DataFrame, monthly_orders DataFrame) with all years.
    """
    try:
        city_level_dtypes = {
            'country': 'category', 'country_code': 'category', 'province_name': 'category',
            'province_id': 'string','city_weight_in_country':'float32','year':'int32', 'allocated_customers': 'int32',
            'allocated_orders':'int32', 'city_allocated_customers':'int32', 'city_allocated_orders':'int32',
            '1_or_2_times':'float32', '3_to_5_times':'float32', '6_to_10_times':'float32', 'more_than_10_times':'float32',
            'city': 'category', 'allocated_orders': 'float32','cust_10_plus': 'float32', 'orders_10_plus': 'float32',
            'total_orders_calc': 'float32', 'sum_segment_customers': 'float32', 'valid_customers':'bool','order_gap':'float32'
        }

        for i in range(1, 11):
            city_level_dtypes[f'cust_{i}'] = 'float32'
        monthly_dtypes = {
            'country': 'category', 'country_code': 'category', 'province_name': 'category',
            'province_id': 'string', 'city': 'category', 'year': 'int32',
            'city_allocated_orders': 'int32'
        }

        for month_abbr in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']:
            monthly_dtypes[f'{month_abbr}_orders'] = 'int32'
        city_df = pd.read_csv(city_level_file, dtype=city_level_dtypes)
        monthly_df = pd.read_csv(monthly_orders_file, dtype=monthly_dtypes)

        # --- Convert all relevant columns to rounded integer values after loading ---
        city_int_cols = ['city_allocated_customers', 'city_allocated_orders', 'cust_10_plus', 'orders_10_plus'
                         ] + [f'cust_{i}' for i in range(1, 11)]
        for col in city_int_cols:
            if col in city_df.columns:
                city_df[col] = np.round(city_df[col]).astype('int32')

        logger.info(f"Distribution data loaded: {city_level_file}, {monthly_orders_file}")
        logger.info(f"city_level_allocation: {len(city_df):,} rows, monthly_orders_df: {len(monthly_df):,} rows loaded.")
        return city_df, monthly_df
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}. Please ensure the files are in the correct location.")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def adjust_basket_behavior(pps_norm: float) -> dict:
    """Determine basket generation parameters based on economic index.
    Args:
        pps_norm (float): Economic index for the country.
    Returns:
        dict: Basket configuration parameters.
    """
    if pps_norm >= 0.5:  # High/medium-high economic maturity
        return {
            'max_items_per_basket': 6,           # Up to 6 different items in a basket
            'premium_product_probability': 0.40, # 40% chance to include premium products
            'quantity_multiplier': 1.4,          # Higher average quantity per item
            'price_tolerance_factor': 0.20,      # ±20% deviation from target total allowed
            'min_unit_price_ratio': 0.03         # No item can be less than 3% of basket total
        }
    elif 0.3 < pps_norm < 0.5:  # Lower-medium economic maturity
        return {
            'max_items_per_basket': 4,           # Up to 4 items
            'premium_product_probability': 0.20, # 20% chance for premium products
            'quantity_multiplier': 1.1,          # Baseline average quantity
            'price_tolerance_factor': 0.25,      # ±25% deviation allowed
            'min_unit_price_ratio': 0.06         # No item less than 6% of basket total
        }
    else:  # Lowest economic maturity
        return {
            'max_items_per_basket': 2,           # Up to 2 items only
            'premium_product_probability': 0.05, # Only 5% chance for premium products
            'quantity_multiplier': 0.9,          # Lower average quantity per item
            'price_tolerance_factor': 0.33,      # ±33% deviation allowed (higher price sensitivity)
            'min_unit_price_ratio': 0.12        # No item less than 12% of basket total
        }

def save_country_data_to_files(country_data: dict, temp_folder: str) -> str:
    """Save country-level product data to temporary files.
    Args:
        country_data (dict): Product data per country.
        temp_folder (str): Directory to save files.
    Returns:
        str: Path to temp_folder.
    """
    os.makedirs(temp_folder, exist_ok=True)
    for country_code, data in country_data.items():
        path = os.path.join(temp_folder, f"{country_code}_data.pkl")
        try:
            essential_data = {
                'all_products': data['all_products'][['product_id', 'local_price', 'category', 'sub_category']],
                'by_category': {
                    cat: df[['product_id', 'local_price', 'category', 'sub_category']]
                    for cat, df in data['by_category'].items()
                }
            }
            pd.to_pickle(essential_data, path)
        except Exception as e:
            logger.error(f"Country {country_code} data could not be saved: {e}")
    logger.info(f"Country data saved to {temp_folder} ")
    return temp_folder

def load_country_data_from_file(country_code: str, temp_folder: str):
    """Load product data for a specific country from file.
    Args:
        country_code (str): Country code.
        temp_folder (str): Directory containing country data files.
    Returns:
        dict or None: Country product data or None if not found/error.
    """
    path = Path(temp_folder) / f"{country_code}_data.pkl"
    if not path.exists():
        logger.warning(f"Country file not found: {path}")
        return None
    try:
        return pd.read_pickle(path)
    except Exception as e:
        logger.error(f"Country data could not be loaded  ({country_code}): {e}")
        return None

def preprocess_country_data_optimized_for_products(
    product_prices_df: pd.DataFrame, products_df: pd.DataFrame, memory_threshold_gb: float = 6.0
) -> str:
    """Preprocess product and price data by country and save to temp files.
    Args:
        product_prices_df (pd.DataFrame): DataFrame of product prices.
        products_df (pd.DataFrame): DataFrame of products.
        memory_threshold_gb (float): Memory threshold for GC.
    Returns:
        str: Path to temp folder containing country data.
    """
    logger.info("Country-level product data preprocessing started...")
    log_memory_usage("Product preprocessing start")
    base_products = products_df[['product_id', 'category', 'sub_category']].copy()
    base_products['category'] = base_products['category'].astype('category')
    base_products['sub_category'] = base_products['sub_category'].astype('category')

    base_prices = product_prices_df[['product_id', 'country_code', 'local_price']].copy()
    base_prices['local_price'] = pd.to_numeric(base_prices['local_price']).astype('float32')
    base_prices['country_code'] = base_prices['country_code'].astype('category')

    country_data = {}
    countries = base_prices['country_code'].unique()

    logger.info(f"Preparing product data for  {len(countries)} countries.")

    for i, country in enumerate(tqdm(countries, desc="Preparing country product data")):
        try:
            country_prices = base_prices[base_prices['country_code'] == country].copy()
            country_products = country_prices.merge(
                base_products, on='product_id', how='inner'
            )
            if country_products.empty:
                continue
            category_groups = {
                cat: df.copy() for cat, df in country_products.groupby('category')
            }
            country_data[country] = {
                'all_products': country_products,
                'by_category': category_groups
            }
            if i % 10 == 0 and get_memory_usage() > memory_threshold_gb:
                gc.collect()
                log_memory_usage(f"Memory cleanup (product preprocessing, country {i})")
        except Exception as e:
            logger.error(f"Error processing product data for country {country} : {e}")
            continue
    logger.info(f"Product data prepared for {len(country_data)} countries.")
    log_memory_usage("Product preprocessing completed")

    temp_folder = save_country_data_to_files(country_data, TEMP_COUNTRY_DATA_DIR)
    del country_data, base_products, base_prices
    gc.collect()
    return temp_folder

def create_realistic_basket_improved(
    target_total: float, country_code: str, pps_norm: float, temp_folder: str, max_attempts: int = 30
) -> list:
    """Generate a realistic basket (order_detail rows) for a single order.
    Args:
        target_total (float): Target basket value.
        country_code (str): Country code for pricing.
        pps_norm (float): Economic index for country.
        temp_folder (str): Folder with preprocessed product data.
        max_attempts (int): Maximum number of attempts for basket assembly.
    Returns:
        list[dict]: List of order_detail dicts for the basket.
    """
    basket_config = adjust_basket_behavior(pps_norm)
    country_data = load_country_data_from_file(country_code, temp_folder)

    if not country_data or country_data['all_products'].empty:
        return []
    available_products = country_data['all_products']
    basket_items = []
    current_total = 0.0
    min_target = target_total * (1 - basket_config['price_tolerance_factor'])
    max_target = target_total * (1 + basket_config['price_tolerance_factor'])

    attempts = 0
    while (
        current_total < min_target and
        attempts < max_attempts and
        len(basket_items) < basket_config['max_items_per_basket']
    ):
        attempts += 1
        remaining_budget = max_target - current_total

        # Only allow products that fit within the remaining budget and are not too cheap
        eligible_products = available_products[
            (available_products['local_price'] <= remaining_budget) &
            (available_products['local_price'] >= target_total * basket_config['min_unit_price_ratio'])
        ]

        # If no eligible product, relax criteria slightly
        if eligible_products.empty:
            eligible_products = available_products[available_products['local_price'] <= remaining_budget * 1.5]
            if eligible_products.empty:
                break

        try:
            product = eligible_products.sample(1).iloc[0]
        except Exception as e:
            logger.debug(f"Product selection failed: {e}")
            break

        unit_price = product['local_price']
        if unit_price <= 0:
            continue

        # Calculate max quantity for this product based on remaining budget, basket config, and a hard cap
        max_quantity = min(
            basket_config['max_items_per_basket'],
            int(remaining_budget / unit_price),
            5
        )
        if max_quantity < 1:
            continue

        # Randomly select quantity (can be further refined using quantity_multiplier or demand curve)
        quantity = random.randint(1, max_quantity)
        total_item_price = unit_price * quantity

        discount_amount = 0.00  # Currently, all prices are retail; discounts can be simulated by changing this.
        sales_amount = round(total_item_price - discount_amount, 2)

        basket_items.append({
            "product_id": product['product_id'],
            "quantity": quantity,
            "unit_price": unit_price,
            "discount_amount": discount_amount,
            "sales_amount": sales_amount
        })

        current_total += sales_amount

    # Fallback: If basket is empty, add a single random item to ensure every order has at least one product
    if not basket_items:
        try:
            product = available_products.sample(1).iloc[0]
            unit_price = product['local_price']
            basket_items.append({
                "product_id": product['product_id'],
                "quantity": 1,
                "unit_price": unit_price,
                "discount_amount": 0.00,
                "sales_amount": round(unit_price, 2)
            })
        except Exception as e:
            logger.warning(f"Fallback basket could not be created: {e}")
    return basket_items


class OrderAndDetailGenerator:
    """Generates orders and order_detail rows for a city/country/month/year."""
    def __init__(
        self,
        customer_country_city_map,
        all_product_ids,
        all_shipment_company_ids,
        all_shipping_address_ids,
        city_level_orders_df,
        monthly_orders_df,
        eurostat_econ_map,
        temp_product_data_folder
    ):
        """
        Args:
            customer_country_city_map (dict): Nested customer map by country/city.
            all_product_ids (np.ndarray): Array of product IDs.
            all_shipment_company_ids (np.ndarray): Array of shipment company IDs.
            all_shipping_address_ids (np.ndarray): Array of shipping address IDs.
            city_level_orders_df (pd.DataFrame): City-level allocation DataFrame.
            monthly_orders_df (pd.DataFrame): Monthly orders DataFrame.
            eurostat_econ_map (dict): Mapping of (country_code, year) to pps_norm.
            temp_product_data_folder (str): Path to product data folder.
        """
        self.customer_country_city_map = customer_country_city_map
        self.all_product_ids = all_product_ids
        self.all_shipment_company_ids = all_shipment_company_ids
        self.all_shipping_address_ids = all_shipping_address_ids
        self.city_level_orders_df = city_level_orders_df
        self.monthly_orders_df = monthly_orders_df
        self.eurostat_econ_map = eurostat_econ_map
        self.temp_product_data_folder = temp_product_data_folder
        self.city_stats_dict = self._prepare_city_stats_dict()
        self.monthly_orders_by_city_country_year = self._prepare_monthly_orders_by_city_country_year()

    def _prepare_city_stats_dict(self) -> dict:
        """Prepare mapping of (country_code, city, year) to city stats dict."""
        city_stats_dict = {}
        for _, row in self.city_level_orders_df.iterrows():
            key = (row['country_code'], row['city'], int(row['year']))
            city_stats_dict[key] = row.to_dict()
        return city_stats_dict

    def _prepare_monthly_orders_by_city_country_year(self) -> dict:
        """Prepare mapping of (country_code, city, year) to dict of monthly orders."""
        monthly_orders_dict = {}
        for _, row in self.monthly_orders_df.iterrows():
            key = (row['country_code'], row['city'], int(row['year']))
            monthly_orders_dict[key] = {
                f'{month}_orders': row[f'{month}_orders']
                for month in ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
            }
        return monthly_orders_dict

    def _select_customer_for_order(self, country_code, city, year, city_stats):
        """Select a customer and frequency for an order."""
        customers = self.customer_country_city_map.get(country_code, {}).get(city, [])
        if len(customers) == 0:
            return None, None, None
        segment_counts = {f'cust_{i}': city_stats.get(f'cust_{i}', 0) for i in range(1, 11)}
        segment_counts['cust_10_plus'] = city_stats.get('cust_10_plus', 0)
        total = sum(segment_counts.values())
        customer_info = random.choice(customers)
        customer_id, reg_date = customer_info[0], customer_info[1]
        if total == 0:
            frequency = random.randint(1, 3)
        else:
            weights = np.array(list(segment_counts.values()), dtype=float)
            probabilities = weights / weights.sum()
            idx = np.random.choice(len(probabilities), p=probabilities)
            frequency = random.randint(11, 30) if idx == 10 else idx + 1
        return customer_id, reg_date, frequency

    def generate_orders_for_city_month_year(
        self, country_code: str, city: str, year: int, month_int: int, target_order_count_for_city_month: int
    ):
        """Generate orders and order_detail rows for a city/country/month/year."""
        orders_batch = []
        order_details_batch = []

        city_key = (country_code, city, year)
        city_stats = self.city_stats_dict.get(city_key)
        if not city_stats:
            logger.warning(f"City statistics not found: {country_code}-{city}-{year}")
            return [], []

        pps_norm = self.eurostat_econ_map.get((country_code, year), 0.6)
        month_name = datetime(year, month_int, 1).strftime('%b')
        target_orders = target_order_count_for_city_month

        generated_orders = 0
        max_attempts = target_orders * 2  # To avoid infinite loops if customer pool is small

        while generated_orders < target_orders and max_attempts > 0:
            max_attempts -= 1
            customer_id, reg_dt, freq = self._select_customer_for_order(country_code, city, year, city_stats)
            if not customer_id:
                continue

            # Simulate order frequency: higher-frequency customers can have >1 order in a month
            if freq <= 12:
                if random.random() < freq / 12.0:
                    num_orders = 1
                else:
                    continue
            else:
                avg = freq / 12.0
                num_orders = max(1, int(np.random.normal(avg, avg * 0.3)))

            for _ in range(num_orders):
                if generated_orders >= target_orders:
                    break

                order_id = str(uuid.uuid4())
                from calendar import monthrange
                last_day = monthrange(year, month_int)[1]
                order_dt = datetime(
                    year, month_int, random.randint(1, last_day),
                    random.randint(0, 23), random.randint(0, 59), random.randint(0, 59)
                )
                # Ensure order date is after customer registration
                if reg_dt > order_dt:
                    order_dt = reg_dt + timedelta(days=random.randint(1, 30))

                shipment_company_id = np.random.choice(self.all_shipment_company_ids)
                shipping_address_id = np.random.choice(self.all_shipping_address_ids)

                initial_target_price = round(random.uniform(50.0, 5000.0), 2)
                basket_items = create_realistic_basket_improved(
                    initial_target_price, country_code, pps_norm, self.temp_product_data_folder
                )
                if not basket_items:
                    continue

                total_price = 0.0
                for item in basket_items:
                    order_detail_id = str(uuid.uuid4())
                    order_details_batch.append({
                        "order_detail_id": order_detail_id,
                        "order_id": order_id,
                        "product_id": item['product_id'],
                        "quantity": item['quantity'],
                        "unit_price": item['unit_price'],
                        "discount_amount": item['discount_amount'],
                        "sales_amount": item['sales_amount'],
                        "year": year
                    })
                    total_price += item['sales_amount']

                is_cancelled = random.random() < RETURN_PROBABILITY
                order_status = 'Cancelled' if is_cancelled else 'Completed'
                is_returned = is_cancelled or (random.random() < RETURN_PROBABILITY)
                return_reason = random.choice(RETURN_REASONS) if is_returned else None

                orders_batch.append({
                    "order_id": order_id,
                    "customer_id": customer_id,
                    "order_date": order_dt.date(),
                    "order_time": order_dt.time(),
                    "shipping_address_id": shipping_address_id,
                    "shipment_company_id": shipment_company_id,
                    "total_price": round(total_price, 2),
                    "order_status": order_status,
                    "is_cancelled": is_cancelled,
                    "is_returned": is_returned,
                    "return_reason": return_reason,
                    "country_code": country_code,
                    "year": year
                })
                generated_orders += 1

        return orders_batch, order_details_batch


def process_city_month_chunk(chunk_data_with_generator_params):
    """Process a city-month-year data chunk for multiprocessing."""
    try:
        (city_country_year_key, month_int, target_order_count,
         customer_country_city_map, all_product_ids, all_shipment_company_ids,
         all_shipping_address_ids,
         city_level_orders_df, monthly_orders_df, eurostat_econ_map, temp_product_data_folder
        ) = chunk_data_with_generator_params

        country_code, city, year = city_country_year_key
        worker_id = mp.current_process().name

        worker_logger = logging.getLogger(f"worker_{worker_id}")
        if not worker_logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(f"%(asctime)s - WORKER_{worker_id} - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            worker_logger.addHandler(handler)
            worker_logger.setLevel(logging.INFO)

        worker_logger.info(f"Worker started: {country_code}-{city}-{year}, Month {month_int}, Target: {target_order_count}")

        if country_code not in customer_country_city_map:
            worker_logger.error(f"Country '{country_code}' not found in customer_country_city_map!")
            worker_logger.info(f"Available countries: {list(customer_country_city_map.keys())[:10]}...")
            return [], []

        if city not in customer_country_city_map[country_code]:
            worker_logger.error(f"City '{city}' not found under country '{country_code}' !")
            worker_logger.info(f"'{country_code}' Available cities for  {list(customer_country_city_map[country_code].keys())[:10]}...")
            return [], []

        customer_count = len(customer_country_city_map[country_code][city])
        worker_logger.info(f" {customer_count} customers found for city '{city}'")

        if not os.path.exists(temp_product_data_folder):
            worker_logger.error(f"temp_product_data_folder not found: {temp_product_data_folder}")
            return [], []

        country_product_file = os.path.join(temp_product_data_folder, f"{country_code}_data.pkl")
        if not os.path.exists(country_product_file):
            worker_logger.error(f"Country product file not found: {country_product_file}")
            available_files = os.listdir(temp_product_data_folder) if os.path.exists(temp_product_data_folder) else []
            worker_logger.info(f"Available product files: {available_files[:10]}...")
            return [], []

        generator = OrderAndDetailGenerator(
            customer_country_city_map, all_product_ids, all_shipment_company_ids,
            all_shipping_address_ids, city_level_orders_df, monthly_orders_df,
            eurostat_econ_map, temp_product_data_folder
        )

        worker_logger.info(f"[{country_code}-{city}-{year}, Month {month_int}] - Generator created, order generation starting...")

        orders_chunk, order_details_chunk = generator.generate_orders_for_city_month_year(
            country_code, city, year, month_int, target_order_count
        )

        worker_logger.info(f"[{country_code}-{city}-{year}, Month {month_int}] - GENERATED: Orders: {len(orders_chunk)}, OrderDetails: {len(order_details_chunk)}")

        if len(orders_chunk) == 0:
            worker_logger.error(f"[{country_code}-{city}-{year}, Month {month_int}] - NO ORDERS GENERATED! Target:{target_order_count}")

        return orders_chunk, order_details_chunk

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        worker_logger.error(f"Worker ERROR - {country_code}-{city}-{year}, Month {month_int}: {e}")
        worker_logger.error(f"Error details:\n{error_detail}")
        return [], []

    finally:
        gc.collect()

def validate_data(df: pd.DataFrame, df_name: str) -> None:
    """Run basic validation checks on generated DataFrames.
    Args:
        df (pd.DataFrame): DataFrame to validate.
        df_name (str): Name for logging.
    """
    if df.empty:
        raise ValueError(f"{df_name} data is empty!")
    if 'order_id' in df.columns and df['order_id'].duplicated().any():
        logger.warning(f"Duplicate 'order_id' found in {df_name}. ")
    if 'total_price' in df.columns and (df['total_price'] < 0).any():
        logger.warning(f"Negative 'total_price' found in {df_name}.")
    if 'sales_amount' in df.columns and (df['sales_amount'] < 0).any():
        logger.warning(f"Negative 'sales_amount' found in {df_name}.")
    logger.info(f"Validation completed for {df_name}.")

def save_data_to_csv(
    data_buffer: list,
    file_prefix: str,
    output_dir: str,
    month_name: str,
    file_part_index: int,
    dtypes_map: dict
) -> None:
    """Convert data list to DataFrame, apply dtypes, and save as CSV file.
    Args:
        data_buffer (list): List of dicts to save.
        file_prefix (str): Prefix for output file.
        output_dir (str): Directory to save file.
        month_name (str): Month name for file.
        file_part_index (int): Partition index.
        dtypes_map (dict): Mapping of column to dtype.
    """
    if not data_buffer:
        logger.info(f"{file_prefix}_{month_name}_part_{file_part_index}.csv has empty data, file not created.")
        return
    df = pd.DataFrame(data_buffer)
    for col, dtype in dtypes_map.items():
        if col in df.columns:
            try:
                if dtype == 'datetime64[ns]':
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                elif dtype == 'string':
                    df[col] = df[col].astype(str)
                else:
                    df[col] = df[col].astype(dtype)
            except Exception as e:
                logger.warning(f"Column '{col}' dtype conversion error to '{dtype}': {e}")
    file_name = f'{file_prefix}_{month_name}_part_{file_part_index}.csv'
    file_path = os.path.join(output_dir, file_name)
    try:
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.info(f"Wrote {len(df):,} records to '{file_name}'.")
    except Exception as e:
        logger.error(f"CSV write error: {file_name} - {e}")
    del df
    gc.collect()

# --- Main Production and Orchestration Function ---
def generate_all_orders_and_details_parallel() -> None:
    """Generate all orders and order details in parallel and save as CSV files."""
    logger.info("===  ORDER AND ORDER DETAIL GENERATION PROCESS STARTED (MONTHLY SEQUENCE) ===")
    log_memory_usage("Start")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

 # --- 1- Load Required Master Data ---
    try:
        required_files = [
            (os.path.join(CSV_DATA_DIR, 'products.csv'), 'products.csv'),
            (os.path.join(CSV_DATA_DIR, 'shipment_company.csv'), 'shipment_company.csv'),
            (CUSTOMERS_CSV_PATH, 'customers.csv'),
            (ADDRESS_CSV_PATH, 'customer_addresses.csv'),
            (CITY_LEVEL_CSV_PATH, 'city_level_allocation.csv'),
            (MONTHLY_ORDERS_CSV_PATH, 'monthly_orders_df.csv'),
            (ECO_SITCH_CSV_PATH, 'eco_sitch.csv'),
            (os.path.join(CSV_DATA_DIR, 'product_prices.csv'), 'product_prices.csv')
        ]

        for file_path, file_name in required_files:
            if not os.path.exists(file_path):
                logger.error(f"REQUIRED FILE NOT FOUND: {file_name} -> {file_path}")
                return
            else:
                file_size = os.path.getsize(file_path) / (1024*1024) 
                logger.info(f"✓ {file_name}: {file_size:.1f} MB")

        all_product_ids = load_ids_from_csv(os.path.join(CSV_DATA_DIR, 'products.csv'), 'product_id')
        logger.info(f"Product IDs loaded: {len(all_product_ids):,} ")

        all_shipment_company_ids = load_ids_from_csv(os.path.join(CSV_DATA_DIR, 'shipment_company.csv'), 'shipment_company_id')
        logger.info(f"Shipment company IDs loaded: {len(all_shipment_company_ids):,} ")

        all_shipping_address_ids = load_ids_from_csv(ADDRESS_CSV_PATH, 'address_id')
        logger.info(f"Address IDs loaded: {len(all_shipping_address_ids):,} ")

        customer_country_city_map = create_customer_country_city_map(CUSTOMERS_CSV_PATH, ADDRESS_CSV_PATH)
        total_customers = sum(len(cities[city]) for cities in customer_country_city_map.values() for city in cities)
        logger.info(f"Customer map created: {len(customer_country_city_map)} countries, {total_customers:,} customers")

        city_level_orders_df, monthly_orders_df = load_distribution_data(CITY_LEVEL_CSV_PATH, MONTHLY_ORDERS_CSV_PATH)

        eurostat_econ_map = load_eurostat_data(ECO_SITCH_CSV_PATH)
        logger.info(f"Eurostat data loaded: {len(eurostat_econ_map)} countries")

        products_df = pd.read_csv(os.path.join(CSV_DATA_DIR, 'products.csv'))
        logger.info(f"Products DataFrame loaded: {len(products_df):,} rows")

        product_prices_df = pd.read_csv(os.path.join(CSV_DATA_DIR, 'product_prices.csv'))
        logger.info(f"Product prices DataFrame loaded: {len(product_prices_df):,} rows")

        log_memory_usage("Master data loaded")

    except Exception as e:
        logger.error(f"Master data loading error: {e}")
        import traceback
        logger.error(f"Error details:\n{traceback.format_exc()}")
        return


    # --- 2- Preprocess Country-level Product Data ---
    temp_product_data_folder = preprocess_country_data_optimized_for_products(product_prices_df, products_df)
    del products_df, product_prices_df
    gc.collect()
    log_memory_usage("Product data preprocessed and saved to disk.")

    # --- 3- Create Multi-Year Production Plan ---
    production_plan = []

    logger.info(f"monthly_orders_df shape: {monthly_orders_df.shape}")
    logger.info(f"monthly_orders_df columns: {monthly_orders_df.columns.tolist()}")
    if len(monthly_orders_df) > 0:
        logger.info(f"First 3 rows:\n{monthly_orders_df.head(3)}")
    else:
        logger.error("monthly_orders_df is empty! Production plan will also be empty.")
        return

    for _, row in monthly_orders_df.iterrows():
        country_code = row['country_code']
        city = row['city']
        year = int(row['year'])
        total_orders_for_city = 0
        for month_idx in range(1, 13):
            month_abbr = datetime(year, month_idx, 1).strftime('%b').lower()
            target_order_count = row.get(f'{month_abbr}_orders', 0)
            if target_order_count > 0:
                production_plan.append({
                    'country_code': country_code,
                    'city': city,
                    'year': year,
                    'month': month_idx,
                    'target_order_count': target_order_count
                })
                total_orders_for_city += target_order_count
        if total_orders_for_city > 0:
            logger.info(f"City: {country_code}-{city}-{year} -> Total target orders: {total_orders_for_city}")

    logger.info(f"Production plan prepared: {len(production_plan)} tasks")
    if len(production_plan) == 0:
        logger.error("Production plan is empty! No order generation will be performed.")
        return

    logger.info("First 5 production plan tasks:")
    for i, item in enumerate(production_plan[:5]):
        logger.info(f"  {i+1}. {item}")

    # Group by (year, month) for processing
    production_plan_by_year_month = {}
    for item in production_plan:
        year = item['year']
        month = item['month']
        ym_key = (year, month)
        if ym_key not in production_plan_by_year_month:
            production_plan_by_year_month[ym_key] = []
        production_plan_by_year_month[ym_key].append(item)

    sorted_year_months = sorted(production_plan_by_year_month.keys())
    logger.info(f"Production plan grouped by year and month. Year-Months to process: {sorted_year_months}")

    total_generated_orders = 0
    max_workers = mp.cpu_count()
    if max_workers < 1: max_workers = 1

    logger.info(f"Parallel generation by year and month starting: using {max_workers} workers.")
    common_generator_params = (
        customer_country_city_map, all_product_ids, all_shipment_company_ids,
        all_shipping_address_ids,
        city_level_orders_df, monthly_orders_df, eurostat_econ_map, temp_product_data_folder
    )

    for year, month_int in sorted_year_months:
        month_name_full = datetime(year, month_int, 1).strftime('%B')
        month_name_abbr = datetime(year, month_int, 1).strftime('%b')
        tasks_for_ym = production_plan_by_year_month.get((year, month_int), [])
        if not tasks_for_ym:
            logger.info(f"No production plan tasks for {year}-{month_int} ({month_name_full}).")
            continue
        logger.info(f"\n=== {year}-{month_int} ({month_name_full}) Order Generation Starting ({len(tasks_for_ym)} tasks) ===")
        log_memory_usage(f"Year {year} Month {month_name_full} Start")
        orders_file_part_idx = 1
        order_details_file_part_idx = 1
        orders_data_buffer = []
        order_details_data_buffer = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for item in tasks_for_ym:
                task_params = (
                    (item['country_code'], item['city'], item['year']),
                    item['month'],
                    item['target_order_count'],
                    *common_generator_params
                )
                futures.append(executor.submit(process_city_month_chunk, task_params))

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"{year}-{month_name_full} Orders Processing"):
                if total_generated_orders >= NUM_ORDERS_TARGET:
                    logger.warning(f"Global target order count ({NUM_ORDERS_TARGET:,}) reached. Remaining tasks are being cancelled.")
                    for f in futures:
                        if not f.done():
                            f.cancel()
                    break
                try:
                    orders_chunk, order_details_chunk = future.result(timeout=1800)
                    orders_data_buffer.extend(orders_chunk)
                    order_details_data_buffer.extend(order_details_chunk)
                    total_generated_orders += len(orders_chunk)
                    if len(orders_data_buffer) >= ORDERS_BATCH_SIZE:
                        save_data_to_csv(
                            orders_data_buffer, 'orders', OUTPUT_DIR, f"{year}_{month_name_abbr}", orders_file_part_idx,
                            dtypes_map={
                                'order_id': 'string', 'customer_id': 'string',
                                'order_date': 'datetime64[ns]', 'order_time': 'string',
                                'shipping_address_id': 'string', 'shipment_company_id': 'string',
                                'total_price': 'float32', 'order_status': 'string',
                                'is_cancelled': 'bool', 'is_returned': 'bool', 'return_reason': 'string',
                                'country_code': 'string', 'year': 'int32'
                            }
                        )
                        orders_data_buffer = []
                        orders_file_part_idx += 1
                        log_memory_usage(f"Orders written (Year {year} Month {month_name_full}, Part {orders_file_part_idx - 1})")
                    if len(order_details_data_buffer) >= ORDER_DETAILS_BATCH_SIZE:
                        save_data_to_csv(
                            order_details_data_buffer, 'order_details', OUTPUT_DIR, f"{year}_{month_name_abbr}", order_details_file_part_idx,
                            dtypes_map={
                                'order_detail_id': 'string', 'order_id': 'string',
                                'product_id': 'string', 'quantity': 'int32',
                                'unit_price': 'float32', 'discount_amount': 'float32',
                                'sales_amount': 'float32', 'year': 'int32'
                            }
                        )
                        order_details_data_buffer = []
                        order_details_file_part_idx += 1
                        log_memory_usage(f"OrderDetails written (Year {year} Month {month_name_full}, Part {order_details_file_part_idx - 1})")
                except concurrent.futures.TimeoutError:
                    logger.error(f"Year {year} Month {month_name_full} - Worker task timed out. Task cancelled.")
                    future.cancel()
                except Exception as exc:
                    import traceback
                    error_detail = traceback.format_exc()
                    logger.error(f"Year {year} Month {month_name_full} - Worker task returned error: {exc}")
                    logger.error(f"Error details:\n{error_detail}")
                    future.cancel()
                finally:
                    gc.collect()
        logger.info(f"=== Year {year} Month {month_int} ({month_name_full}) All tasks completed or cancelled. Saving remaining data. ===")
        log_memory_usage(f"Year {year} Month {month_name_full} End Memory Status")
        if orders_data_buffer:
            save_data_to_csv(
                orders_data_buffer, 'orders', OUTPUT_DIR, f"{year}_{month_name_abbr}", orders_file_part_idx,
                dtypes_map={
                    'order_id': 'string', 'customer_id': 'string',
                    'order_date': 'datetime64[ns]', 'order_time': 'string',
                    'shipping_address_id': 'string', 'shipment_company_id': 'string',
                    'total_price': 'float32', 'order_status': 'string',
                    'is_cancelled': 'bool', 'is_returned': 'bool', 'return_reason': 'string',
                    'country_code': 'string', 'year': 'int32'
                }
            )
            orders_data_buffer = []
        if order_details_data_buffer:
            save_data_to_csv(
                order_details_data_buffer, 'order_details', OUTPUT_DIR, f"{year}_{month_name_abbr}", order_details_file_part_idx,
                dtypes_map={
                    'order_detail_id': 'string', 'order_id': 'string',
                    'product_id': 'string', 'quantity': 'int32',
                    'unit_price': 'float32', 'discount_amount': 'float32',
                    'sales_amount': 'float32', 'year': 'int32'
                }
            )
            order_details_data_buffer = []
        if total_generated_orders >= NUM_ORDERS_TARGET:
            logger.warning(f"Global target order count ({NUM_ORDERS_TARGET:,}) reached, loop terminated.")
            break
        gc.collect()
        log_memory_usage(f"Year {year} Month {month_name_full} Completed - After Memory Status")

    logger.info("Year-month loop completed. Checking remaining buffers.")
    if orders_data_buffer:
        logger.warning(f"After loop, {len(orders_data_buffer)} records remain in Orders buffer. Saving without month name..")
        save_data_to_csv(
            orders_data_buffer, 'orders_remaining', OUTPUT_DIR, 'final', 1, 
            dtypes_map={
                 'order_id': 'string', 'customer_id': 'string',
                 'order_date': 'datetime64[ns]', 'order_time': 'string',
                 'shipping_address_id': 'string', 'shipment_company_id': 'string',
                 'total_price': 'float32', 'order_status': 'string',
                 'is_cancelled': 'bool', 'is_returned': 'bool', 'return_reason': 'string',
                 'country_code': 'string', 'year': 'int32'
            }
        )
    if order_details_data_buffer:
         logger.warning(f"After loop, {len(order_details_data_buffer)} records remain in OrderDetails buffer. Saving without month name.")
         save_data_to_csv(
            order_details_data_buffer, 'order_details_remaining', OUTPUT_DIR, 'final', 1, 
            dtypes_map={
                'order_detail_id': 'string', 'order_id': 'string',
                'product_id': 'string', 'quantity': 'int32',
                'unit_price': 'float32', 'discount_amount': 'float32',
                'sales_amount': 'float32', 'year': 'int32'
            }
        )
         
    logger.info(f"\nTotal {total_generated_orders:,} orders generated (target:{NUM_ORDERS_TARGET:,}).")
    logger.info(f"Orders files saved by year and month.")
    logger.info(f"OrderDetail files saved by year and month.")

    try:
        import shutil
        if os.path.exists(TEMP_COUNTRY_DATA_DIR):
            shutil.rmtree(TEMP_COUNTRY_DATA_DIR)
            logger.info("Temporary product data files cleaned up.")
    except Exception as e:
        logger.error(f"Error cleaning up temporary files: {e}")

    log_memory_usage("Process completed")
    logger.info("=== ORDER AND ORDER DETAIL GENERATION PROCESS SUCCESSFULLY COMPLETED ===")

if __name__ == "__main__":
    try:
        generate_all_orders_and_details_parallel()
    except Exception as e:
        logger.error(f"Error running main script: {e}", exc_info=True)
