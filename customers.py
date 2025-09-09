"""
customers.py

Generates synthetic customer and customer address datasets for a Eurozone e-commerce simulation project.

Customer allocations and city/month breakdowns are driven by the Phase 1 output:
'monthly_new_customers_df.csv'.

Input:
    - monthly_new_customers_df.csv: City-level monthly new customer allocations for the target year.

Outputs:
    - customers.csv: Row-level synthetic customers, with realistic demographic,
     and segmentation attributes, suitable for downstream order and transaction generation.
    - customer_addresses.csv: Linked address data, one row per customer, featuring address_id and customer_id.

Usage:
    Ensure that 'monthly_new_customers_df.csv' is available from the data prep phase.
    Then run:
        python customers.py

All statistical and business logic are consistent with Phase 1 outputs and project documentation.
Requirements: Faker-37.4.2, unidecode-1.4.0

"""

import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import unidecode

# Locale map for Eurozone countries
locale_map = {
    "DE": "de_DE", "FR": "fr_FR", "ES": "es_ES", "IT": "it_IT", "NL": "nl_NL",
    "BE": "fr_BE", "FI": "fi_FI", "AT": "de_AT", "PT": "pt_PT", "IE": "en_IE",
    "SI": "sl_SI", "SK": "sk_SK", "LT": "lt_LT", "LV": "lv_LV", "EE": "et_EE",
    "GR": "el_GR", "LU": "fr_LU", "MT": "en_GB", "CY": "en_GB"
}

# Seed for reproducibility
Faker.seed(42)
random.seed(42)
np.random.seed(42)

# Demographic distribution parameters
AVG_AGE = 28
STD_AGE = 8
MIN_AGE = 18
MAX_AGE = 75

# Customer segmentation and flags
LOYALTY_MEMBER_PROB = 0.25
FRAUD_SUSPECT_PROB = 0.02
CUSTOMER_SEGMENTS = ['New', 'Returning', 'VIP']
SEGMENT_PROBS = [0.6, 0.35, 0.05]

def get_safe_locale(country_code):
    """Return a valid Faker locale string for the given Eurozone country code."""
    from faker.config import AVAILABLE_LOCALES
    suggested_locale = locale_map.get(country_code, 'en_US')
    return suggested_locale if suggested_locale in AVAILABLE_LOCALES else 'en_US'

def to_ascii(text):
    """Convert any unicode string to ASCII using unidecode."""
    if text is None:
        return None
    return unidecode.unidecode(str(text))

def generate_gaussian_birthdate():
    """Return a random birth date for a customer using a truncated Gaussian distribution."""
    age = int(np.clip(np.random.normal(AVG_AGE, STD_AGE), MIN_AGE, MAX_AGE))
    today = datetime.today().date()
    birth_date = today - timedelta(days=age * 365 + random.randint(0, 364))
    return birth_date

def generate_registration_datetime(year, month):
    """Return a random registration datetime within business hours for a given month/year."""
    day = random.randint(1, 28)
    hour = random.randint(8, 22)
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return datetime(year, month, day, hour, minute, second)

def generate_postal_code(faker_instance):
    """Generate a postal code using a locale-aware Faker instance."""
    return faker_instance.postcode()

def clean_address(address):
    """Remove commas from address field (replace with space)."""
    if address is None:
        return None
    return address.replace(',', ' ')

def main():
    # Load Phase 1 output
    df = pd.read_csv('/content/drive/MyDrive/etrade/outputs/monthly_new_customers_df.csv')

    month_map = {
        1: 'jan_new_customers', 2: 'feb_new_customers', 3: 'mar_new_customers', 4: 'apr_new_customers',
        5: 'may_new_customers', 6: 'jun_new_customers', 7: 'jul_new_customers', 8: 'aug_new_customers',
        9: 'sep_new_customers', 10: 'oct_new_customers', 11: 'nov_new_customers', 12: 'dec_new_customers'
    }

    customers = []
    addresses = []
    global_customer_id = 1

    for _, row in df.iterrows():
        country = row['country']
        country_code = row['country_code']
        province = row['province_name']
        province_code = f"{country_code}_{row['province_id']}"
        city = row['city']
        year = int(row['year'])

        faker = Faker(get_safe_locale(country_code))

        city_allocated_customers = int(row['city_allocated_customers'])
        monthly_new_customers = {month: int(row[month_map[month]]) for month in range(1, 13)}

        # Validation: monthly sum within 40 of allocated customers
        tolerance = 10
        total_new = sum(monthly_new_customers.values())
        difference = abs(total_new - city_allocated_customers)
        assert difference <= tolerance, (
            f"Mismatch in {city}, {country}: Expected {city_allocated_customers}, got {total_new}. "
            f"Difference of {difference} exceeds allowed tolerance of {tolerance}."
        )

        for month, count in monthly_new_customers.items():
            for _ in range(count):
                gender = random.choice(['Male', 'Female'])
                profile = faker.simple_profile(sex='M' if gender == 'Male' else 'F')
                first_name = faker.first_name_male() if gender == "Male" else faker.first_name_female()
                last_name = faker.last_name()
                registration_datetime = generate_registration_datetime(year, month)
                address_raw = faker.address().replace('\n', ' ')
                address_clean = clean_address(address_raw)

                # Store customer core fields
                customers.append({
                    "customer_id": global_customer_id,
                    "username": to_ascii(profile['username']),
                    "email": to_ascii(profile['mail']),
                    "first_name": to_ascii(first_name),
                    "last_name": to_ascii(last_name),
                    "gender": gender,
                    "birth_date": generate_gaussian_birthdate(),
                    "phone_number": to_ascii(faker.phone_number()),
                    "registration_date": registration_datetime.date(),
                    "registration_datetime": registration_datetime,
                    "is_loyalty_member": random.choices([True, False], weights=[LOYALTY_MEMBER_PROB, 1 - LOYALTY_MEMBER_PROB])[0],
                    "is_fraud_suspected": random.choices([True, False], weights=[FRAUD_SUSPECT_PROB, 1 - FRAUD_SUSPECT_PROB])[0],
                    "customer_segment": random.choices(CUSTOMER_SEGMENTS, weights=SEGMENT_PROBS)[0]
                })

                # Store address fields
                addresses.append({
                    "customer_id": global_customer_id,
                    "country": to_ascii(country),
                    "country_code": country_code,
                    "province": to_ascii(province),
                    "province_code": province_code,
                    "city": to_ascii(city),
                    "district": to_ascii(faker.street_name()),
                    "postal_code": to_ascii(generate_postal_code(faker)),
                    "address": to_ascii(address_clean)
                })

                global_customer_id += 1

    customers_df = pd.DataFrame(customers)
    addresses_df = pd.DataFrame(addresses)
  
    addresses_df.insert(0, 'address_id', addresses_df.index + 1)

    # Validation: check uniqueness and nulls
    assert customers_df['customer_id'].is_unique, "Customer IDs are not unique."
    assert customers_df.notnull().all().all(), "Null values found in customer DataFrame."
    assert addresses_df['address_id'].is_unique, "Address IDs are not unique."
    assert addresses_df['customer_id'].is_unique, "Customer to address mapping is not 1:1."
    assert addresses_df.notnull().all().all(), "Null values found in address DataFrame."

    # Save
    customers_columns = ['customer_id', 'username', 'email', 'first_name', 'last_name', 'gender',
                        'birth_date', 'phone_number', 'registration_date', 'registration_datetime',
                        'is_loyalty_member', 'is_fraud_suspected', 'customer_segment']
    addresses_columns = ['address_id', 'customer_id', 'country', 'country_code', 'province',
                        'province_code', 'city', 'district', 'postal_code', 'address']

    customers_df.to_csv("customers.csv", columns=customers_columns, index=False)
    addresses_df.to_csv("customer_addresses.csv", columns=addresses_columns, index=False)

    print(f"Generated {len(customers_df):,} customers and {len(addresses_df):,} addresses.")
    print("Written to customers.csv and customer_addresses.csv")

if __name__ == "__main__":
    main()