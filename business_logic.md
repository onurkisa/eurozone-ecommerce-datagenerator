## Business Logic Documentation

This document details the business logic, simulation algorithms, and data generation methodology for each script in the Eurozone E-commerce Synthetic Data Generator.

### Table of Contents

1. [customers.py](#customerspy)
2. [products_and_prices.py](#products_and_pricespy)
3. [shipment_company.py](#shipment_companypy)
4. [payment_channel.py](#payment_channelpy)
5. [orders_n_orderdetail.py](#orders_n_orderdetailpy)
6. [shipments.py](#shipmentspy)
7. [invoice_n_invoicedetail.py](#invoice_n_invoicedetailpy)

---

### [`customers.py`](./customers.py)

#### Purpose
Generates realistic customer demographics and address data for Eurozone e-commerce simulation, with customer allocations driven by **[eurozone-ecommerce-modeling](https://github.com/onurkisa/eurozone-ecommerce-modeling)** outputs.

#### Simulation Logic
- **Customer Allocation**: Uses `monthly_new_customers_df.csv` to determine exact customer counts per city/country/month
- **Demographic Generation**: Leverages Faker library for realistic name, email, and personal data generation
- **Address Assignment**: Each customer receives one primary address in their allocated city/country
- **Segmentation Logic**: Customers are classified into behavioral segments based on configurable business rules

#### Key Functions
- **Customer Creation**: Generates synthetic demographic profiles with realistic variation
- **Address Linking**: Maintains 1:1 relationship between customers and primary addresses
- **ASCII Normalization**: Ensures text compatibility using unidecode for international characters
- **Batch Processing**: Memory-efficient generation for large customer volumes

#### Output Relationships
- `customers.csv`: Primary dimension table referenced by orders and shipments
- `customer_addresses.csv`: Address dimension linked to customers via customer_id

---

###  [`products_and_prices.py`](./products_and_prices.py)

#### Purpose
Creates a comprehensive product catalog with Eurozone-wide base pricing and country-specific price adjustments based on economic indicators.

#### Simulation Logic
- **Product Catalog Generation**: Creates diverse product categories with realistic brand and attribute distributions
- **Base Pricing Model**: Calculates unit prices using cost-plus-margin methodology with category-specific margins
- **Economic Price Adjustment**: Applies purchasing power parity (PPS) normalization to create country-specific pricing
- **Price Multiplier Formula**: `blended = 1 + (pps_norm - 0.5) * 2`, clamped between 0.85-1.2 to prevent unrealistic arbitrage

#### Key Functions
- **Category-Based Pricing**: Different margin structures for various product categories
- **Economic Multiplier Calculation**: Uses **[eurozone-ecommerce-modeling](https://github.com/onurkisa/eurozone-ecommerce-modeling)** economic indicator (`eco_sitch.csv`) for local price adjustment
- **Price Clamping**: Prevents extreme price deviations while maintaining economic realism
- **ASCII Text Normalization**: Ensures product names and descriptions are compatibility-safe

#### Output Relationships
- `products.csv`: Master product dimension with base Eurozone pricing
- `product_prices.csv`: Country-specific pricing table (product_id + country_code composite key)

---

### [`shipment_company.py`](./shipment_company.py)

#### Purpose
Generates a realistic roster of shipping and delivery companies operating across Eurozone markets with various service offerings.

#### Simulation Logic
- **Multi-Country Operations**: Companies can operate in multiple countries (comma-separated lists)
- **Service Type Diversity**: Different delivery types (Standard, Express, Same-day, etc.) per company
- **Geographic Coverage**: Realistic assignment of operating countries based on company size and type
- **Intentional 1NF Violation**: Uses comma-separated fields to enable downstream data normalization exercises

#### Key Functions
- **Company Profile Generation**: Creates realistic shipping company profiles with service capabilities
- **Geographic Assignment**: Assigns operating territories based on company characteristics
- **Service Type Allocation**: Different companies offer different delivery service portfolios

#### Output Relationships
- `shipment_company.csv`: Dimension table referenced by shipments for carrier assignment

---

### [`payment_channel.py`](./payment_channel.py)

#### Purpose
Creates a comprehensive list of payment methods and providers available across Eurozone markets, including both global and local payment solutions.

#### Simulation Logic
- **Global vs Local Methods**: Distinguishes between internationally available payment methods and country-specific solutions
- **Payment Type Classification**: Categorizes methods as Card, Wallet, BankTransfer, or Other
- **Country-Specific Providers**: Includes local payment solutions popular in specific Eurozone countries
- **Provider Diversity**: Represents both major global players and regional specialists

#### Key Functions
- **Method Classification**: Systematic categorization of payment types and providers
- **Geographic Assignment**: Assigns payment methods to appropriate country markets
- **Provider Mapping**: Links payment methods to their respective service providers

#### Output Relationships
- `payment_channel.csv`: Dimension table for payment method assignment in order processing

---

### [`orders_n_orderdetail.py`](./orders_n_orderdetail.py)

#### Purpose
Generates the core transactional data (orders and order details) with sophisticated basket behavior modeling based on economic indicators and customer segments, serving as the primary fact table for data warehouse analytics.

#### Simulation Logic
- **Strict Volume Allocation**: Each city/country/month receives exactly the prescribed number of orders from `monthly_orders_df.csv` - no deviation allowed
- **Customer-Geography Mapping**: Orders assigned only to customers with primary addresses in the target city/country, derived from first address in `customer_addresses.csv`
- **Economic-Tiered Basket Behavior**: Three-tiered basket generation via `adjust_basket_behavior()` function based on PPS normalization:
  - **High/Medium-High (pps_norm ≥ 0.5)**: Up to 6 items, premium product preference, higher quantities, greater price tolerance
  - **Lower-Medium (0.3 < pps_norm < 0.5)**: Up to 4 items, moderate premium probability, balanced price/quantity behavior
  - **Lowest (pps_norm ≤ 0.3)**: Up to 2 items, lowest premium probability, highest price sensitivity, minimal quantities
- **Country-Specific Pricing**: All transaction prices sourced from `product_prices.csv` (never global prices from `products.csv`)
- **Sales Amount Calculation**: `sales_amount = (unit_price × quantity) - discount_amount` with extensible discount logic
- **Return/Cancellation Logic**: Cancelled orders automatically marked as returned; completed orders subject to `RETURN_PROBABILITY` with predefined reason sampling
- **Star Schema Design**: order_detail serves as primary sales_fact table with each row representing one product within an order for granular analytics

#### Key Functions
- **`adjust_basket_behavior()`**: Core economic-based parameter adjustment controlling max basket size, premium product probability, price deviation allowance, minimum price ratios, and quantity multipliers
- **Batch Processing Architecture**: Configurable `ORDER_DETAILS_BATCH_SIZE` and `ORDERS_BATCH_SIZE` for memory-constrained environments supporting millions of rows
- **Foreign Key Validation**: All dimension table references (products, shipment companies, addresses) validated for referential integrity
- **Customer-Location Allocation**: Ensures geographic consistency between order location and customer primary address

#### Output Relationships
- `orders.csv`: Order header table with total_price as sum of order_details sales_amounts
- `order_detail.csv`: Primary fact table (1:N with orders) enabling granular product-level sales analysis

---

### [`shipments.py`](./shipments.py)

#### Purpose
Generates realistic shipment records for eligible orders, simulating the logistics and delivery process with appropriate carrier assignment and status tracking.

#### Simulation Logic
- **Eligibility Filtering**: Only non-cancelled, completed orders receive shipment records
- **Carrier Assignment**: Shipment company selection based on customer country and available service types
- **Date Calculations**: Realistic shipment and delivery date generation from order dates
- **Status Simulation**: Business-driven probabilities for shipment statuses (Delivered, In Transit, Returned, etc.)
- **Customer Segment Consideration**: Higher-tier customers may receive premium shipping services

#### Key Functions
- **Order Filtering**: Identifies eligible orders for shipment processing
- **Carrier Matching**: Assigns appropriate shipping companies based on geographic coverage
- **Date Logic**: Calculates realistic shipment timelines considering business days and service types
- **Status Assignment**: Probabilistic status assignment reflecting real-world delivery patterns

#### Output Relationships
- `shipments.csv`: Shipment tracking table linked to orders via order_id

---

### [`invoice_n_invoicedetail.py`](./invoice_n_invoicedetail.py)

#### Purpose
Processes completed orders to generate VAT-compliant invoices and detailed line items, ensuring proper financial record-keeping across Eurozone jurisdictions.

#### Simulation Logic
- **Completion-Based Invoicing**: Only orders with 'Completed' status are eligible for invoicing
- **VAT Rate Application**: Accurate Eurozone VAT rates applied based on customer country codes
- **Business Day Calculations**: Invoice dates generated by adding 1-7 business days to order dates
- **Sequential Numbering**: Zero-padded sequential invoice numbers for audit trail compliance
- **1:1 Order-Invoice Relationship**: One invoice per completed order with detailed line items

#### Key Functions
- **Order Filtering**: Identifies completed orders requiring invoicing
- **VAT Calculation**: Applies country-specific VAT rates to order totals
- **Date Generation**: Business day-aware invoice date calculation
- **Batch Processing**: Memory-efficient processing for large order volumes
- **Number Generation**: Sequential, zero-padded invoice numbering system

#### Output Relationships
- `invoices.csv`: Invoice header table linked to orders
- `invoice_details.csv`: Invoice line items mirroring order_detail structure with VAT calculations

---

### Cross-Script Data Flow

#### **[eurozone-ecommerce-modeling](https://github.com/onurkisa/eurozone-ecommerce-modeling)** Dependencies
All scripts depend on the outputs from the **[eurozone-ecommerce-modeling](https://github.com/onurkisa/eurozone-ecommerce-modeling)** repository:
- `monthly_orders_df.csv`: Order volume targets
- `monthly_new_customers_df.csv`: Customer allocation targets  
- `eco_sitch.csv`: Economic indicators and PPS normalization
- `city_level_allocation.csv`: Geographic allocation parameters

#### Execution Order
1. **Dimension Tables**: customers.py, products_and_prices.py, shipment_company.py, payment_channel.py
2. **Fact Tables**: orders_n_orderdetail.py
3. **Dependent Tables**: shipments.py, invoice_n_invoicedetail.py

