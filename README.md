## Eurozone E-commerce Synthetic Data Generator

A comprehensive Python toolkit for generating realistic synthetic e-commerce datasets simulating business operations across Eurozone countries. This project creates interconnected datasets including customers, products, orders, shipments, and invoices with realistic business logic and economic modeling.

### Purpose

This repository generates synthetic e-commerce data that mirrors real-world business patterns across Eurozone markets. The generated datasets are designed for:

- Data warehouse and ETL testing
- E-commerce platform simulation and testing

### Features

- **Realistic Customer Generation**: Synthetic customers with demographic attributes and address data
- **Economic-Based Product Pricing**: Country-specific pricing adjusted for purchasing power parity
- **Multi-Tiered Order Simulation**: Order volumes and basket behaviors driven by economic indicators
- **Complete Transaction Lifecycle**: From order placement through shipment and invoicing
- **Eurozone VAT Compliance**: Accurate VAT calculations for all supported countries
- **Scalable Architecture**: Batch processing support for millions of records
- **Referential Integrity**: All datasets maintain proper foreign key relationships
- **For more features**, check [`business_logic.md`](./business_logic.md) and each script's explanatory header.

### Requirements

- Python 3.7+
- Required packages listed in [`requirements.txt`](./requirements.txt)

### ⚠️ Important Notice

**This repository contains only the data generation scripts.** To run these scripts, you must first generate the required input CSV files using the companion repository:

 **[eurozone-ecommerce-modeling](https://github.com/onurkisa/eurozone-ecommerce-modeling)**

The modeling repository generates the foundational datasets that drive the allocations and business logic in this generation toolkit.

### Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/onurkisa/eurozone-ecommerce-datagenerator.git
   cd eurozone-ecommerce-datagenerator

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare input data:**
   Follow the instructions in the [eurozone-ecommerce-modeling](https://github.com/your-org/eurozone-ecommerce-modeling) repository to generate the required input CSV files.

### Usage

Execute the scripts **in the following sequence** to ensure data consistency and foreign key relationships across datasets:

```bash
# 1. Dimension tables (must be created first)
python customers.py
python products_and_prices.py  
python shipment_company.py
python payment_channel.py

# 2. Fact tables (rely on dimension data)
python orders_n_orderdetail.py

# 3. Dependent tables (shipments and invoices based on order data)
python shipments.py
python invoice_n_invoicedetail.py
```

### Generated Datasets

| Dataset | Description | Key Features |
|---------|-------------|--------------|
| `customers.csv` | Customer master data | Demographics, segmentation, registration data |
| `customer_addresses.csv` | Customer address records | Primary addresses linked to customers |
| `products.csv` | Product catalog | Categories, brands, base pricing |
| `product_prices.csv` | Country-specific pricing | Local prices adjusted for economic indicators |
| `payment_channel.csv` | Payment methods | Global and local payment providers |
| `shipment_company.csv` | Delivery providers | Operating countries and service types |
| `orders.csv` | Order transactions | Order header information |
| `order_detail.csv` | Order line items | Product quantities and pricing details |
| `shipments.csv` | Shipment records | Delivery tracking and status |
| `invoices.csv` | Invoice headers | VAT-compliant invoicing |
| `invoice_details.csv` | Invoice line items | Detailed billing information |

### Configuration

Each script contains configurable parameters at the top of the file:

- **Batch sizes** for memory management
- **Probability settings** for returns, cancellations, and shipment status
- **Economic thresholds** for basket behavior adjustment
- **Date ranges** and business day calculations

### License

This project is licensed under the MIT License - see the LICENSE file for details.

### Related Projects

- [eurozone-ecommerce-modeling](https://github.com/onurkisa/eurozone-ecommerce-modeling) - Input data preparation and economic modeling
