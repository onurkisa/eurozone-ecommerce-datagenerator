"""
Script: shipments.py

Purpose:
    Generate synthetic shipment records for each eligible order, assigning realistic shipment companies, shipment types, dates, and status fields.
Inputs:
    - orders_part_*.csv
    - shipment_company.csv
    - customers.csv
Outputs:
    - output/Shipment.csv 
Usage:
    python shipments.py
Business Rules:
    - Only non-cancelled, completed orders are eligible for shipment.
    - Shipment company and type are assigned based on country, available delivery types, and customer segment.
    - Shipment and delivery dates are realistic and calculated from the order date.
    - Status fields (“Delivered”, “In Transit”, “Returned”, etc.) follow business-driven probabilities.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from collections import defaultdict
from tqdm import tqdm
import logging
from typing import Tuple, Optional, List, Dict
import warnings
from pathlib import Path

# Set random seeds for reproducibility 
random.seed(42)
np.random.seed(42)
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BATCH_SIZE = 300000
DEFAULT_EXPRESS_PROBABILITY = 0.15


class OptimizedShipmentDataGenerator:
    """Optimized Shipment Data Generator with progress tracking and error handling."""

    def __init__(self, base_path: str = "/content/drive/MyDrive/Colab Notebooks/new"):
        self.base_path = Path(base_path)
        self.output_dir = self.base_path / "output"

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data containers
        self.orders_df = None
        self.shipment_companies_df = None
        self.customers_df = None
        self.shipment_data = []
        self.shipment_id_counter = 1

        # Country to companies mapping for performance
        self.country_to_companies = defaultdict(list)

        # Status distributions for non-returned orders
        self.shipment_status_weights = {
            "Delivered": 0.88,
            "In Transit": 0.06,
            "Processing": 0.04,
            "Lost": 0.02,
        }

        self.delivery_status_weights = {
            "Successful": 0.90,
            "Delayed": 0.08,
            "Failed": 0.02,
        }

        # Customer segment to express shipping probability
        self.express_probability = {"VIP": 0.65, "Returning": 0.25, "New": 0.12}

    def load_data(self) -> bool:
        logger.info("Starting data loading process...")

        loaded_data = {}

        # Load orders parts
        orders_files = [self.base_path / f"orders_part_{i}.csv" for i in range(1, 12)]
        orders_dfs = []
        for file in tqdm(orders_files, desc="Loading Orders", unit="file"):
            if file.exists():
                try:
                    df = pd.read_csv(file)
                    orders_dfs.append(df)
                    logger.info(f"Loaded {file.name}: {len(df):,} records")
                except Exception as e:
                    logger.error(f"Error loading {file.name}: {e}")
                    return False
            else:
                logger.error(f"File not found: {file.name}")
                return False
        self.orders_df = pd.concat(orders_dfs, ignore_index=True)

        # Load shipment companies
        shipment_companies_file = self.base_path / "shipment_company.csv"
        if shipment_companies_file.exists():
            try:
                self.shipment_companies_df = pd.read_csv(shipment_companies_file)
                logger.info(
                    f"Loaded {shipment_companies_file.name}: {len(self.shipment_companies_df):,} records"
                )
            except Exception as e:
                logger.error(f"Error loading {shipment_companies_file.name}: {e}")
                return False
        else:
            logger.error(f"File not found: {shipment_companies_file.name}")
            return False

        # Load customers
        customers_file = self.base_path / "customers.csv"
        if customers_file.exists():
            try:
                self.customers_df = pd.read_csv(customers_file)
                logger.info(
                    f"Loaded {customers_file.name}: {len(self.customers_df):,} records"
                )
            except Exception as e:
                logger.error(f"Error loading {customers_file.name}: {e}")
                return False
        else:
            logger.error(f"File not found: {customers_file.name}")
            return False

        # Convert date columns and validate
        self._prepare_data()
        self._validate_reference_data()

        return True

    def _prepare_data(self) -> None:
        """Prepare and preprocess loaded data."""
        logger.info("Preparing data...")

        # Convert date columns
        self.orders_df["order_date"] = pd.to_datetime(self.orders_df["order_date"])

        # Prepare shipment company mapping for performance
        self._prepare_shipment_company_mapping()

        logger.info("Data preparation completed")

    def _prepare_shipment_company_mapping(self) -> None:
        """Create optimized mapping between countries and available shipment companies."""
        logger.info("Preparing shipment company mappings...")

        self.country_to_companies = defaultdict(list)

        with tqdm(
            self.shipment_companies_df.iterrows(),
            total=len(self.shipment_companies_df),
            desc="Mapping companies",
            unit="company",
        ) as pbar:

            for _, company in pbar:
                # Parse operating countries (assuming comma-separated)
                if pd.notna(company["operating_countries"]):
                    countries = [
                        c.strip()
                        for c in str(company["operating_countries"]).split(",")
                    ]
                    for country in countries:
                        # Parse delivery types
                        delivery_types = []
                        if pd.notna(company["delivery_types"]):
                            delivery_types = [
                                dt.strip()
                                for dt in str(company["delivery_types"]).split(",")
                            ]

                        self.country_to_companies[country].append(
                            {
                                "company_id": company["shipment_company_id"],
                                "delivery_types": delivery_types,
                                "company_name": company["company_name"],
                            }
                        )

        logger.info(
            f"Mapped shipment companies to {len(self.country_to_companies)} countries"
        )

    def _validate_reference_data(self) -> None:
        """Validate reference data integrity with detailed reporting."""
        logger.info("Validating reference data...")

        # Check for missing shipping_address_id in orders
        missing_addresses = self.orders_df["shipping_address_id"].isna().sum()
        if missing_addresses > 0:
            logger.warning(
                f"{missing_addresses} orders have missing shipping_address_id"
            )


        # Check data relationships
        order_customers = set(self.orders_df["customer_id"].dropna())
        available_customers = set(self.customers_df["customer_id"])
        missing_customers = order_customers - available_customers

        if missing_customers:
            logger.warning(
                f"{len(missing_customers)} orders reference missing customers"
            )

        logger.info("Data validation completed")

    def get_eligible_orders(self) -> pd.DataFrame:
        """Filter orders eligible for shipment with logging."""
        logger.info("Filtering eligible orders...")

        # Only non-cancelled, completed orders
        eligible_orders = self.orders_df[
            (self.orders_df["is_cancelled"] == False)
            & (self.orders_df["order_status"] == "Completed")
        ].copy()

        logger.info(f"Found {len(eligible_orders):,} eligible orders for shipment")
        return eligible_orders

    def get_customer_segment(self, customer_id: int) -> str:
        """Get customer segment for shipment type decision with caching."""
        customer = self.customers_df[self.customers_df["customer_id"] == customer_id]
        if not customer.empty:
            return customer.iloc[0]["customer_segment"]
        return "New"  # Default

    def select_shipment_company_and_type(
        self, country_code: str, customer_segment: str
    ) -> Tuple[int, str]:
        """Select appropriate shipment company and determine shipment type."""
        available_companies = self.country_to_companies.get(country_code, [])

        # Fallback logic for missing country mappings
        if not available_companies:
            # Use all companies as fallback
            for _, company in self.shipment_companies_df.iterrows():
                delivery_types = []
                if pd.notna(company["delivery_types"]):
                    delivery_types = [
                        dt.strip() for dt in str(company["delivery_types"]).split(",")
                    ]

                available_companies.append(
                    {
                        "company_id": company["shipment_company_id"],
                        "delivery_types": delivery_types,
                        "company_name": company["company_name"],
                    }
                )

        # Determine preferred shipment type based on customer segment
        express_prob = self.express_probability.get(
            customer_segment, DEFAULT_EXPRESS_PROBABILITY
        )
        preferred_type = "Express" if random.random() < express_prob else "Standard"

        # Filter companies that support the preferred type
        compatible_companies = []
        for company in available_companies:
            delivery_types = company["delivery_types"]
            if (
                not delivery_types
            ):  # If no delivery types specified, assume all types supported
                compatible_companies.append(company)
            elif preferred_type in delivery_types or any(
                preferred_type.lower() in dt.lower() for dt in delivery_types
            ):
                compatible_companies.append(company)

        # Select company and finalize type
        if not compatible_companies:
            selected_company = random.choice(available_companies)
            actual_type = (
                random.choice(selected_company["delivery_types"])
                if selected_company["delivery_types"]
                else preferred_type
            )
        else:
            selected_company = random.choice(compatible_companies)
            actual_type = preferred_type

        return selected_company["company_id"], actual_type

    def calculate_shipment_dates(
        self, order_date: datetime, shipment_type: str
    ) -> Tuple[datetime, datetime]:
        """Calculate realistic shipment and delivery dates."""
        # Processing time: 1-4 days (more realistic)
        processing_days = random.randint(1, 4)

        # Calculate shipment date
        shipment_date = order_date + timedelta(days=processing_days)

        # Skip weekends for shipment (business days only)
        while shipment_date.weekday() >= 5:  # Saturday=5, Sunday=6
            shipment_date += timedelta(days=1)

        # Delivery duration based on type
        if shipment_type.lower() in ["express", "expedited", "priority"]:
            delivery_days = random.randint(1, 3)
        else:  # Standard or other
            delivery_days = random.randint(2, 7)

        delivery_date = shipment_date + timedelta(days=delivery_days)

        return shipment_date, delivery_date

    def determine_shipment_status(
        self, delivery_date: datetime, is_returned: bool
    ) -> str:
        """Determine shipment status based on delivery date and return status."""
        from datetime import datetime
        # For returned orders: they must have been delivered first, then returned
        if is_returned:
            return "Returned" if random.random() < 0.95 else "Delivered"

        # If delivery date hasn't passed, determine in-progress status
        now = datetime.today()
        if delivery_date > now:
            days_until_delivery = (delivery_date - now).days
            return "In Transit" if days_until_delivery <= 2 else "Processing"

        # For completed deliveries (non-returned)
        return random.choices(
            list(self.shipment_status_weights.keys()),
            weights=list(self.shipment_status_weights.values()),
        )[0]

    def determine_delivery_status(self, shipment_status: str, is_returned: bool) -> str:
        """Determine delivery status based on shipment status and return status."""
        if shipment_status == "Returned":
            return "Returned"
        elif shipment_status == "Delivered":
            return random.choices(
                list(self.delivery_status_weights.keys()),
                weights=list(self.delivery_status_weights.values()),
            )[0]
        elif shipment_status in ["In Transit", "Processing"]:
            return "Pending"
        else:  # Lost
            return "Failed"

    def _process_order_batch(
        self, batch_orders: pd.DataFrame, batch_start_idx: int
    ) -> List[Dict]:
        """Process a batch of orders for shipment generation."""
        batch_shipments = []

        for idx, (_, order) in enumerate(batch_orders.iterrows()):
            try:
                # Get customer segment
                customer_segment = self.get_customer_segment(order["customer_id"])

                # Select shipment company and determine type
                (
                    shipment_company_id,
                    shipment_type,
                ) = self.select_shipment_company_and_type(
                    order["country_code"], customer_segment
                )

                # Calculate dates with validation
                shipment_date, delivery_date = self.calculate_shipment_dates(
                    order["order_date"], shipment_type
                )

                # Validate date logic
                if shipment_date < order["order_date"]:
                    shipment_date = order["order_date"] + timedelta(days=1)
                    delivery_date = shipment_date + timedelta(days=random.randint(1, 5))

                # Determine status based on order return status
                shipment_status = self.determine_shipment_status(
                    delivery_date, order["is_returned"]
                )
                delivery_status = self.determine_delivery_status(
                    shipment_status, order["is_returned"]
                )

                # Create shipment record
                shipment_record = {
                    "shipment_id": f"SHP{(batch_start_idx + idx + 1):08d}",
                    "order_id": order["order_id"],
                    "shipment_company_id": shipment_company_id,
                    "shipping_address_id": order[
                        "customer_id"
                    ],  # 1:1 mapping as per requirement
                    "shipment_date": shipment_date.strftime("%Y-%m-%d"),
                    "delivery_date": delivery_date.strftime("%Y-%m-%d"),
                    "shipment_type": shipment_type,
                    "shipment_status": shipment_status,
                    "delivery_status": delivery_status,
                }

                batch_shipments.append(shipment_record)

            except Exception as e:
                logger.warning(
                    f"Error processing order {order.get('order_id', 'unknown')}: {e}"
                )
                continue

        return batch_shipments

    def generate_shipment_data(self) -> Optional[pd.DataFrame]:
        """Generate shipment data for all eligible orders with optimized batch processing."""
        logger.info("Starting shipment data generation...")

        if self.orders_df is None:
            if not self.load_data():
                return None

        eligible_orders = self.get_eligible_orders()

        if eligible_orders.empty:
            logger.warning("No eligible orders found!")
            return None

        all_shipments = []

        # Process in batches for better memory management
        total_batches = (len(eligible_orders) + BATCH_SIZE - 1) // BATCH_SIZE

        logger.info(
            f"Processing {len(eligible_orders):,} orders in {total_batches} batches..."
        )

        with tqdm(
            total=len(eligible_orders), desc="Generating Shipments", unit="order"
        ) as pbar:
            for batch_num in range(total_batches):
                start_idx = batch_num * BATCH_SIZE
                end_idx = min(start_idx + BATCH_SIZE, len(eligible_orders))

                batch_orders = eligible_orders.iloc[start_idx:end_idx]
                batch_shipments = self._process_order_batch(
                    batch_orders, len(all_shipments)
                )

                all_shipments.extend(batch_shipments)

                pbar.update(len(batch_orders))
                pbar.set_postfix(
                    {
                        "Shipments": len(all_shipments),
                        "Batch": f"{batch_num + 1}/{total_batches}",
                    }
                )

        if not all_shipments:
            logger.warning("No shipments generated!")
            return None

        shipment_df = pd.DataFrame(all_shipments)

        self._validate_generated_data(shipment_df)
        self._save_results(shipment_df)

        logger.info("Shipment data generation completed successfully!")
        logger.info(f"Generated {len(shipment_df):,} shipment records")

        return shipment_df

    def _validate_generated_data(self, df: pd.DataFrame) -> None:
        """Validate generated shipment data with comprehensive checks."""
        logger.info("Validating generated shipment data...")

        # Check for duplicates
        duplicates = df["shipment_id"].duplicated().sum()
        logger.info(
            f"Unique shipment IDs: {duplicates == 0} (duplicates: {duplicates})"
        )

        # Check date consistency
        df_temp = df.copy()
        df_temp["shipment_date"] = pd.to_datetime(df_temp["shipment_date"])
        df_temp["delivery_date"] = pd.to_datetime(df_temp["delivery_date"])

        # Validate with order dates
        validation_df = df_temp.merge(
            self.orders_df[["order_id", "order_date"]], on="order_id", how="left"
        )

        # Check shipment date >= order date
        early_shipments = (
            validation_df["shipment_date"] < validation_df["order_date"]
        ).sum()
        logger.info(
            f"Shipment after order: {early_shipments == 0} (early shipments: {early_shipments})"
        )

        # Check delivery date >= shipment date
        date_inconsistencies = (
            df_temp["delivery_date"] < df_temp["shipment_date"]
        ).sum()
        logger.info(
            f"Delivery after shipment: {date_inconsistencies == 0} (inconsistencies: {date_inconsistencies})"
        )

        # Status distribution logging
        logger.info("Status Distribution:")
        logger.info("Shipment Status:")
        for status, count in df["shipment_status"].value_counts().items():
            logger.info(f"{status}: {count:,} ({count/len(df):.1%})")

        logger.info("Delivery Status:")
        for status, count in df["delivery_status"].value_counts().items():
            logger.info(f"  {status}: {count:,} ({count/len(df):.1%})")

    def _save_results(self, df: pd.DataFrame) -> None:
        """Save DataFrame to CSV with error handling."""
        try:
            output_file = self.output_dir / "Shipment.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"Saved {len(df):,} records to {output_file}")

            # Log final schema
            logger.info("Final Schema:")
            for col in df.columns:
                logger.info(f"  • {col}: {df[col].dtype}")

        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise

    def get_summary_stats(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics for the generated shipments."""
        if df is None or df.empty:
            return {}

        eligible_orders = (
            len(self.get_eligible_orders()) if self.orders_df is not None else 0
        )
        total_orders = len(self.orders_df) if self.orders_df is not None else 0

        return {
            "total_orders": total_orders,
            "eligible_orders": eligible_orders,
            "generated_shipments": len(df),
            "success_rate": f"{(len(df)/eligible_orders)*100:.1f}%"
            if eligible_orders > 0
            else "0%",
            "countries_covered": len(df["shipment_company_id"].unique()),
            "shipment_companies_used": len(df["shipment_company_id"].unique()),
        }


def main():
    """Main function to run the shipment generation process."""
    print("StartingShipment Data Generation...")
    print("=" * 60)

    # Initialize generator
    generator = OptimizedShipmentDataGenerator()

    try:
        # Generate shipments
        shipment_df = generator.generate_shipment_data()

        if shipment_df is not None:
            print("\n" + "=" * 60)
            print("GENERATION SUMMARY")
            print("=" * 60)

            # Display summary statistics
            stats = generator.get_summary_stats(shipment_df)
            for key, value in stats.items():
                print(f"{key.replace('_', ' ').title()}: {value}")

            print(f"\nShipment Summary:")
            print(f"Total Shipments: {len(shipment_df):,}")
            print(f"Date Range: {shipment_df['shipment_date'].min()} to {shipment_df['delivery_date'].max()}"
            )

            print("\nSample Shipment Data:")
            print(shipment_df.head())

            print("\nTop Shipment Companies:")
            company_counts = shipment_df["shipment_company_id"].value_counts().head()
            for company_id, count in company_counts.items():
                print(f"Company {company_id}: {count:,} shipments")

        else:
            print("Shipment generation failed. Please check the logs for details.")

    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
