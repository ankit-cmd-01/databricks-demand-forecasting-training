"""Stage 1 entrypoint: Bronze -> Silver -> Gold -> Features -> Feature Store.

Run as a Databricks job task (spark_python_task). This is the only file in
the stage that knows about config.yml; the classes get plain values.
"""
import sys
from pathlib import Path

try:
    THIS_FILE = Path(__file__).resolve()
except NameError:  # serverless runs the script with exec(), so __file__ is missing
    THIS_FILE = Path(sys._getframe().f_code.co_filename).resolve()

STAGE_DIR = THIS_FILE.parent
SRC_DIR = STAGE_DIR.parent
for p in (str(STAGE_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common.config_loader import load_config, resolve_project_path, table_name
from common.spark_session import get_spark

from bronze import BronzeIngestion
from silver import SilverTransformation
from gold import GoldAggregation
from gold_feature import FeatureEngineering
from feature_store import FeatureStoreRegistration


def main():
    config = load_config()
    spark = get_spark()

    bronze_table = table_name(config, "bronze", "bronze_raw")
    silver_table = table_name(config, "silver", "silver_clean")
    failed_table = table_name(config, "config", "dq_failed")
    gold_table = table_name(config, "gold", "gold_daily")
    features_table = table_name(config, "gold", "gold_features")
    feature_store_table = table_name(config, "gold", "feature_store")

    print("\n##### Step 1: Bronze #####")
    BronzeIngestion(
        spark=spark,
        source_path=config["paths"]["source_csv"],
        target_table=bronze_table,
    ).run()

    print("\n##### Step 2: Silver #####")
    SilverTransformation(
        spark=spark,
        bronze_table=bronze_table,
        silver_table=silver_table,
        failed_table=failed_table,
        dq_rules_path=resolve_project_path(config["paths"]["dq_rules"]),
    ).run()

    print("\n##### Step 3: Gold aggregation #####")
    GoldAggregation(
        spark=spark,
        silver_table=silver_table,
        gold_table=gold_table,
    ).run()

    print("\n##### Step 4: Feature engineering #####")
    FeatureEngineering(
        spark=spark,
        gold_table=gold_table,
        features_table=features_table,
    ).run()

    print("\n##### Step 5: Feature Store registration #####")
    FeatureStoreRegistration(
        spark=spark,
        features_table=features_table,
        feature_store_table=feature_store_table,
        primary_keys=config["feature_store"]["primary_keys"],
        description=" ".join(config["feature_store"]["description"].split()),
    ).run()


if __name__ == "__main__":
    main()