"""Stage 2 entrypoint: chronological data splits, then CatBoost training and registration."""
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

from common.config_loader import load_config, model_name, table_name
from common.spark_session import get_spark

from data_splitting import DataSplitting
from model_training import ModelTraining


def main():
    config = load_config()
    spark = get_spark()

    feature_store_table = table_name(config, "gold", "feature_store")
    train_table = table_name(config, "ml", "train")
    validation_table = table_name(config, "ml", "validation")

    print("\n##### Step 1: Data splitting #####")
    DataSplitting(
        spark=spark,
        feature_store_table=feature_store_table,
        train_pool_table=table_name(config, "ml", "train_pool"),
        holdout_table=table_name(config, "ml", "inference_holdout"),
        train_table=train_table,
        validation_table=validation_table,
        split_fraction=config["split"]["fraction"],
    ).run()

    print("\n##### Step 2: Model training #####")
    model_cfg = config["model"]
    ModelTraining(
        spark=spark,
        train_table=train_table,
        validation_table=validation_table,
        model_name=model_name(config),
        experiment_name=model_cfg["experiment_name"],
        params=model_cfg["params"],
        features=model_cfg["features"],
        cat_features=model_cfg["cat_features"],
        target=model_cfg["target"],
        run_name=model_cfg["run_name"],
    ).run()


if __name__ == "__main__":
    main()