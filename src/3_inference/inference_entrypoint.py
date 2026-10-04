"""Stage 3 entrypoint: score the validation set with the latest registered model."""
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

from model_predictions import ModelPredictions


def main():
    config = load_config()
    spark = get_spark()

    print("\n##### Predictions on the validation set #####")
    ModelPredictions(
        spark=spark,
        model_name=model_name(config),
        validation_table=table_name(config, "ml", "validation"),
        predictions_table=table_name(config, "ml", "predictions_validation"),
    ).run()


if __name__ == "__main__":
    main()