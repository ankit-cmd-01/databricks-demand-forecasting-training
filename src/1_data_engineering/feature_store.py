from databricks.feature_engineering import FeatureEngineeringClient


class FeatureStoreRegistration:
    """Registers the features table as a governed Feature Store table (safe to re-run)."""

    def __init__(self, spark, features_table, feature_store_table, primary_keys, description):
        self.spark = spark
        self.features_table = features_table
        self.feature_store_table = feature_store_table
        self.primary_keys = primary_keys
        self.description = description

    def run(self):
        fe = FeatureEngineeringClient()
        df = self.spark.table(self.features_table)

        # create_table fails if the table exists and write_table only supports merge,
        # so drop the old table first to get a clean overwrite on every run
        self.spark.sql(f"DROP TABLE IF EXISTS {self.feature_store_table}")

        fe.create_table(
            name=self.feature_store_table,
            primary_keys=self.primary_keys,
            df=df,
            description=self.description,
        )
        print(f"Feature table registered: {self.feature_store_table}")