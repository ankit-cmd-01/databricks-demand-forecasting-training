from pyspark.sql import functions as F
from pyspark.sql.window import Window


class FeatureEngineering:
    """Builds calendar, lag and rolling features from the Gold daily table."""

    def __init__(self, spark, gold_table, features_table):
        self.spark = spark
        self.gold_table = gold_table
        self.features_table = features_table
        self.final_columns = [
            "transaction_date", "product_name", "destination_city", "total_demand",
            "avg_unit_price", "total_inventory", "day_of_week", "month",
            "demand_lag_1", "demand_lag_7", "rolling_avg_7",
        ]

    def read_gold(self):
        return self.spark.table(self.gold_table)

    def build_features(self, df):
        """Rename to feature names, add calendar, lag and rolling features."""
        w = Window.partitionBy("product_name", "destination_city").orderBy("transaction_date")

        return (
            df
            .withColumnRenamed("demand_quantity", "total_demand")
            .withColumnRenamed("avg_unit_price_usd", "avg_unit_price")
            .withColumnRenamed("available_inventory", "total_inventory")
            .withColumn("day_of_week", F.dayofweek("transaction_date"))
            .withColumn("month", F.month("transaction_date"))
            .withColumn("demand_lag_1", F.lag("total_demand", 1).over(w))
            .withColumn("demand_lag_7", F.lag("total_demand", 7).over(w))
            # window of the 7 previous rows, excluding the current row (no leakage)
            .withColumn("rolling_avg_7", F.avg("total_demand").over(w.rowsBetween(-7, -1)))
            # early rows have no history yet, so drop them
            .dropna(subset=["demand_lag_1", "demand_lag_7", "rolling_avg_7"])
            .select(*self.final_columns)
        )

    def write_features_table(self, df):
        """Plain Gold table with the final 11 columns."""
        (
            df.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(self.features_table)
        )

    def run(self):
        df = self.build_features(self.read_gold())
        print("Final row count:", df.count())
        print("Final column count:", len(df.columns))
        df.printSchema()
        self.write_features_table(df)