from pyspark.sql import functions as F


class GoldAggregation:
    """Aggregates Silver transactions to one row per product + city + date."""

    def __init__(self, spark, silver_table, gold_table):
        self.spark = spark
        self.silver_table = silver_table
        self.gold_table = gold_table
        self.group_cols = ["product_name", "destination_city", "transaction_date"]

    def read_silver(self):
        return self.spark.table(self.silver_table)

    def aggregate(self, df):
        """Collapse supplier-level rows into the daily grain."""
        return (
            df.groupBy(*self.group_cols)
            .agg(
                # flow quantities and costs: add up across suppliers
                F.sum("demand_quantity").alias("demand_quantity"),
                F.sum("ordered_quantity").alias("ordered_quantity"),
                F.sum("product_cost_usd").alias("product_cost_usd"),
                F.sum("transportation_cost_usd").alias("transportation_cost_usd"),
                F.sum("total_cost_usd").alias("total_cost_usd"),

                # inventory is a stock level, not a flow: summing would double count
                F.max("available_inventory").alias("available_inventory"),

                # price does not add up across suppliers: use the average
                F.avg("unit_price_usd").alias("avg_unit_price_usd"),

                # constant per product: carry one value through
                F.first("product_category").alias("product_category"),
                F.first("quantity_unit").alias("quantity_unit"),

                # how many supplier transactions were combined into this row
                F.count("*").alias("num_transactions"),
            )
        )

    def write_table(self, df):
        (
            df.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(self.gold_table)
        )

    def run(self):
        df_silver = self.read_silver()
        df_gold = self.aggregate(df_silver)
        print("Gold row count:", df_gold.count())
        df_gold.orderBy("transaction_date").show(10)
        self.write_table(df_gold)