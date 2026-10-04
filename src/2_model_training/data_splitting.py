from pyspark.sql import functions as F


class DataSplitting:
    """Chronological splits: feature table -> train pool + inference holdout -> train + validation."""

    def __init__(self, spark, feature_store_table, train_pool_table, holdout_table,
                 train_table, validation_table, split_fraction=0.8):
        self.spark = spark
        self.split_fraction = split_fraction
        self.feature_store_table = feature_store_table
        self.train_pool_table = train_pool_table
        self.holdout_table = holdout_table
        self.train_table = train_table
        self.validation_table = validation_table

    def find_cutoff_date(self, df):
        """Date at the split_fraction point of transaction_date (unix_date -> number, then back to a date)."""
        return (
            df
            .selectExpr(f"percentile_approx(unix_date(transaction_date), {self.split_fraction}) as cutoff")
            .withColumn("cutoff_date", F.expr("date_add('1970-01-01', cast(cutoff as int))"))
            .collect()[0]["cutoff_date"]
        )

    def split_by_date(self, df):
        """Chronological split: earlier share of dates vs later share."""
        cutoff = self.find_cutoff_date(df)
        earlier = df.filter(F.col("transaction_date") <= cutoff)
        later = df.filter(F.col("transaction_date") > cutoff)
        return earlier, later, cutoff

    def save(self, df, table_name):
        (
            df.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(table_name)
        )

    def run(self):
        # Split 1: feature table -> train pool (80%) + inference holdout (20%)
        df_fs = self.spark.table(self.feature_store_table)
        df_train_pool, df_holdout, cutoff = self.split_by_date(df_fs)

        print("Cutoff date:", cutoff)
        print("Total rows:", df_fs.count())
        print("Train pool:", df_train_pool.count())
        print("Inference holdout:", df_holdout.count())

        self.save(df_train_pool, self.train_pool_table)
        self.save(df_holdout, self.holdout_table)

        # Split 2: train pool -> training (80%) + validation (20%)
        df_train_pool = self.spark.table(self.train_pool_table)
        df_train, df_validation, cutoff_2 = self.split_by_date(df_train_pool)

        print("\nTrain/Validation cutoff:", cutoff_2)
        print("Train pool:", df_train_pool.count())
        print("Training:", df_train.count())
        print("Validation:", df_validation.count())

        self.save(df_train, self.train_table)
        self.save(df_validation, self.validation_table)