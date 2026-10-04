from pyspark.sql import functions as F


class BronzeIngestion:
    """Reads the raw CSV from the Volume and writes it as a Bronze Delta table."""

    def __init__(self, spark, source_path, target_table):
        self.spark = spark
        self.source_path = source_path
        self.target_table = target_table

    def read_source(self):
        """Read raw CSV as-is, no cleaning in Bronze."""
        df = (
            self.spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .csv(self.source_path)
        )
        print("Row count from source:", df.count())
        df.printSchema()
        return df

    def add_audit_columns(self, df):
        """Add technical lineage columns (not business features)."""
        return (
            df
            .withColumn("_bronze_ingestion_timestamp", F.current_timestamp())
            .withColumn("_source_file", F.col("_metadata.file_path"))
        )

    def write_table(self, df):
        """Write to Bronze as a Delta table."""
        (
            df.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(self.target_table)
        )
        print("Bronze table created:", self.target_table)

    def run(self):
        df = self.read_source()
        df = self.add_audit_columns(df)
        self.write_table(df)
        self.spark.table(self.target_table).limit(10).show(truncate=False)