import json

from pyspark.sql import functions as F
from pyspark.sql.window import Window


class SilverTransformation:
    """Applies config-driven DQ rules to Bronze. Passed rows go to Silver, failed rows go to the config schema."""

    def __init__(self, spark, bronze_table, silver_table, failed_table, dq_rules_path):
        self.spark = spark
        self.bronze_table = bronze_table
        self.silver_table = silver_table
        self.failed_table = failed_table
        self.dq_rules_path = dq_rules_path
        self.dq_rules = {}
        self.checks = []        # list of (check_name, pass_condition)
        self.violations = {}    # check_name -> number of failing rows

    def load_rules(self):
        """Read DQ rules from the JSON file."""
        with open(self.dq_rules_path, "r") as f:
            self.dq_rules = json.load(f)
        print(json.dumps(self.dq_rules, indent=2))

    def read_bronze(self):
        """Load Bronze and drop technical and zero-value columns."""
        df_bronze = self.spark.table(self.bronze_table)
        df = df_bronze.drop("_bronze_ingestion_timestamp", "_source_file")
        df = df.drop("source_system", "operation_type")
        print("Bronze row count:", df_bronze.count())
        print("Silver row count before DQ:", df.count())
        return df

    def add_row_number(self, df):
        """Number rows per transaction_id so duplicates can be detected.

        Ordering by a hash of the whole row means the same duplicate is always
        the one kept. The number of failures is the same as in the notebook.
        """
        w = Window.partitionBy("transaction_id").orderBy(F.xxhash64(*df.columns))
        return df.withColumn("_dq_row_number", F.row_number().over(w))

    def _add_check(self, df, name, condition):
        """Register a check and count how many rows fail it."""
        self.checks.append((name, condition))
        self.violations[name] = df.filter(~condition).count()

    def apply_column_rules(self, df):
        """Null, unique and min checks from column_rules."""
        for col_name, rules in self.dq_rules["column_rules"].items():
            if rules.get("nullable") is False:
                self._add_check(df, f"{col_name}_null_check", F.col(col_name).isNotNull())
            if rules.get("unique") is True:
                # first occurrence passes, repeats fail
                self._add_check(df, f"{col_name}_unique_check", F.col("_dq_row_number") == 1)
            if "min" in rules:
                self._add_check(df, f"{col_name}_min_check", F.col(col_name) >= rules["min"])

    def apply_business_rules(self, df):
        """Business rule checks; the condition is read straight from the rules file."""
        for rule in self.dq_rules["business_rules"]:
            self._add_check(df, rule["name"], F.expr(rule["condition"]))

    def tag_failures(self, df):
        """Add the list of failed check names per row and a failed flag."""
        reasons = [F.when(~cond, name) for name, cond in self.checks]
        df = df.withColumn("_dq_failure_reasons", F.array_compact(F.array(*reasons)))
        return df.withColumn("_dq_failed", F.size("_dq_failure_reasons") > 0)

    def print_report(self, df):
        print("=== DQ VIOLATION REPORT ===")
        for check, count in self.violations.items():
            print(f"{check}: {count} violation(s)")

        failed = df.filter(F.col("_dq_failed")).count()
        passed = df.filter(~F.col("_dq_failed")).count()
        print("\n=== DQ SUMMARY ===")
        print("Failed rows:", failed)
        print("Passed rows:", passed)
        print("Total rows:", df.count())

    def write_failed(self, df):
        """Failed rows go to the config schema with their failure reasons."""
        (
            df.filter(F.col("_dq_failed"))
            .drop("_dq_failed", "_dq_row_number")
            .withColumn("_dq_failure_timestamp", F.current_timestamp())
            .write.format("delta").mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(self.failed_table)
        )

    def write_clean(self, df):
        """Passed rows go to Silver."""
        (
            df.filter(~F.col("_dq_failed"))
            .drop("_dq_failed", "_dq_row_number", "_dq_failure_reasons")
            .withColumn("_silver_processed_timestamp", F.current_timestamp())
            .write.format("delta").mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(self.silver_table)
        )

    def run(self):
        self.load_rules()
        df = self.read_bronze()
        df = self.add_row_number(df)
        self.apply_column_rules(df)
        self.apply_business_rules(df)
        df = self.tag_failures(df)
        self.print_report(df)
        self.write_failed(df)
        self.write_clean(df)