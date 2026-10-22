from typing import Any

import mlflow


class ModelPredictions:
    """Loads the champion model from Unity Catalog and scores it on the validation set."""

    def __init__(self, spark, model_name, validation_table, predictions_table):
        self.spark = spark
        self.model_name = model_name
        self.validation_table = validation_table
        self.predictions_table = predictions_table
        self.model: Any = None
        self.features = None

    def load_model(self):
        """Point MLflow at Unity Catalog and load the model that holds the alias "champion"."""
        mlflow.set_registry_uri("databricks-uc")
        model_uri = f"models:/{self.model_name}@champion"
        self.model = mlflow.pyfunc.load_model(model_uri)
        print("Loaded champion model:", model_uri)

        # Required feature columns come from the model's own signature, not hardcoded
        self.features = self.model.metadata.get_input_schema().input_names()
        print("Features from model signature:", self.features)

    def predict(self):
        """Load validation data and score it with the loaded model."""
        df_val = self.spark.table(self.validation_table).toPandas()
        df_val["predicted_demand"] = self.model.predict(df_val[self.features])

        return df_val[
            ["transaction_date", "product_name", "destination_city", "total_demand", "predicted_demand"]
        ]

    def save(self, df_result):
        (
            self.spark.createDataFrame(df_result).write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(self.predictions_table)
        )
        print("Predictions written to:", self.predictions_table)

    def run(self):
        self.load_model()
        df_result = self.predict()
        self.save(df_result)
        self.spark.table(self.predictions_table).limit(10).show()