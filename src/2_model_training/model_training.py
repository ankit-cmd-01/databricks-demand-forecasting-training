import mlflow
import mlflow.catboost
import numpy as np
from catboost import CatBoostRegressor
from mlflow.models.signature import infer_signature
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ModelTraining:
    """Trains a CatBoost model on the train split, logs it to MLflow and registers it in Unity Catalog."""

    def __init__(self, spark, train_table, validation_table, model_name, experiment_name,
                 params, features, cat_features, target, run_name="catboost_baseline"):
        self.spark = spark
        self.train_table = train_table
        self.validation_table = validation_table
        self.model_name = model_name
        self.experiment_name = experiment_name
        self.params = params
        self.features = features
        self.cat_features = cat_features
        self.target = target
        self.run_name = run_name
        self.run_id = None
        self.model_uri = None

    def load_data(self):
        """Load train and validation splits as pandas and separate features from target."""
        df_train = self.spark.table(self.train_table).toPandas()
        df_val = self.spark.table(self.validation_table).toPandas()

        self.X_train, self.y_train = df_train[self.features], df_train[self.target]
        self.X_val, self.y_val = df_val[self.features], df_val[self.target]

        print("Train:", df_train.shape)
        print("Validation:", df_val.shape)
        print("Features:", self.features)

    def setup_mlflow(self):
        """Use Unity Catalog as the model registry and select the experiment."""
        mlflow.set_registry_uri("databricks-uc")
        mlflow.set_experiment(self.experiment_name)

    def evaluate(self, preds):
        """Regression metrics on the validation set."""
        return {
            "mae": mean_absolute_error(self.y_val, preds),
            "rmse": np.sqrt(mean_squared_error(self.y_val, preds)),
            "r2": r2_score(self.y_val, preds),
        }

    def train_and_log(self):
        """Train CatBoost and log params, metrics and the model (with signature) to MLflow."""
        with mlflow.start_run(run_name=self.run_name) as run:
            model = CatBoostRegressor(**self.params, cat_features=self.cat_features, verbose=False)
            model.fit(self.X_train, self.y_train)

            preds = model.predict(self.X_val)
            metrics = self.evaluate(preds)

            mlflow.log_params(self.params)
            mlflow.log_metrics(metrics)

            # Unity Catalog requires a signature (input and output schema)
            signature = infer_signature(self.X_train, preds)
            model_info = mlflow.catboost.log_model(
                model,
                name="model",
                signature=signature,
                input_example=self.X_train.head(5),
            )
            self.run_id = run.info.run_id
            self.model_uri = model_info.model_uri

        print("MAE:", metrics["mae"], "RMSE:", metrics["rmse"], "R2:", metrics["r2"])
        print("Run ID:", self.run_id)

    def register_model(self):
        """Register the logged model as a new version in the Unity Catalog Model Registry."""
        registered = mlflow.register_model(model_uri=self.model_uri, name=self.model_name)
        print("Registered model:", registered.name)
        print("Version:", registered.version)
        return registered

    def run(self):
        self.load_data()
        self.setup_mlflow()
        self.train_and_log()
        self.register_model()