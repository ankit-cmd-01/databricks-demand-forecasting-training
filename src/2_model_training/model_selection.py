"""Champion / challenger model selection (Unity Catalog aliases).

challenger = the model we just trained (it only lives in the MLflow run for now)
champion   = the registered model that holds the alias "champion"

Rule: the challenger is registered in Unity Catalog ONLY if it beats the champion.
If it wins, it gets the alias "champion". If it loses, nothing is registered.
"""
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error

CHAMPION = "champion"   # alias name in Unity Catalog


def select_champion(model_name, model_uri, challenger_mae, X_val, y_val):
    """Register the new model and make it champion only if it has a lower MAE."""
    mlflow.set_registry_uri("databricks-uc")
    client = MlflowClient()

    # 1. Is there already a champion?
    try:
        champion = client.get_model_version_by_alias(model_name, CHAMPION)
    except MlflowException:
        champion = None   # first run, nobody is champion yet

    # 2. Score the champion on the SAME validation data (fair comparison)
    if champion is None:
        print("No champion yet -> the new model becomes champion.")
        new_is_better = True
    else:
        champion_model = mlflow.pyfunc.load_model(f"models:/{model_name}@{CHAMPION}")
        champion_mae = mean_absolute_error(y_val, champion_model.predict(X_val))
        print(f"Champion (v{champion.version}) MAE: {champion_mae:.2f}")
        print(f"Challenger (new)  MAE: {challenger_mae:.2f}")
        new_is_better = challenger_mae < champion_mae   # lower MAE = better

    # 3. Winner gets stored in Unity Catalog and takes the alias
    if new_is_better:
        registered = mlflow.register_model(model_uri=model_uri, name=model_name)
        client.set_registered_model_alias(model_name, CHAMPION, registered.version)
        print(f"New champion: version {registered.version}")
    else:
        print("Challenger is not better -> NOT registered, champion stays.")