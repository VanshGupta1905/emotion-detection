import json
from mlflow.tracking import MlflowClient

client = MlflowClient()
with open("reports/metrics.json", "r") as f:
    metrics = json.load(f)

run_id = metrics["run_id"]
try:
    client.get_registered_model("random_forest_model")
except:
    client.create_registered_model("random_forest_model")
client.create_model_version(
    name="random_forest_model",
    description="A random forest model for sentiment analysis",
    run_id=run_id,
    tags={"version": "1",'Author':'Vansh Gupta'},
    source=f"runs:/{run_id}/model",
)