import argparse
import mlflow
import os
import joblib

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
args = parser.parse_args()

model_path = args.model

print(f"Loading model from: {model_path}")
model = joblib.load(os.path.join(model_path, "model.pkl"))

mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    registered_model_name="used_car_price_model"
)
