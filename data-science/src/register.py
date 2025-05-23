# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Registers the best-trained ML model from the sweep job.
"""

import argparse
from pathlib import Path
import mlflow
import os 
import json
import mlflow.sklearn

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Name under which model will be registered')
    parser.add_argument('--model_path', type=str, help='Model directory')
    parser.add_argument("--model_info_output_path", type=str, help="Path to write model info JSON")
    args, _ = parser.parse_known_args()
    print(f'Arguments: {args}')

    return args

def main(args):
    '''Loads the best-trained model from the sweep job and registers it'''

    print("Registering", args.model_name)

    # Step 1: Load model (optional for validation, not required for logging)
    model_uri = args.model_path

    # Step 2: Log model to MLflow
    mlflow.log_artifacts(model_uri)

    # Step 3: Register the model
    registered_model = mlflow.register_model(model_uri=model_uri, name=args.model_name)
    print(f"Registered model: {registered_model.name}, version: {registered_model.version}")

    # Step 4: Write model registration info to output JSON
    model_info = {
        "model_name": registered_model.name,
        "model_version": registered_model.version
    }

    output_dir = Path(args.model_info_output_path).parent
    os.makedirs(output_dir, exist_ok=True)

    with open(args.model_info_output_path, "w") as f:
        json.dump(model_info, f)

    print(f"Model registration info saved to: {args.model_info_output_path}")


if __name__ == "__main__":
    mlflow.start_run()
    
    args = parse_args()

    lines = [
        f"Model name: {args.model_name}",
        f"Model path: {args.model_path}",
        f"Model info output path: {args.model_info_output_path}"
    ]

    for line in lines:
        print(line)

    main(args)
    mlflow.end_run()
