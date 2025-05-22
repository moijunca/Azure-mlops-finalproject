# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Prepares raw data and provides training and test datasets.
"""

import argparse
from pathlib import Path
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import mlflow

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser(description="prep")
    parser.add_argument("--raw_data", type=str, help="Path to raw data")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--test_data", type=str, help="Path to test dataset")
    parser.add_argument("--test_train_ratio", type=float, default=0.2, help="Test-train ratio")
    return parser.parse_args()

def main(args):
    '''Read, preprocess, split, and save datasets'''

    # Read raw CSV
    df = pd.read_csv(args.raw_data)

    # Label encode all categorical columns
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Train-test split
    train_df, test_df = train_test_split(df, test_size=args.test_train_ratio, random_state=42)

    # Create output dirs
    os.makedirs(args.train_data, exist_ok=True)
    os.makedirs(args.test_data, exist_ok=True)

    # Write outputs
    train_path = os.path.join(args.train_data, "train.csv")
    test_path = os.path.join(args.test_data, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Log some basic metrics
    mlflow.log_metric("train_rows", len(train_df))
    mlflow.log_metric("test_rows", len(test_df))

if __name__ == "__main__":
    mlflow.start_run()

    args = parse_args()

    print(f"Raw data path: {args.raw_data}")
    print(f"Train dataset output path: {args.train_data}")
    print(f"Test dataset output path: {args.test_data}")
    print(f"Test-train ratio: {args.test_train_ratio}")

    main(args)

    mlflow.end_run()
