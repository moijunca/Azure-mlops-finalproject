# prep.py
import argparse
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import mlflow

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare data")
    parser.add_argument("--raw_data", type=str, help="Path to raw data")
    parser.add_argument("--train_data", type=str, help="Output path for training data")
    parser.add_argument("--test_data", type=str, help="Output path for test data")
    parser.add_argument("--test_train_ratio", type=float, default=0.2, help="Test/train ratio")
    return parser.parse_args()

def main(args):
    df = pd.read_csv(args.raw_data)

    # Label encoding
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    train_df, test_df = train_test_split(df, test_size=args.test_train_ratio, random_state=42)

    # Create directories if needed
    Path(args.train_data).mkdir(parents=True, exist_ok=True)
    Path(args.test_data).mkdir(parents=True, exist_ok=True)

    # Save to CSV inside the output folders
    train_df.to_csv(Path(args.train_data) / "train.csv", index=False)
    test_df.to_csv(Path(args.test_data) / "test.csv", index=False)

    mlflow.log_metric("train_rows", train_df.shape[0])
    mlflow.log_metric("test_rows", test_df.shape[0])

if __name__ == "__main__":
    mlflow.start_run()
    args = parse_args()

    print(f"Raw data path: {args.raw_data}")
    print(f"Train data output path: {args.train_data}")
    print(f"Test data output path: {args.test_data}")
    print(f"Test/train ratio: {args.test_train_ratio}")

    main(args)
    mlflow.end_run()
