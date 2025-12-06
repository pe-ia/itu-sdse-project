import os
import joblib
import yaml
import pandas as pd
from pathlib import Path

def load_config():
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parents[2] / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

CONFIG = load_config()

def get_model_path(filename):
    """Get the full path for a model file."""
    model_dir = CONFIG.get("models", {}).get("dir", "models/")
    # Ensure directory exists
    os.makedirs(model_dir, exist_ok=True)
    return os.path.join(model_dir, filename)

def get_processed_data_path(filename):
    """Get the full path for a processed data file."""
    data_dir = CONFIG.get("data", {}).get("processed", "data/processed/")
    
    # If the config points to a file, extract the directory
    if not data_dir.endswith("/"):
         data_dir = os.path.dirname(data_dir)
    
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, filename)

def save_model(model, filename, model_dir=None):
    """Save a model to the configured models directory."""
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, filename)
    else:
        path = get_model_path(filename)
    
    joblib.dump(model, path)
    print(f"Model saved to {path}")
    return path

def load_model(filename):
    """Load a model from the configured models directory."""
    path = get_model_path(filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)

def save_data(df, filename):
    """Save a dataframe to the processed data directory."""
    path = get_processed_data_path(filename)
    if filename.endswith(".csv"):
        df.to_csv(path, index=False)
    elif filename.endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        # Default to csv
        df.to_csv(path, index=False)
    print(f"Data saved to {path}")
    return path
