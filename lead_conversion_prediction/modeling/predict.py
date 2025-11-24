"""Model inference - loads best model and makes predictions."""
from pathlib import Path
import pandas as pd
import typer
from xgboost import XGBRFClassifier

from loguru import logger
from lead_conversion_prediction.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    model_path: Path = MODELS_DIR / "lead_model_xgboost.json",
    features_path: Path = PROCESSED_DATA_DIR / "X_test.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "y_test.csv",
):
    """Perform inference using the best model."""
    logger.info("Starting model inference...")
    
    # Load the XGBoost model
    model = XGBRFClassifier()
    model.load_model(str(model_path))
    logger.info(f"Loaded model from {model_path}")
    
    # Load test data
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path)
    logger.info(f"Loaded test data: {len(X)} samples")
    
    # Make predictions on first 5 rows
    predictions = model.predict(X.head(5))
    actual = y.head(5)
    
    # Print results in the expected format
    print(predictions, "   lead_indicator")
    print(actual.to_string(header=False))
    
    logger.success("Inference complete!")


if __name__ == "__main__":
    app()
