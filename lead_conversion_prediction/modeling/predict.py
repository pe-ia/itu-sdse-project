"""Model inference - loads best model and makes predictions."""
from pathlib import Path
import pandas as pd
import typer
import joblib
from loguru import logger
from lead_conversion_prediction.config import MODELS_DIR, TEST_X_PATH, TEST_Y_PATH

app = typer.Typer()


@app.command()
def main(
    model_path: Path = MODELS_DIR / "lead_model_xgboost.pkl",
    features_path: Path = TEST_X_PATH,
    labels_path: Path = TEST_Y_PATH,
):
    """Perform inference using the best model."""
    logger.info("Starting model inference...")
    
    # Load the XGBoost model
    model = joblib.load(model_path)
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
