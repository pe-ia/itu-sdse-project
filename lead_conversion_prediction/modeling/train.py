"""Model training - trains XGBoost and LogisticRegression models."""
from pathlib import Path
import pandas as pd
import json
import typer
import joblib
import datetime
import warnings
from xgboost import XGBRFClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import uniform, randint

from loguru import logger
from lead_conversion_prediction.config import MODELS_DIR, REPORTS_DIR, TRAIN_DATA_PATH
from lead_conversion_prediction.utils.storage import save_model

warnings.filterwarnings('ignore')

app = typer.Typer()


def create_dummy_cols(df, col):
    """Create one-hot encoding columns."""
    df_dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    new_df = pd.concat([df, df_dummies], axis=1)
    new_df = new_df.drop(col, axis=1)
    return new_df


@app.command()
def main(
    input_path: Path = TRAIN_DATA_PATH,
    models_dir: Path = MODELS_DIR,
    reports_dir: Path = REPORTS_DIR,
):
    """Model training pipeline."""
    logger.info("Starting model training...")
    
    # Load training data
    data = pd.read_csv(input_path)
    logger.info(f"Training data length: {len(data)}")
    
    # Convert categorical columns to object type (they may be read as int/bool from CSV)
    categorical_cols = ["lead_id", "lead_indicator", "customer_group", "onboarding", "source", "customer_code"]
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].astype("object")
    logger.info("Converted categorical columns to object type")
    
    # Drop ID columns and date
    data = data.drop(["lead_id", "customer_code", "date_part"], axis=1)
    
    # Separate categorical columns (matching notebook)
    cat_cols = ["customer_group", "onboarding", "bin_source", "source"]
    cat_vars = data[cat_cols]
    other_vars = data.drop(cat_cols, axis=1)
    
    # Create dummy variables for categorical columns
    for col in cat_vars:
        cat_vars[col] = cat_vars[col].astype("category")
        cat_vars = create_dummy_cols(cat_vars, col)
    
    # Combine data
    data = pd.concat([other_vars, cat_vars], axis=1)
    
    # Convert all to float64
    for col in data:
        data[col] = data[col].astype("float64")
    
    # Split features and target
    y = data["lead_indicator"]
    X = data.drop(["lead_indicator"], axis=1)
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")
    
    # Train/test split for model training and evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.15, stratify=y
    )
    
    logger.info(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    # Note: We do NOT save X_test/y_test to data/processed/ here because
    # data/processed/ already contains the hardcoded test data required for
    # the inference script (predict.py) to match the notebook's output.
    # The split above is used solely for model training and metric evaluation.
    
    # Train XGBoost model
    logger.info("Training XGBoost model...")
    xgb_model = XGBRFClassifier(random_state=42)
    xgb_params = {
        "learning_rate": uniform(1e-2, 3e-1),
        "min_split_loss": uniform(0, 10),
        "max_depth": randint(3, 10),
        "subsample": uniform(0, 1),
        "objective": ["reg:squarederror", "binary:logistic", "reg:logistic"],
        "eval_metric": ["aucpr", "error"]
    }
    
    xgb_grid = RandomizedSearchCV(xgb_model, param_distributions=xgb_params, 
                                   n_jobs=-1, verbose=3, n_iter=10, cv=10, random_state=42)
    xgb_grid.fit(X_train, y_train)
    
    # Evaluate XGBoost
    y_pred_train_xgb = xgb_grid.predict(X_train)
    y_pred_test_xgb = xgb_grid.predict(X_test)
    
    logger.info(f"XGBoost - Accuracy train: {accuracy_score(y_pred_train_xgb, y_train):.4f}")
    logger.info(f"XGBoost - Accuracy test: {accuracy_score(y_pred_test_xgb, y_test):.4f}")
    
    # Save XGBoost model
    xgboost_model = xgb_grid.best_estimator_
    save_model(xgboost_model, 'lead_model_xgboost.pkl', model_dir=models_dir)
    
    # Store XGBoost results
    xgb_report = classification_report(y_train, y_pred_train_xgb, output_dict=True)
    model_results = {
        "lead_model_xgboost.pkl": xgb_report
    }
    
    # Train Logistic Regression model
    logger.info("Training Logistic Regression model...")
    lr_model = LogisticRegression()
    lr_params = {
        'solver': ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
        'penalty':  ["none", "l1", "l2", "elasticnet"],
        'C' : [100, 10, 1.0, 0.1, 0.01]
    }
    
    lr_grid = RandomizedSearchCV(lr_model, param_distributions=lr_params, 
                                  verbose=3, n_iter=10, cv=3, random_state=42)
    lr_grid.fit(X_train, y_train)
    
    # Evaluate Logistic Regression
    best_lr_model = lr_grid.best_estimator_
    y_pred_train_lr = lr_grid.predict(X_train)
    y_pred_test_lr = lr_grid.predict(X_test)
    
    logger.info(f"LogReg - Accuracy train: {accuracy_score(y_pred_train_lr, y_train):.4f}")
    logger.info(f"LogReg - Accuracy test: {accuracy_score(y_pred_test_lr, y_test):.4f}")
    
    # Save Logistic Regression model
    save_model(best_lr_model, 'lead_model_lr.pkl', model_dir=models_dir)
    
    # Store LR results
    lr_report = classification_report(y_test, y_pred_test_lr, output_dict=True)
    model_results["lead_model_lr.pkl"] = lr_report
    
    # Save column list
    column_list_path = models_dir / 'columns_list.json'
    with open(column_list_path, 'w+') as columns_file:
        columns = {'column_names': list(X_train.columns)}
        json.dump(columns, columns_file)
    logger.info(f"Column list saved to {column_list_path}")
    
    # Save model results
    model_results_path = reports_dir / 'model_results.json'
    with open(model_results_path, 'w+') as results_file:
        json.dump(model_results, results_file)
    logger.info(f"Model results saved to {model_results_path}")
    
    logger.success("Model training complete!")


if __name__ == "__main__":
    app()
