"""Data preparation script - loads raw data, cleans, filters, and handles outliers."""
from pathlib import Path
import pandas as pd
import numpy as np
import json
import datetime
import typer

from loguru import logger
from lead_conversion_prediction.config import RAW_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()


def describe_numeric_col(x):
    """Calculate descriptive stats for a numeric column."""
    return pd.Series(
        [x.count(), x.isnull().count(), x.mean(), x.min(), x.max()],
        index=["Count", "Missing", "Mean", "Min", "Max"]
    )


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "raw_data.csv",
    output_path: Path = INTERIM_DATA_DIR / "data_cleaned.csv",
    max_date: str = "2024-01-31",
    min_date: str = "2024-01-01",
):
    """Main data preparation pipeline."""
    logger.info("Starting data preparation...")
    
    # Load raw data
    logger.info(f"Loading training data from {input_path}")
    data = pd.read_csv(input_path)
    logger.info(f"Total rows loaded: {len(data)}")
    
    # Parse and filter by date
    if not max_date:
        max_date_parsed = pd.to_datetime(datetime.datetime.now().date()).date()
    else:
        max_date_parsed = pd.to_datetime(max_date).date()
    
    min_date_parsed = pd.to_datetime(min_date).date()
    
    # Time limit data
    data["date_part"] = pd.to_datetime(data["date_part"]).dt.date
    data = data[(data["date_part"] >= min_date_parsed) & (data["date_part"] <= max_date_parsed)]
    
    actual_min_date = data["date_part"].min()
    actual_max_date = data["date_part"].max()
    date_limits = {"min_date": str(actual_min_date), "max_date": str(actual_max_date)}
    with open(INTERIM_DATA_DIR / 'date_limits.json', 'w') as f:
        json.dump(date_limits, f)
    logger.info(f"Date range: {actual_min_date} to {actual_max_date}")
    
    # Feature selection - drop irrelevant columns
    data = data.drop(
        [
            "is_active", "marketing_consent", "first_booking", 
            "existing_customer", "last_seen"
        ],
        axis=1
    )
    
    # Remove columns that will be added back after EDA
    data = data.drop(
        ["domain", "country", "visited_learn_more_before_booking", "visited_faq"],
        axis=1
    )
    
    # Data cleaning
    data["lead_indicator"].replace("", np.nan, inplace=True)
    data["lead_id"].replace("", np.nan, inplace=True)
    data["customer_code"].replace("", np.nan, inplace=True)
    
    data = data.dropna(axis=0, subset=["lead_indicator"])
    data = data.dropna(axis=0, subset=["lead_id"])
    
    data = data[data.source == "signup"]
    result = data.lead_indicator.value_counts(normalize=True)
    
    logger.info("Target value distribution:")
    for val, n in zip(result.index, result):
        logger.info(f"  {val}: {n:.4f}")
    
    # Create categorical data columns
    vars_to_convert = [
        "lead_id", "lead_indicator", "customer_group", "onboarding", 
        "source", "customer_code"
    ]
    
    for col in vars_to_convert:
        data[col] = data[col].astype("object")
    
    # Separate categorical and continuous columns
    cont_vars = data.loc[:, ((data.dtypes=="float64")|(data.dtypes=="int64"))]
    cat_vars = data.loc[:, (data.dtypes=="object")]
    
    logger.info(f"Continuous columns: {list(cont_vars.columns)}")
    logger.info(f"Categorical columns: {list(cat_vars.columns)}")
    
    # Outlier removal using Z-score (2 std devs)
    cont_vars = cont_vars.apply(
        lambda x: x.clip(lower=(x.mean()-2*x.std()), upper=(x.mean()+2*x.std()))
    )
    outlier_summary = cont_vars.apply(describe_numeric_col).T
    outlier_summary.to_csv(INTERIM_DATA_DIR / 'outlier_summary.csv')
    logger.info("Outlier removal complete")
    
    # Save categorical missing impute reference
    cat_missing_impute = cat_vars.mode(numeric_only=False, dropna=True)
    cat_missing_impute.to_csv(INTERIM_DATA_DIR / 'cat_missing_impute.csv')
    
    # Save cleaned data (before imputation and scaling)
    data_cleaned = pd.concat([cat_vars, cont_vars], axis=1)
    data_cleaned.to_csv(output_path, index=False)
    
    logger.success(f"Data preparation complete. Cleaned data saved to {output_path}")
    logger.info(f"Final row count: {len(data_cleaned)}")


if __name__ == "__main__":
    app()
