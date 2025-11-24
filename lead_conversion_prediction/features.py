"""Feature engineering - imputation, scaling, binning, and encoding."""
from pathlib import Path
import pandas as pd
import json
import typer
import joblib
from sklearn.preprocessing import MinMaxScaler

from loguru import logger
from lead_conversion_prediction.config import INTERIM_DATA_DIR, MODELS_DIR

app = typer.Typer()


def impute_missing_values(x, method="mean"):
    """Impute missing values for a column."""
    if (x.dtype == "float64") | (x.dtype == "int64"):
        x = x.fillna(x.mean()) if method=="mean" else x.fillna(x.median())
    else:
        x = x.fillna(x.mode()[0])
    return x


@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / "data_cleaned.csv",
    output_path: Path = INTERIM_DATA_DIR / "train_data_gold.csv",
):
    """Feature engineering pipeline."""
    logger.info("Starting feature engineering...")
    
    # Load cleaned data
    data = pd.read_csv(input_path)
    logger.info(f"Loaded {len(data)} rows")
    
    # Convert categorical columns to object type (they may be read as int/bool from CSV)
    categorical_cols = ["lead_id", "lead_indicator", "customer_group", "onboarding", "source", "customer_code"]
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].astype("object")
    logger.info("Converted categorical columns to object type")
    
    # Separate categorical and continuous columns
    cont_vars = data.loc[:, ((data.dtypes=="float64")|(data.dtypes=="int64"))]
    cat_vars = data.loc[:, (data.dtypes=="object")]
    
    # Impute continuous variables
    cont_vars = cont_vars.apply(impute_missing_values)
    logger.info("Continuous variables imputed")
    
    # Impute categorical variables
    cat_vars.loc[cat_vars['customer_code'].isna(),'customer_code'] = 'None'
    cat_vars = cat_vars.apply(impute_missing_values)
    logger.info("Categorical variables imputed")
    
    # Data standardization using MinMaxScaler
    scaler_path = MODELS_DIR / 'scaler.pkl'
    scaler = MinMaxScaler()
    scaler.fit(cont_vars)
    joblib.dump(value=scaler, filename=scaler_path)
    logger.info(f'Saved scaler to {scaler_path}')
    
    cont_vars = pd.DataFrame(scaler.transform(cont_vars), columns=cont_vars.columns)
    logger.info("Data standardization complete")
    
    # Combine data
    cont_vars = cont_vars.reset_index(drop=True)
    cat_vars = cat_vars.reset_index(drop=True)
    data = pd.concat([cat_vars, cont_vars], axis=1)
    logger.info(f"Data combined. Rows: {len(data)}")
    
    # Save columns for drift detection
    data_columns = list(data.columns)
    with open(INTERIM_DATA_DIR / 'columns_drift.json', 'w+') as f:
        json.dump(data_columns, f)
    
    # Save training data
    data.to_csv(INTERIM_DATA_DIR / 'training_data.csv', index=False)
    
    # Binning source column
    data['bin_source'] = data['source']
    values_list = ['li', 'organic','signup','fb']
    data.loc[~data['source'].isin(values_list),'bin_source'] = 'Others'
    mapping = {
        'li' : 'socials', 
        'fb' : 'socials', 
        'organic': 'group1', 
        'signup': 'group1'
    }
    data['bin_source'] = data['source'].map(mapping)
    logger.info("Source binning complete")
    
    # Save gold dataset
    data.to_csv(output_path, index=False)
    logger.success(f"Feature engineering complete. Gold data saved to {output_path}")


if __name__ == "__main__":
    app()
