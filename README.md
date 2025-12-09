# Lead Conversion Prediction

Predicts whether a user signup will convert into a paying customer.

## Project Context

This repository is a solution for the **ITU BDS MLOPS'25 Project**. The goal was to take an existing monolithic Python notebook and restructure it into a professional, reproducible, and automated MLOps project. Standard MLOps practices are adheared to, and the cookie cutter data science project structure is used.

### Architecture

The project follows the architecture defined in the diagram below:

![Project Architecture](docs/project-architecture.png)

## Quick Start

### 1. Environment Setup

First, ensure Python 3.10+ is installed. Then install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Fetching Data

[DVC (Data Version Control)](https://dvc.org/) is used to manage the datasets. The data is hosted remotely. To pull the latest version of the raw data:

```bash
dvc update data/raw/raw_data.csv.dvc
```

This will populate `data/raw/` with the original immutable dataset.

## Project Structure

```
├── .github/workflows  <- GitHub Actions for CI/CD
├── dagger/            <- Dagger pipeline definitions (Go)
├── data/              <- Data directory (raw, interim, processed)
├── lead_conversion_prediction/ <- Main source code
│   ├── dataset.py     <- Data cleaning and download scripts
│   ├── features.py    <- Feature engineering logic
│   ├── modeling/      <- Model training and inference
│   └── utils/         <- Utility scripts (storage, config)
├── models/            <- Serialized model artifacts (tracked by DVC)
├── notebooks/         <- Jupyter notebooks for exploration
├── tests/             <- Unit tests
├── Makefile           <- Make commands
├── dagger.json        <- Dagger module config
└── dvc.yaml           <- DVC pipeline configuration
```

## Running the Pipeline

[Dagger](https://dagger.io/) is used to containerize the ML pipeline. This ensures that the code runs exactly the same way locally as it does in the CI environment.

**Prerequisites:**
- Docker running
- Dagger CLI

### Run the Full Pipeline

To clean data, engineer features, train the model, and package the artifacts:

```bash
dagger call pipeline --source . export --path ./models.tar
```

This commands runs the entire end-to-end flow and exports the trained model to `models.tar`.

### Run Individual Steps

You can also run specific stages of the pipeline:

```bash
dagger call prepare-data --source .  # Data cleaning & features
dagger call train --source .         # Training only
dagger call predict --source .       # Run inference validation
```

## Automation (CI/CD)

GitHub Actions is used to automate the workflow.

1.  **CI Pipeline**: Triggered on push/PR to `main`.
    -   Pulls data via DVC.
    -   Runs Unit Tests (`pytest`).
    -   Executes the Dagger pipeline to train the model.
    -   Uploads the model artifact.
2.  **Validation**: A generic `model-validator` action picks up the trained artifact and runs inference to verify it produces the expected output.

## Data & Artifact Management

-   **Raw Data**: Stored in `data/raw/`, never modified.
-   **Processed Data**: Stored in `data/processed/`.
-   **Model Artifacts**: Saved in `models/`.

A helper module `lead_conversion_prediction.utils.storage` is used to ensure standardized paths for loading and saving these files.
