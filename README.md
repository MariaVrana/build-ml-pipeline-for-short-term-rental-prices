# NYC Airbnb Price Prediction Pipeline

This project implements a full machine learning pipeline for predicting short-term rental prices in New York City using Airbnb data. The pipeline is modular, reproducible, and leverages MLflow, Weights & Biases (wandb), and conda environments for experiment tracking and reproducibility.

## Project Structure

```
├── components/                # Reusable pipeline components (data cleaning, splitting, etc.)
├── src/                      # Source code for pipeline steps
│   ├── basic_cleaning/       # Data cleaning step
│   ├── data_check/           # Data validation and checks
│   ├── eda/                  # Exploratory Data Analysis
│   ├── train_random_forest/  # Model training step
│   └── ...                   # Other steps
├── config.yaml               # Main pipeline configuration
├── conda.yml                 # Main conda environment
├── environment.yml           # Main environment to run the project
├── MLproject                 # MLflow project file
├── main.py                   # Pipeline orchestrator
├── README.md                 # Project documentation
└── ...
```

## Features
- **Data Download & Cleaning**: Download raw data from W&B, clean and filter it.
- **EDA**: Automated exploratory data analysis.
- **Data Validation**: Check for schema, distribution, and value ranges.
- **Data Splitting**: Train/validation/test split with stratification.
- **Model Training**: Train a Random Forest regressor with hyperparameter tuning.
- **Model Evaluation**: Test the trained model and log metrics/artifacts.
- **Experiment Tracking**: All steps tracked with MLflow and wandb.
- **Reproducibility**: Each step runs in its own conda environment.

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/MariaVrana/build-ml-pipeline-for-short-term-rental-prices.git
   cd build-ml-pipeline-for-short-term-rental-prices
   ```

2. **Set up conda environment**
   ```bash
   conda env create -f environment.yml
   conda activate nyc_airbnb_dev
   ```

3. **Configure wandb**
   ```bash
   wandb login
   ```

4. **Run the pipeline**
   ```bash
   mlflow run . -P steps=all
   # Or run specific steps:
   mlflow run . -P steps=download,basic_cleaning,data_check,data_split,train_random_forest
   ```

5. **Check results**
   - Artifacts and metrics are logged to wandb and MLflow.
   - Cleaned data and trained models are saved as artifacts.

## Configuration
- All pipeline parameters are set in `config.yaml`.
- Each component has its own `conda.yml` for dependencies.

## Requirements
- Python 3.13 
- conda
- MLflow
- wandb

## Useful Commands
- **Run a single step:**
  ```bash
  mlflow run . -P steps=basic_cleaning
  ```
- **Override config values:**
  ```bash
  mlflow run . -P steps=basic_cleaning -P etl.min_price=20
  ```

## Authors
- Maria Vrana
- Udacity

## License
This project is licensed under the MIT License.

