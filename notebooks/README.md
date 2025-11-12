# Notebooks

This directory contains Jupyter notebooks for exploratory data analysis and experimentation.

## Available Notebooks

### 01_exploratory_data_analysis.py

- Data overview and statistics
- Target variable analysis
- Numerical and categorical feature analysis
- Correlation analysis
- Key insights about customer churn

## Running Notebooks

### Convert Python script to Jupyter notebook:

```bash
jupytext --to notebook 01_exploratory_data_analysis.py
```

### Or run as Python script:

```bash
python3 01_exploratory_data_analysis.py
```

## Generated Visualizations

The EDA script generates the following visualizations:

- `churn_distribution.png` - Distribution of churn vs non-churn
- `numerical_distributions.png` - Distribution of numerical features
- `churn_by_numerical.png` - Boxplots of numerical features by churn
- `churn_by_contract.png` - Churn rate by contract type
- `churn_by_payment.png` - Churn rate by payment method
- `churn_by_internet.png` - Churn rate by internet service
- `correlation_matrix.png` - Correlation heatmap
