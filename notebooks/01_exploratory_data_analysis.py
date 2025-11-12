"""
---
jupytext:
  text_representation:
    extension: .py
    format_name: light
    format_version: '1.5'
    jupytext_version: 1.16.2
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
"""

# %% [markdown]
# # Customer Churn - Exploratory Data Analysis
#
# This notebook explores the customer churn dataset and provides insights.

# %% [markdown]
# ## 1. Setup and Data Loading

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
import os

# --- Robust Path Setup for both .py Script and .ipynb Notebook ---
try:
    # Path setup for when running as a normal Python script.
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    # Path setup for when running interactively inside a Jupyter Notebook cell.
    PROJECT_ROOT = Path(os.getcwd()).parent

# Define absolute paths for file I/O
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
NOTEBOOKS_DIR.mkdir(exist_ok=True)

# Save original CWD and change to Project Root to fix relative path issues
original_cwd = os.getcwd()
os.chdir(PROJECT_ROOT)

# Add project root to sys.path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from config import config
from src.utils.logger import get_logger

# Setup
logger = get_logger(__name__)
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# Load data
data_path = config.get("data.raw_data_path")
df = pd.read_csv(data_path)

# Reset CWD back after loading data
os.chdir(original_cwd)

print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

# %% [markdown]
# ## 2. Data Overview

# %%
print("\n=== Data Info ===")
print(df.info())

print("\n=== Statistical Summary ===")
print(df.describe())

print("\n=== Missing Values ===")
print(df.isnull().sum())

# %% [markdown]
# ## 3. Target Variable Analysis

# %%
print("\n=== Churn Distribution ===")
churn_counts = df["Churn"].value_counts()
print(churn_counts)
print(f"\nChurn Rate: {(df['Churn'] == 'Yes').mean():.2%}")

# Plot churn distribution
plt.figure(figsize=(8, 6))
churn_counts.plot(kind="bar", color=["green", "red"])
plt.title("Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
# Using absolute path for savefig
plt.savefig(NOTEBOOKS_DIR / "churn_distribution.png")
print(f"Saved: {NOTEBOOKS_DIR / 'churn_distribution.png'}")

# %% [markdown]
# ## 4. Numerical Features Analysis

# %%
numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

print("\n=== Numerical Features Statistics ===")
print(df[numerical_cols].describe())

# Distribution plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for idx, col in enumerate(numerical_cols):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        axes[idx].hist(df[col].dropna(), bins=30, edgecolor="black")
        axes[idx].set_title(f"{col} Distribution")
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel("Frequency")

plt.tight_layout()
# Using absolute path for savefig
plt.savefig(NOTEBOOKS_DIR / "numerical_distributions.png")
print(f"Saved: {NOTEBOOKS_DIR / 'numerical_distributions.png'}")

# %% [markdown]
# ## 5. Churn by Numerical Features

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for idx, col in enumerate(numerical_cols):
    if col in df.columns:
        df_clean = df[[col, "Churn"]].copy()
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
        df_clean = df_clean.dropna()
        df_clean.boxplot(column=col, by="Churn", ax=axes[idx])
        axes[idx].set_title(f"{col} by Churn")
        axes[idx].set_xlabel("Churn")
        axes[idx].set_ylabel(col)

plt.suptitle("")
plt.tight_layout()
# Using absolute path for savefig
plt.savefig(NOTEBOOKS_DIR / "churn_by_numerical.png")
print(f"Saved: {NOTEBOOKS_DIR / 'churn_by_numerical.png'}")

# %% [markdown]
# ## 6. Categorical Features Analysis

# %%
categorical_cols = [
    "Contract",
    "PaymentMethod",
    "InternetService",
    "gender",
    "SeniorCitizen",
]

print("\n=== Categorical Features ===")
for col in categorical_cols:
    if col in df.columns:
        print(f"\n{col}:")
        print(df[col].value_counts())

# %% [markdown]
# ## 7. Churn by Categorical Features

# %%
# Contract vs Churn
if "Contract" in df.columns:
    plt.figure(figsize=(10, 6))
    contract_churn = pd.crosstab(df["Contract"], df["Churn"], normalize="index") * 100
    contract_churn.plot(kind="bar", stacked=False)
    plt.title("Churn Rate by Contract Type")
    plt.xlabel("Contract Type")
    plt.ylabel("Percentage")
    plt.xticks(rotation=45)
    plt.legend(title="Churn")
    plt.tight_layout()
    # Using absolute path for savefig
    plt.savefig(NOTEBOOKS_DIR / "churn_by_contract.png")
    print(f"Saved: {NOTEBOOKS_DIR / 'churn_by_contract.png'}")

# Payment Method vs Churn
if "PaymentMethod" in df.columns:
    plt.figure(figsize=(12, 6))
    payment_churn = (
        pd.crosstab(df["PaymentMethod"], df["Churn"], normalize="index") * 100
    )
    payment_churn.plot(kind="bar", stacked=False)
    plt.title("Churn Rate by Payment Method")
    plt.xlabel("Payment Method")
    plt.ylabel("Percentage")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Churn")
    plt.tight_layout()
    # Using absolute path for savefig
    plt.savefig(NOTEBOOKS_DIR / "churn_by_payment.png")
    print(f"Saved: {NOTEBOOKS_DIR / 'churn_by_payment.png'}")

# Internet Service vs Churn
if "InternetService" in df.columns:
    plt.figure(figsize=(10, 6))
    internet_churn = (
        pd.crosstab(df["InternetService"], df["Churn"], normalize="index") * 100
    )
    internet_churn.plot(kind="bar", stacked=False)
    plt.title("Churn Rate by Internet Service")
    plt.xlabel("Internet Service")
    plt.ylabel("Percentage")
    plt.xticks(rotation=0)
    plt.legend(title="Churn")
    plt.tight_layout()
    # Using absolute path for savefig
    plt.savefig(NOTEBOOKS_DIR / "churn_by_internet.png")
    print(f"Saved: {NOTEBOOKS_DIR / 'churn_by_internet.png'}")

# %% [markdown]
# ## 8. Correlation Analysis

# %%
# Convert categorical to numerical for correlation
df_corr = df.copy()
df_corr["Churn"] = df_corr["Churn"].map({"Yes": 1, "No": 0})

# Select numerical columns
numerical_features = ["tenure", "MonthlyCharges", "TotalCharges", "Churn"]
df_corr_num = df_corr[numerical_features].copy()

for col in df_corr_num.columns:
    df_corr_num[col] = pd.to_numeric(df_corr_num[col], errors="coerce")

df_corr_num = df_corr_num.dropna()

# Correlation matrix
plt.figure(figsize=(10, 8))
correlation_matrix = df_corr_num.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Matrix")
plt.tight_layout()
# Using absolute path for savefig
plt.savefig(NOTEBOOKS_DIR / "correlation_matrix.png")
print(f"Saved: {NOTEBOOKS_DIR / 'correlation_matrix.png'}")

print("\n=== Correlation with Churn ===")
print(correlation_matrix["Churn"].sort_values(ascending=False))

# %% [markdown]
# ## 9. Key Insights

# %%
print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)

# Churn rate
churn_rate = (df["Churn"] == "Yes").mean()
print(f"\n1. Overall churn rate: {churn_rate:.2%}")

# Tenure analysis
if "tenure" in df.columns:
    df_tenure = df.copy()
    df_tenure["tenure"] = pd.to_numeric(df_tenure["tenure"], errors="coerce")
    avg_tenure_churn = df_tenure[df_tenure["Churn"] == "Yes"]["tenure"].mean()
    avg_tenure_stay = df_tenure[df_tenure["Churn"] == "No"]["tenure"].mean()
    print(f"\n2. Average tenure:")
    print(f"   - Churned customers: {avg_tenure_churn:.1f} months")
    print(f"   - Retained customers: {avg_tenure_stay:.1f} months")

# Contract analysis
if "Contract" in df.columns:
    contract_churn_rate = df.groupby("Contract")["Churn"].apply(
        lambda x: (x == "Yes").mean()
    )
    print(f"\n3. Churn rate by contract type:")
    for contract, rate in contract_churn_rate.items():
        print(f"   - {contract}: {rate:.2%}")

# Monthly charges analysis
if "MonthlyCharges" in df.columns:
    df_charges = df.copy()
    df_charges["MonthlyCharges"] = pd.to_numeric(
        df_charges["MonthlyCharges"], errors="coerce"
    )
    avg_charges_churn = df_charges[df_charges["Churn"] == "Yes"][
        "MonthlyCharges"
    ].mean()
    avg_charges_stay = df_charges[df_charges["Churn"] == "No"]["MonthlyCharges"].mean()
    print(f"\n4. Average monthly charges:")
    print(f"   - Churned customers: ${avg_charges_churn:.2f}")
    print(f"   - Retained customers: ${avg_charges_stay:.2f}")

# Internet service analysis
if "InternetService" in df.columns:
    internet_churn_rate = df.groupby("InternetService")["Churn"].apply(
        lambda x: (x == "Yes").mean()
    )
    print(f"\n5. Churn rate by internet service:")
    for service, rate in internet_churn_rate.items():
        print(f"   - {service}: {rate:.2%}")

print("\n" + "=" * 60)
print("EDA Complete! Check the notebooks/ folder for visualizations.")
print("=" * 60)
