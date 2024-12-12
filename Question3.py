import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from scipy.stats import ttest_ind, pearsonr

# Load the data
FzRfg_SM_2020 = pd.read_excel("Fz_Rfg Substitute Meat_POS_2020.xlsx")
FzRfg_SM_2021 = pd.read_excel("Fz_Rfg Substitute Meat_POS_2021.xlsx")
FzRfg_SM_2022 = pd.read_excel("Fz_Rfg Substitute Meat_POS_2022.xlsx")
FzRfg_SM_2023 = pd.read_excel("Fz_Rfg Substitute Meat_POS_2023.xlsx")
FzRfg_SM_2024 = pd.read_excel("Fz_Rfg Substitute Meat_POS_2024.xlsx")

# Function to clean the data
def clean_data(data):
    data_cleaned = data.dropna(subset=["Dollar Sales", "ACV Weighted Distribution"])
    data_cleaned = data_cleaned[data_cleaned["Geography"] != "Total US - Multi Outlet + Conv"]
    data_cleaned["Dollar Sales"] = pd.to_numeric(data_cleaned["Dollar Sales"], errors="coerce")
    data_cleaned["ACV Weighted Distribution"] = pd.to_numeric(data_cleaned["ACV Weighted Distribution"], errors="coerce")

    # Add high vs. low coverage classification
    median_acv = data_cleaned["ACV Weighted Distribution"].median()
    data_cleaned["Coverage"] = np.where(data_cleaned["ACV Weighted Distribution"] >= median_acv, "High", "Low")
    
    return data_cleaned

# Clean the datasets
datasets = [FzRfg_SM_2020, FzRfg_SM_2021, FzRfg_SM_2022, FzRfg_SM_2023, FzRfg_SM_2024]
cleaned_datasets = [clean_data(dataset) for dataset in datasets]

# Descriptive analytics
for year, data in zip(range(2020, 2025), cleaned_datasets):
    summary_stats = data.groupby("Coverage")["Dollar Sales"].agg(["mean", "median"]).reset_index()
    print(f"Summary Statistics for {year}:\n{summary_stats}\n")

# Correlation analysis
for year, data in zip(range(2020, 2025), cleaned_datasets):
    high_corr, _ = pearsonr(data.loc[data["Coverage"] == "High", "ACV Weighted Distribution"],
                            data.loc[data["Coverage"] == "High", "Dollar Sales"])
    low_corr, _ = pearsonr(data.loc[data["Coverage"] == "Low", "ACV Weighted Distribution"],
                           data.loc[data["Coverage"] == "Low", "Dollar Sales"])
    print(f"{year} High Coverage Correlation: {high_corr}")
    print(f"{year} Low Coverage Correlation: {low_corr}\n")

# T-tests
for year, data in zip(range(2020, 2025), cleaned_datasets):
    high_group = data.loc[data["Coverage"] == "High", "Dollar Sales"]
    low_group = data.loc[data["Coverage"] == "Low", "Dollar Sales"]
    t_stat, p_value = ttest_ind(high_group, low_group, nan_policy="omit")
    print(f"T-test for {year}: t-stat={t_stat}, p-value={p_value}\n")

# Scatter plots
for year, data in zip(range(2020, 2025), cleaned_datasets):
    plt.figure()
    sns.scatterplot(data=data, x="ACV Weighted Distribution", y="Dollar Sales", hue="Coverage", alpha=0.6)
    plt.title(f"{year}: Relationship between ACV and Dollar Sales")
    plt.xlabel("ACV Weighted Distribution")
    plt.ylabel("Dollar Sales")
    plt.legend(title="Coverage")
    plt.show()

# Box plots
for year, data in zip(range(2020, 2025), cleaned_datasets):
    plt.figure()
    sns.boxplot(data=data, x="Coverage", y="Dollar Sales", palette="Set2")
    plt.ylim(0, 5000)
    plt.title(f"{year}: Dollar Sales by Coverage")
    plt.xlabel("Coverage")
    plt.ylabel("Dollar Sales")
    plt.show()

# Function to build and evaluate a linear regression model
def build_linear_model(data, year):
    # Drop rows with missing data in relevant columns
    data = data.dropna(subset=["Dollar Sales", "ACV Weighted Distribution"])
    
    # Define independent (X) and dependent (y) variables
    X = data["ACV Weighted Distribution"]
    y = data["Dollar Sales"]
    
    # Add a constant for the intercept in the regression model
    X = sm.add_constant(X)
    
    # Fit the linear regression model
    model = sm.OLS(y, X).fit()
    
    # Print the model summary
    print(f"Linear Regression Model for {year}")
    print(model.summary())
    
    # Scatter plot with regression line
    plt.figure()
    plt.scatter(data["ACV Weighted Distribution"], data["Dollar Sales"], alpha=0.6, label="Data")
    plt.plot(data["ACV Weighted Distribution"], model.predict(X), color='red', label="Regression Line")
    plt.title(f"{year}: Dollar Sales vs ACV Weighted Distribution")
    plt.xlabel("ACV Weighted Distribution")
    plt.ylabel("Dollar Sales")
    plt.legend()
    plt.show()

# Build and evaluate models for each year
for year, data in zip(range(2024, 2025), cleaned_datasets):
    build_linear_model(data, year)