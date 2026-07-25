import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# -------------------------
# Load Dataset
# -------------------------

df = pd.read_csv("titanic.csv")


print("="*60)
print("FIRST 5 ROWS")
print("="*60)

print(df.head())


# -------------------------
# Dataset Information
# -------------------------

print("\nDATASET INFORMATION")

df.info()


# -------------------------
# Shape
# -------------------------

print("\nDataset Shape")

print(df.shape)

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# -------------------------
# Column Names
# -------------------------

print("\nColumn Names")

print(df.columns)


# -------------------------
# Data Types
# -------------------------

print("\nData Types")

print(df.dtypes)


# -------------------------
# Missing Values
# -------------------------

print("\nMissing Values")

print(df.isnull().sum())


# -------------------------
# Duplicate Rows
# -------------------------

print("\nDuplicate Rows")

print(df.duplicated().sum())


# -------------------------
# Summary Statistics
# -------------------------

print("\nSummary Statistics")

print(df.describe())


# -------------------------
# Survival Count
# -------------------------

print("\nSurvival Count")

print(df["Survived"].value_counts())


# -------------------------
# Gender Count
# -------------------------

print("\nGender Count")

print(df["Sex"].value_counts())


# -------------------------
# Correlation
# -------------------------

print("\nCorrelation Matrix")

numeric_df = df.select_dtypes(include=np.number)

print(numeric_df.corr())


# -------------------------
# Survival Graph
# -------------------------

plt.figure(figsize=(7,5))

sns.countplot(data=df, x="Survived")

plt.title("Passenger Survival Count")

plt.xlabel("Survived (0 = No, 1 = Yes)")

plt.show()


# -------------------------
# Gender vs Survival
# -------------------------

plt.figure(figsize=(7,5))

sns.countplot(
    data=df,
    x="Sex",
    hue="Survived"
)

plt.title("Survival Based on Gender")

plt.show()


# -------------------------
# Age Distribution
# -------------------------

plt.figure(figsize=(8,5))

sns.histplot(
    data=df,
    x="Age",
    bins=30
)

plt.title("Passenger Age Distribution")

plt.show()


# -------------------------
# Passenger Class Analysis
# -------------------------

plt.figure(figsize=(7,5))

sns.countplot(
    data=df,
    x="Pclass",
    hue="Survived"
)

plt.title("Survival Based on Passenger Class")

plt.show()


# -------------------------
# Box Plot Age
# -------------------------

plt.figure(figsize=(7,5))

sns.boxplot(
    data=df,
    y="Age"
)

plt.title("Age Outliers")

plt.show()


# -------------------------
# Heatmap
# -------------------------

plt.figure(figsize=(8,6))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.show()


print("\nEDA COMPLETED SUCCESSFULLY")
