"""
Statistics and Trends Assignment
--------------------------------
This file analyzes the coffee sales dataset and demonstrates:
- Data preprocessing and exploration
- Computation of four statistical moments
- Visualization using relational, categorical, and statistical plots
The code follows the provided template and PEP-8 conventions.
"""

from corner import corner  # type: ignore # keep as per template, though not used here
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss # type: ignore
import seaborn as sns


def plot_relational_plot(df):
    """Scatter plot showing relationship between hour_of_day and money."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="hour_of_day",
        y="money",
        hue="Time_of_Day",
        s=100,
        palette="coolwarm",
        ax=ax
    )
    ax.set_title("Relational Plot: Money vs Hour of Day")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Money Spent")
    plt.legend(title="Time of Day")
    plt.tight_layout()
    plt.savefig("relational_plot.png")
    plt.show()
    return


def plot_categorical_plot(df):
    """Bar plot showing average money spent by coffee type."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=df,
        x="coffee_name",
        y="money",
        estimator=np.mean,
        palette="viridis",
        ax=ax
    )
    ax.set_title("Categorical Plot: Average Spending by Coffee Type")
    ax.set_xlabel("Coffee Name")
    ax.set_ylabel("Average Money Spent")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("categorical_plot.png")
    plt.show()
    return


def plot_statistical_plot(df):
    """Box plot showing distribution of money across times of day."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="Time_of_Day",
        y="money",
        palette="Set2",
        ax=ax
    )
    ax.set_title("Statistical Plot: Money Distribution by Time of Day")
    ax.set_xlabel("Time of Day")
    ax.set_ylabel("Money Spent")
    plt.tight_layout()
    plt.savefig("statistical_plot.png")
    plt.show()
    return


def statistical_analysis(df, col: str):
    """Calculate the four statistical moments for the given column."""
    mean = df[col].mean()
    stddev = df[col].std()
    skew = df[col].skew()
    excess_kurtosis = df[col].kurtosis()
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """
    Preprocess the dataset:
    - Display info, summary, and missing values
    - Fill missing numeric values with mean
    - Display correlations
    """
    print("----- FIRST 5 ROWS OF DATA -----")
    print(df.head())

    print("\n----- DATASET INFORMATION -----")
    print(df.info())

    print("\n----- DESCRIPTIVE STATISTICS -----")
    print(df.describe())

    print("\n----- CHECKING MISSING VALUES -----")
    print(df.isnull().sum())

    # Fill numeric missing values with mean
    df.fillna(df.mean(numeric_only=True), inplace=True)

    print("\n----- CORRELATION HEATMAP -----")
    plt.figure(figsize=(6, 4))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="Blues", fmt=".2f")
    plt.title("Correlation Heatmap of Numeric Columns")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png")
    plt.show()

    return df


def writing(moments, col):
    """Display interpretation of the four moments."""
    print(f"\nFor the attribute '{col}':")
    print(f"Mean = {moments[0]:.2f}, "
          f"Standard Deviation = {moments[1]:.2f}, "
          f"Skewness = {moments[2]:.2f}, and "
          f"Excess Kurtosis = {moments[3]:.2f}.")

    # Interpret skewness and kurtosis
    skew_text = ("right-skewed" if moments[2] > 0 else
                 "left-skewed" if moments[2] < 0 else
                 "symmetrical")
    kurt_text = ("leptokurtic" if moments[3] > 0 else
                 "platykurtic" if moments[3] < 0 else
                 "mesokurtic")

    print(f"The data is {skew_text} and {kurt_text}.")
    return


def main():
    """Main workflow of the analysis."""
    # Load dataset
    df = pd.read_csv(r"data.csv")

    # Preprocess
    df = preprocessing(df)

    # Choose column for analysis
    col = "money"

    # Plots
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)

    # Statistical analysis
    moments = statistical_analysis(df, col)

    # Write interpretation
    writing(moments, col)
    return


if __name__ == "__main__":
    main()