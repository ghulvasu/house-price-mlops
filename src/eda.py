# src/eda.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def perform_eda():
    # 1. Load Data
    data_path = os.path.join('data', 'raw', 'Housing.csv')
    df = pd.read_csv(data_path)
    
    # Ensure plots folder exists
    os.makedirs('plots', exist_ok=True)
    print("Starting EDA... Saving plots to 'plots/' folder.")

    # Set a clean visual style
    sns.set_style("whitegrid")

    # -----------------------------------------------------------
    # CHART 1: Distribution of House Prices
    # -----------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.histplot(df['price'], kde=True, color='blue')
    plt.title('Distribution of House Prices', fontsize=14)
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join('plots', '1_price_distribution.png'))
    plt.close()
    print("✅ Saved: 1_price_distribution.png")

    # -----------------------------------------------------------
    # CHART 2: Numeric Correlations (Heatmap)
    # -----------------------------------------------------------
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap (Numerical Features)', fontsize=14)
    plt.savefig(os.path.join('plots', '2_correlation_heatmap.png'))
    plt.close()
    print("✅ Saved: 2_correlation_heatmap.png")

    # -----------------------------------------------------------
    # CHART 3: Bar Plot - Average Price vs Air Conditioning
    # -----------------------------------------------------------
    # This is much clearer than a boxplot. It shows the exact average price difference.
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x='airconditioning', y='price', data=df, palette='viridis', errorbar=None)
    
    plt.title('Average House Price: AC vs No AC', fontsize=14)
    plt.ylabel('Average Price', fontsize=12)
    
    # Add exact price numbers on top of the bars
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 50000, 
                f'${int(height):,}', ha="center", fontsize=12, fontweight='bold')

    plt.savefig(os.path.join('plots', '3_price_vs_ac.png'))
    plt.close()
    print("✅ Saved: 3_price_vs_ac.png")

    # -----------------------------------------------------------
    # CHART 4: Bar Plot - Average Price by Furnishing Status
    # -----------------------------------------------------------
    plt.figure(figsize=(10, 6))
    # We explicitly set order so the chart is easy to read
    ax = sns.barplot(x='furnishingstatus', y='price', data=df, 
                     order=['unfurnished', 'semi-furnished', 'furnished'],
                     palette='magma', errorbar=None)
    
    plt.title('Average House Price by Furnishing Status', fontsize=14)
    plt.ylabel('Average Price', fontsize=12)

    # Add exact price numbers on top of the bars
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 50000, 
                f'${int(height):,}', ha="center", fontsize=11, fontweight='bold')

    plt.savefig(os.path.join('plots', '4_price_vs_furnishing.png'))
    plt.close()
    print("✅ Saved: 4_price_vs_furnishing.png")

if __name__ == "__main__":
    perform_eda()