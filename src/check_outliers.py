# src/check_outliers.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def analyze_outliers():
    # 1. Load Data
    df = pd.read_csv(os.path.join('data', 'raw', 'Housing.csv'))
    
    # 2. Define "Outliers" using the IQR Rule
    # Q3 = The value at the 75% mark (expensive houses)
    # Q1 = The value at the 25% mark (cheap houses)
    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1

    # The Rule: Anything above (Q3 + 1.5 * IQR) is an outlier
    upper_limit = Q3 + 1.5 * IQR
    
    # Create a new column to label them
    df['is_outlier'] = df['price'] > upper_limit
    
    # Count them
    num_outliers = df['is_outlier'].sum()
    print(f"\n--- OUTLIER STATISTICS ---")
    print(f"Upper Limit Price: ${int(upper_limit):,}")
    print(f"Total Houses: {len(df)}")
    print(f"Number of Outliers: {num_outliers}")
    print(f"Percentage of Outliers: {(num_outliers/len(df))*100:.2f}%")

    # 3. Visualizing the Outliers (Scatter Plot)
    plt.figure(figsize=(10, 6))
    
    # We plot Area vs Price
    # Hue='is_outlier' will color them differently automatically
    sns.scatterplot(data=df, x='area', y='price', hue='is_outlier', palette={False: 'blue', True: 'red'}, s=100)
    
    plt.title('Outlier Detection: Price vs Area', fontsize=15)
    plt.xlabel('Area (sq ft)')
    plt.ylabel('Price')
    plt.axhline(y=upper_limit, color='green', linestyle='--', label=f'Outlier Threshold (${int(upper_limit):,})')
    plt.legend(title='Is Outlier?')
    
    # Save plot
    os.makedirs('plots', exist_ok=True)
    plt.savefig(os.path.join('plots', '5_outliers_scatter.png'))
    print("✅ Saved plot: plots/5_outliers_scatter.png")

if __name__ == "__main__":
    analyze_outliers()