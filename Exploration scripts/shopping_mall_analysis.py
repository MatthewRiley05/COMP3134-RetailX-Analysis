import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("sales_15.csv")
def count_items(products):
    if isinstance(products, str):
        cleaned = products.strip('"')
        return len(cleaned.split(',')) if ',' in cleaned else 1
    return 1

# Calculate sales volume by mall
df['Items_Count'] = df['Product id list'].apply(count_items)
mall_sales = df.groupby('Shopping mall')['Items_Count'].sum().sort_values(ascending=False)

# Print results
print("SALES VOLUME BY SHOPPING MALL")
print("=" * 35)
for mall, volume in mall_sales.items():
    print(f"{mall}: {volume:,} products")

print("-" * 35)
print(f"TOTAL: {mall_sales.sum():,} products")

# Create bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(mall_sales.index, mall_sales.values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 20,
             f'{height:,}', ha='center', va='bottom', fontweight='bold')

# Customize the chart
plt.title('Sales Volume by Shopping Mall', fontsize=14, fontweight='bold')
plt.xlabel('Shopping Mall', fontweight='bold')
plt.ylabel('Number of Products Sold', fontweight='bold')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()