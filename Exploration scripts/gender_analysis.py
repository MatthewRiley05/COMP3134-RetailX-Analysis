import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_gender_revenue_chart(sales_file, customers_file, products_file):
    """
    Create bar chart showing which gender generates more revenue

    """
    plt.style.use('seaborn-v0_8')
    
    # Load datasets
    sales_df = pd.read_csv(sales_file)
    customers_df = pd.read_csv(customers_file)
    products_df = pd.read_csv(products_file)
    
    # Data cleaning - handle missing values
    sales_df = sales_df.dropna()
    customers_df = customers_df.dropna()
    products_df = products_df.dropna()
    
    # Expand sales data to have one row per product
    expanded_sales = []
    for idx, row in sales_df.iterrows():
        product_ids = str(row['Product id list']).split(',')
        for product_id in product_ids:
            product_id = product_id.strip()
            if product_id in products_df['Product id'].values:
                product_price = products_df.loc[products_df['Product id'] == product_id, 'Price'].values[0]
                # Ensure price is finite
                if np.isfinite(product_price):
                    expanded_sales.append({
                        'Invoice no': row['Invoice no'],
                        'Customer id': row['Customer id'],
                        'Product id': product_id,
                        'Price': product_price
                    })
    
    expanded_sales_df = pd.DataFrame(expanded_sales)
    
    # Merge sales data with customer information
    merged_df = pd.merge(expanded_sales_df, customers_df, on='Customer id', how='left')
    
    # Remove any rows with NaN values after merge
    merged_df = merged_df.dropna()
    
    # Calculate revenue by gender
    revenue_by_gender = merged_df.groupby('Gender').agg({
        'Price': 'sum',           # Total revenue
        'Invoice no': 'nunique',  # Number of transactions
        'Customer id': 'nunique'  # Number of unique customers
    }).rename(columns={
        'Price': 'Total Revenue',
        'Invoice no': 'Transaction Count',
        'Customer id': 'Unique Customers'
    })
    
    # Calculate additional metrics
    revenue_by_gender['Average Transaction Value'] = revenue_by_gender['Total Revenue'] / revenue_by_gender['Transaction Count']
    revenue_by_gender['Revenue per Customer'] = revenue_by_gender['Total Revenue'] / revenue_by_gender['Unique Customers']
    
    # Create a single comprehensive bar chart
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Revenue Analysis by Gender', fontsize=16, fontweight='bold')
    colors = ['#1f77b4','#ff7f0e']  
    
    # Plot 1: Total Revenue by Gender
    bars1 = ax1.bar(revenue_by_gender.index, revenue_by_gender['Total Revenue'], 
                   color=colors, alpha=0.7, edgecolor='black')
    ax1.set_title('Total Revenue by Gender', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Gender')
    ax1.set_ylabel('Total Revenue (HKD)')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:,.0f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Plot 2: Number of Transactions by Gender
    bars2 = ax2.bar(revenue_by_gender.index, revenue_by_gender['Transaction Count'],
                   color=colors, alpha=0.7, edgecolor='black')
    ax2.set_title('Number of Transactions by Gender', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Gender')
    ax2.set_ylabel('Number of Transactions')
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:,.0f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Plot 3: Average Transaction Value by Gender
    bars3 = ax3.bar(revenue_by_gender.index, revenue_by_gender['Average Transaction Value'],
                   color=colors, alpha=0.7, edgecolor='black')
    ax3.set_title('Average Transaction Value by Gender', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Gender')
    ax3.set_ylabel('Average Value (HKD)')
    
    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:,.0f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Plot 4: Revenue per Customer by Gender
    bars4 = ax4.bar(revenue_by_gender.index, revenue_by_gender['Revenue per Customer'],
                   color=colors, alpha=0.7, edgecolor='black')
    ax4.set_title('Revenue per Customer by Gender', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Gender')
    ax4.set_ylabel('Revenue per Customer (HKD)')
    
    # Add value labels on bars
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:,.0f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.show()

# Example usage with different file paths:
if __name__ == "__main__":
    create_gender_revenue_chart(
        sales_file= "sales_15.csv",
        customers_file= "customers_15.csv",
        products_file= "products_15.csv"
    )
    
    