import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_category_revenue_chart(sales_file, customers_file, products_file):
    """
    Create bar chart showing which product category generates the most revenue
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
                product_category = products_df.loc[products_df['Product id'] == product_id, 'Category'].values[0]
                # Ensure price is finite
                if np.isfinite(product_price):
                    expanded_sales.append({
                        'Invoice no': row['Invoice no'],
                        'Customer id': row['Customer id'],
                        'Product id': product_id,
                        'Price': product_price,
                        'Category': product_category
                    })
    
    expanded_sales_df = pd.DataFrame(expanded_sales)
    
    # Calculate revenue by product category
    revenue_by_category = expanded_sales_df.groupby('Category').agg({
        'Price': 'sum',           # Total revenue
        'Invoice no': 'nunique',  # Number of transactions
        'Product id': 'count'     # Number of items sold
    }).rename(columns={
        'Price': 'Total Revenue',
        'Invoice no': 'Transaction Count',
        'Product id': 'Items Sold'
    })
    
    # Sort by total revenue (descending)
    revenue_by_category = revenue_by_category.sort_values('Total Revenue', ascending=False)
    
    # Calculate additional metrics
    revenue_by_category['Average Transaction Value'] = revenue_by_category['Total Revenue'] / revenue_by_category['Transaction Count']
    revenue_by_category['Average Item Price'] = revenue_by_category['Total Revenue'] / revenue_by_category['Items Sold']
    
    # Create a comprehensive bar chart
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Revenue Analysis by Product Category', fontsize=16, fontweight='bold')
    
    # Color palette for different categories
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot 1: Total Revenue by Category
    bars1 = ax1.bar(revenue_by_category.index, revenue_by_category['Total Revenue'], 
                   color=colors, alpha=0.7, edgecolor='black')
    ax1.set_title('Total Revenue by Product Category', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Product Category')
    ax1.set_ylabel('Total Revenue (HKD)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:,.0f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Plot 2: Number of Transactions by Category
    bars2 = ax2.bar(revenue_by_category.index, revenue_by_category['Transaction Count'],
                   color=colors, alpha=0.7, edgecolor='black')
    ax2.set_title('Number of Transactions by Category', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Product Category')
    ax2.set_ylabel('Number of Transactions')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:,.0f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Plot 3: Average Transaction Value by Category
    bars3 = ax3.bar(revenue_by_category.index, revenue_by_category['Average Transaction Value'],
                   color=colors, alpha=0.7, edgecolor='black')
    ax3.set_title('Average Transaction Value by Category', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Product Category')
    ax3.set_ylabel('Average Value (HKD)')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:,.0f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Plot 4: Number of Items Sold by Category
    bars4 = ax4.bar(revenue_by_category.index, revenue_by_category['Items Sold'],
                   color=colors, alpha=0.7, edgecolor='black')
    ax4.set_title('Number of Items Sold by Category', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Product Category')
    ax4.set_ylabel('Items Sold')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:,.0f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.show()

create_category_revenue_chart(
    sales_file="sales_15.csv",
    customers_file="customers_15.csv", 
    products_file= "products_15.csv"
)
