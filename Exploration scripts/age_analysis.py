import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_age_sales_graphs(sales_file, customers_file, products_file):
    """
    Create bar graphs showing how age ranges affect sales volume
    """
    
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
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
                        'Invoice date': row['Invoice date'],
                        'Shopping mall': row['Shopping mall'],
                        'Price': product_price
                    })
    
    expanded_sales_df = pd.DataFrame(expanded_sales)
    
    # Create age groups with ranges
    bins = [0, 25, 40, 65, 100]
    labels = ['Under 25', '25-40', '40-65', 'Above 65']
    customers_df['Age Group'] = pd.cut(customers_df['Age'], bins=bins, labels=labels, right=False)
    
    # Merge sales data with customer information
    merged_df = pd.merge(expanded_sales_df, customers_df, on='Customer id', how='left')
    
    # Remove any rows with NaN values after merge
    merged_df = merged_df.dropna()
    
    # Calculate sales volume metrics by age group
    sales_by_age = merged_df.groupby('Age Group').agg({
        'Invoice no': 'nunique',  # Number of transactions
        'Price': 'sum',           # Total revenue
        'Customer id': 'nunique'  # Number of unique customers
    }).rename(columns={
        'Invoice no': 'Transaction Count',
        'Price': 'Total Revenue',
        'Customer id': 'Unique Customers'
    })
    
    # Handle division by zero and ensure finite values
    sales_by_age = sales_by_age.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Calculate average transaction value (with safety check)
    sales_by_age['Average Transaction Value'] = np.where(
        sales_by_age['Transaction Count'] > 0,
        sales_by_age['Total Revenue'] / sales_by_age['Transaction Count'],
        0
    )
    
    # Calculate sales per customer (with safety check)
    sales_by_age['Revenue per Customer'] = np.where(
        sales_by_age['Unique Customers'] > 0,
        sales_by_age['Total Revenue'] / sales_by_age['Unique Customers'],
        0
    )
    
    # Ensure all values are finite
    sales_by_age = sales_by_age.replace([np.inf, -np.inf], 0).fillna(0)
    
    # Create the graphs
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Sales Analysis by Age Groups', fontsize=16, fontweight='bold')
    
    # Plot 1: Total Revenue by Age Group
    bars1 = ax1.bar(sales_by_age.index, sales_by_age['Total Revenue'], 
                   color='skyblue', edgecolor='navy', alpha=0.7)
    ax1.set_title('Total Revenue by Age Group', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Age Group')
    ax1.set_ylabel('Total Revenue (HKD)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars (only if value is finite and positive)
    for bar in bars1:
        height = bar.get_height()
        if np.isfinite(height) and height >= 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:,.0f}',
                    ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Transaction Count by Age Group
    bars2 = ax2.bar(sales_by_age.index, sales_by_age['Transaction Count'],
                   color='lightcoral', edgecolor='darkred', alpha=0.7)
    ax2.set_title('Number of Transactions by Age Group', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Age Group')
    ax2.set_ylabel('Number of Transactions')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars (only if value is finite and positive)
    for bar in bars2:
        height = bar.get_height()
        if np.isfinite(height) and height >= 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:,.0f}',
                    ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Average Transaction Value by Age Group
    bars3 = ax3.bar(sales_by_age.index, sales_by_age['Average Transaction Value'],
                   color='lightgreen', edgecolor='darkgreen', alpha=0.7)
    ax3.set_title('Average Transaction Value by Age Group', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Age Group')
    ax3.set_ylabel('Average Value (HKD)')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars (only if value is finite and positive)
    for bar in bars3:
        height = bar.get_height()
        if np.isfinite(height) and height >= 0:
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:,.0f}',
                    ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Revenue per Customer by Age Group
    bars4 = ax4.bar(sales_by_age.index, sales_by_age['Revenue per Customer'],
                   color='gold', edgecolor='darkorange', alpha=0.7)
    ax4.set_title('Revenue per Customer by Age Group', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Age Group')
    ax4.set_ylabel('Revenue per Customer (HKD)')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars (only if value is finite and positive)
    for bar in bars4:
        height = bar.get_height()
        if np.isfinite(height) and height >= 0:
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:,.0f}',
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

sales_file = "sales_15.csv"
customers_file = "customers_15.csv"  
products_file = "products_15.csv"  

create_age_sales_graphs(sales_file, customers_file, products_file)