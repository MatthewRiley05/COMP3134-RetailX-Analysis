# Customer Segmentation Clustering

## Required Packages

Install the required Python packages using pip:

```bash
pip install pandas numpy scikit-learn scipy plotly
```

## How to Run

1. **Ensure you have the required data files in the parent directory:**
   - `customers_15.csv`
   - `sales_15.csv`
   - `products_15.csv`

2. **Run the clustering script:**
   ```bash
   python customer_segment_clustering.py
   ```

3. **Output files generated:**
   - `clustering_features.csv` - Engineered features used for clustering
   - `cluster_ready_data.csv` - Scaled and processed data ready for clustering
   - `clustered_customers.csv` - Customer assignments to clusters
   - `cluster_profiles.csv` - Statistical profiles of each cluster
   - `segment_recommendations.csv` - Marketing recommendations for each segment

## What the Script Does

The clustering analysis performs the following steps:
1. Loads customer, sales, and product data
2. Engineers RFM (Recency, Frequency, Monetary) features and other customer metrics
3. Scales and preprocesses the data
4. Performs K-Means clustering to identify customer segments
5. Analyzes and profiles each cluster
6. Generates actionable marketing recommendations for each segment
