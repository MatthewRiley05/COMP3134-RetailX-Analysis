"""
Customer Clustering - Optimized ETL & K-Means Segmentation
Transforms sales data into RFM metrics, location/category proportions, and actionable segments
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from scipy.stats import mstats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ===========================
# 1. LOAD DATA
# ===========================
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (project root)
project_root = os.path.dirname(script_dir)

print("Loading data...")
customers = pd.read_csv(os.path.join(project_root, "customers_15.csv"))
sales = pd.read_csv(os.path.join(project_root, "sales_15.csv"))
products = pd.read_csv(os.path.join(project_root, "products_15.csv"))

# ===========================
# 2. EXPLODE THE BASKET
# ===========================
print("Exploding product baskets...")
# Split comma-separated Product id list into rows
sales_exploded = sales.copy()
sales_exploded["Product id"] = sales_exploded["Product id list"].str.split(",")
sales_exploded = sales_exploded.explode("Product id")
sales_exploded["Product id"] = sales_exploded["Product id"].str.strip()

# Join with products to get Category and Price
sales_with_products = sales_exploded.merge(products, on="Product id", how="left")

# ===========================
# 3. INVOICE TOTALS
# ===========================
print("Calculating invoice totals...")
# Sum Price per Invoice no to get invoice value
invoice_totals = (
    sales_with_products.groupby(
        ["Invoice no", "Customer id", "Invoice date", "Shopping mall"]
    )
    .agg({"Price": "sum"})
    .reset_index()
)
invoice_totals.rename(columns={"Price": "Invoice Value"}, inplace=True)

# ===========================
# 4. TIME & RECENCY
# ===========================
print("Computing recency...")
# Parse Invoice date (DD/MM/YYYY)
invoice_totals["Invoice date"] = pd.to_datetime(
    invoice_totals["Invoice date"], format="%d/%m/%Y"
)

# Find the maximum date in the dataset
max_date = invoice_totals["Invoice date"].max()

# Compute Recency per customer = days since last purchase
customer_recency = (
    invoice_totals.groupby("Customer id").agg({"Invoice date": "max"}).reset_index()
)
customer_recency["Recency"] = (max_date - customer_recency["Invoice date"]).dt.days

# ===========================
# 5. LOCATION PROPORTIONS
# ===========================
print("Calculating location proportions...")
# Count invoices per Shopping mall per customer
location_counts = (
    invoice_totals.groupby(["Customer id", "Shopping mall"])
    .size()
    .reset_index(name="Count")
)

# Pivot to get malls as columns
location_pivot = location_counts.pivot(
    index="Customer id", columns="Shopping mall", values="Count"
).fillna(0)

# Convert to proportions (share of visits)
location_proportions = location_pivot.div(location_pivot.sum(axis=1), axis=0)
location_proportions.columns = [f"Mall_{col}" for col in location_proportions.columns]

# ===========================
# 6. CATEGORY MIX
# ===========================
print("Calculating category spend proportions...")
# Sum spend by category per customer
category_spend = (
    sales_with_products.groupby(["Customer id", "Category"])
    .agg({"Price": "sum"})
    .reset_index()
)

# Pivot to get categories as columns
category_pivot = category_spend.pivot(
    index="Customer id", columns="Category", values="Price"
).fillna(0)

# Convert to spend proportions
category_proportions = category_pivot.div(category_pivot.sum(axis=1), axis=0)
category_proportions.columns = [
    f"Category_{col}" for col in category_proportions.columns
]

# ===========================
# 7. RFM METRICS
# ===========================
print("Computing RFM metrics...")
# Aggregate by Customer id
rfm = (
    invoice_totals.groupby("Customer id")
    .agg(
        {
            "Invoice no": "count",  # Frequency
            "Invoice Value": "sum",  # Monetary
        }
    )
    .reset_index()
)

rfm.rename(
    columns={"Invoice no": "Frequency", "Invoice Value": "Monetary"}, inplace=True
)

# Add Recency
rfm = rfm.merge(
    customer_recency[["Customer id", "Recency"]], on="Customer id", how="left"
)

# ===========================
# 8. COMBINE ALL FEATURES
# ===========================
print("Combining all features...")
# Start with customer base
cluster_data = customers[["Customer id", "Gender", "Age", "Payment method"]].copy()

# Merge RFM
cluster_data = cluster_data.merge(rfm, on="Customer id", how="left")

# Merge location proportions
cluster_data = cluster_data.merge(location_proportions, on="Customer id", how="left")

# Merge category proportions
cluster_data = cluster_data.merge(category_proportions, on="Customer id", how="left")

# FILTER OUT CUSTOMERS WITH NO PURCHASE HISTORY
# These are customers in the database but with no transactions (Frequency = NaN)
customers_before = len(cluster_data)
cluster_data = cluster_data[cluster_data["Frequency"].notna()].copy()
customers_after = len(cluster_data)
customers_removed = customers_before - customers_after

if customers_removed > 0:
    print(f"\n⚠️  Filtered out {customers_removed} customers with no purchase history")
    print(f"   (These are registered customers who haven't made any purchases yet)")
    print(f"   Clustering will be performed on {customers_after} active customers")

# Fill any remaining NaN values with 0 (for proportions where customer bought from some but not all malls/categories)
cluster_data.fillna(0, inplace=True)

# ===========================
# 9. SELECT CLUSTERING FEATURES
# ===========================
print("\nSelecting clustering features (behavioral focus)...")

# Core RFM features
rfm_features = ["Recency", "Frequency", "Monetary"]

# Category proportion features (drop one to avoid perfect collinearity)
category_cols = [col for col in cluster_data.columns if col.startswith("Category_")]
category_features = category_cols[:-1]  # Drop last category column
dropped_category = category_cols[-1] if category_cols else None

# Mall proportion features (drop one to avoid perfect collinearity)
mall_cols = [col for col in cluster_data.columns if col.startswith("Mall_")]
mall_features = mall_cols[:-1]  # Drop last mall column
dropped_mall = mall_cols[-1] if mall_cols else None

# Optional: Age (will be z-scored for scale compatibility)
age_scaler = StandardScaler()
cluster_data["Age_scaled"] = age_scaler.fit_transform(cluster_data[["Age"]])
demographic_features = ["Age_scaled"]

# Combine all clustering features
clustering_features = (
    rfm_features + category_features + mall_features + demographic_features
)

# Hold-out features for profiling (not used in clustering)
holdout_features = ["Gender", "Payment method"]

print(f"\n✓ Clustering features selected ({len(clustering_features)} total):")
print(f"  - Core RFM: {rfm_features}")
print(f"  - Category mix: {category_features}")
if dropped_category:
    print(f"    (dropped {dropped_category} to avoid collinearity)")
print(f"  - Location mix: {mall_features}")
if dropped_mall:
    print(f"    (dropped {dropped_mall} to avoid collinearity)")
print(f"  - Demographics: {demographic_features} (z-scored)")
print("\n✓ Hold-out features for post-clustering profiling:")
print(f"  {holdout_features}")

# Create clustering dataset
X_cluster = cluster_data[clustering_features].copy()

print(f"\n✓ Clustering matrix shape: {X_cluster.shape}")
print("\nSample of clustering features:")
print(X_cluster.head())

# ===========================
# 10. SAVE OUTPUTS
# ===========================
# Save full cluster-ready data with all features
output_file = os.path.join(script_dir, "cluster_ready_data.csv")
cluster_data.to_csv(output_file, index=False)
print(f"\n✓ Full dataset saved to {output_file}")

# Save clustering feature matrix
clustering_matrix_file = os.path.join(script_dir, "clustering_features.csv")
clustering_output = cluster_data[["Customer id"] + clustering_features].copy()
clustering_output.to_csv(clustering_matrix_file, index=False)
print(f"✓ Clustering features saved to {clustering_matrix_file}")

print("\nDataset summary:")
print(f"  Total customers: {len(cluster_data)}")
print(f"  Clustering features: {len(clustering_features)}")
print(f"  Holdout features: {len(holdout_features)}")

# ===========================
# 11. TRANSFORM FEATURES FOR CLUSTERING
# ===========================
print("\n" + "=" * 60)
print("FEATURE TRANSFORMATION")
print("=" * 60)

# Create a copy for transformation
X_transformed = X_cluster.copy()

# Log-transform RFM features to tame right skew
print("\n1. Log-transforming RFM features (log1p)...")
X_transformed["Recency"] = np.log1p(X_transformed["Recency"])
X_transformed["Frequency"] = np.log1p(X_transformed["Frequency"])
X_transformed["Monetary"] = np.log1p(X_transformed["Monetary"])

# Winsorize extreme spend to avoid outlier destabilization
print("2. Winsorizing extreme values (top 1%)...")
for col in ["Recency", "Frequency", "Monetary"]:
    X_transformed[col] = mstats.winsorize(X_transformed[col], limits=[0, 0.01])

# Standardize all features (z-score)
print("3. Standardizing all features (z-score)...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_transformed)
X_scaled_df = pd.DataFrame(
    X_scaled, columns=X_transformed.columns, index=X_transformed.index
)

print(f"\n✓ Transformed feature matrix shape: {X_scaled_df.shape}")
print("\nTransformed feature statistics:")
print(X_scaled_df.describe().round(2))

# ===========================
# 12. K SELECTION
# ===========================
print("\n" + "=" * 60)
print("K SELECTION (Elbow & Calinski-Harabasz)")
print("=" * 60)

k_range = range(2, 11)
inertias = []
ch_scores = []
silhouette_scores = []

print("\nTesting K values from 2 to 10...")
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    inertias.append(kmeans.inertia_)
    ch_scores.append(calinski_harabasz_score(X_scaled, labels))
    silhouette_scores.append(silhouette_score(X_scaled, labels))

    print(
        f"  K={k}: Inertia={kmeans.inertia_:.0f}, CH={ch_scores[-1]:.2f}, Silhouette={silhouette_scores[-1]:.3f}"
    )

# Create elbow plot
fig_elbow = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=(
        "Elbow Method (Inertia)",
        "Calinski-Harabasz Score",
        "Silhouette Score",
    ),
)

fig_elbow.add_trace(
    go.Scatter(
        x=list(k_range),
        y=inertias,
        mode="lines+markers",
        name="Inertia",
        line=dict(color="blue", width=3),
        marker=dict(size=8),
    ),
    row=1,
    col=1,
)

fig_elbow.add_trace(
    go.Scatter(
        x=list(k_range),
        y=ch_scores,
        mode="lines+markers",
        name="CH Score",
        line=dict(color="green", width=3),
        marker=dict(size=8),
    ),
    row=1,
    col=2,
)

fig_elbow.add_trace(
    go.Scatter(
        x=list(k_range),
        y=silhouette_scores,
        mode="lines+markers",
        name="Silhouette",
        line=dict(color="orange", width=3),
        marker=dict(size=8),
    ),
    row=1,
    col=3,
)

fig_elbow.update_xaxes(title_text="Number of Clusters (K)", row=1, col=1)
fig_elbow.update_xaxes(title_text="Number of Clusters (K)", row=1, col=2)
fig_elbow.update_xaxes(title_text="Number of Clusters (K)", row=1, col=3)
fig_elbow.update_yaxes(title_text="Inertia", row=1, col=1)
fig_elbow.update_yaxes(title_text="CH Score", row=1, col=2)
fig_elbow.update_yaxes(title_text="Silhouette Score", row=1, col=3)

fig_elbow.update_layout(
    height=400,
    showlegend=False,
    template="plotly_white",
    title_text="K-Means Cluster Evaluation Metrics",
)

fig_elbow.show()

# Recommend K based on CH score (maximum)
optimal_k = list(k_range)[np.argmax(ch_scores)]
print(f"\n✓ Recommended K based on Calinski-Harabasz: {optimal_k}")

# ===========================
# 13. RUN K-MEANS CLUSTERING
# ===========================
print("\n" + "=" * 60)
print(f"K-MEANS CLUSTERING (K={optimal_k})")
print("=" * 60)

# Run final K-means
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
cluster_data["Cluster"] = kmeans_final.fit_predict(X_scaled)

print(f"\n✓ Clustering complete!")
print(f"\nCluster sizes:")
print(cluster_data["Cluster"].value_counts().sort_index())

# ===========================
# 14. PCA VISUALIZATION
# ===========================
print("\n" + "=" * 60)
print("PCA VISUALIZATION")
print("=" * 60)

# PCA to 2D for visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

cluster_data["PCA1"] = X_pca[:, 0]
cluster_data["PCA2"] = X_pca[:, 1]

print(f"\n✓ PCA variance explained:")
print(f"  PC1: {pca.explained_variance_ratio_[0]:.1%}")
print(f"  PC2: {pca.explained_variance_ratio_[1]:.1%}")
print(f"  Total: {pca.explained_variance_ratio_.sum():.1%}")

# Create PCA scatter plot
fig_pca = px.scatter(
    cluster_data,
    x="PCA1",
    y="PCA2",
    color="Cluster",
    title=f"Customer Segments (K-Means, K={optimal_k})",
    labels={
        "PCA1": f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
        "PCA2": f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
    },
    color_discrete_sequence=px.colors.qualitative.Bold,
    hover_data=["Customer id", "Frequency", "Monetary", "Recency"],
)

fig_pca.update_traces(
    marker=dict(size=8, opacity=0.7, line=dict(width=0.5, color="white"))
)
fig_pca.update_layout(template="plotly_white", height=600, font=dict(size=12))

fig_pca.show()

# ===========================
# 15. CLUSTER PROFILING
# ===========================
print("\n" + "=" * 60)
print("CLUSTER PROFILING")
print("=" * 60)

# Calculate mean values for each cluster (on original scale)
profile_features = (
    ["Recency", "Frequency", "Monetary", "Age"] + category_cols + mall_cols
)

profile_df = cluster_data.groupby("Cluster")[profile_features].mean()
cluster_sizes = cluster_data["Cluster"].value_counts().sort_index()

print("\nCluster Profiles (original scale):")
print("\n--- RFM Metrics ---")
print(profile_df[["Recency", "Frequency", "Monetary"]].round(2))

print("\n--- Demographics ---")
print(profile_df[["Age"]].round(1))

print("\n--- Category Preferences (proportions) ---")
print(profile_df[category_cols].round(3))

print("\n--- Mall Preferences (proportions) ---")
print(profile_df[mall_cols].round(3))

# Gender and Payment method distribution per cluster
print("\n--- Gender Distribution (%) ---")
gender_dist = pd.crosstab(
    cluster_data["Cluster"], cluster_data["Gender"], normalize="index"
)
print((gender_dist * 100).round(1))

print("\n--- Payment Method Distribution (%) ---")
payment_dist = pd.crosstab(
    cluster_data["Cluster"], cluster_data["Payment method"], normalize="index"
)
print((payment_dist * 100).round(1))

# ===========================
# 16. AUTO-NAME CLUSTERS (Business-Friendly Labels)
# ===========================
print("\n" + "=" * 60)
print("CLUSTER NAMING & INTERPRETATION")
print("=" * 60)


def name_cluster(cluster_id, profile_row, size):
    """Generate business-friendly cluster names based on behavioral characteristics"""

    recency = profile_row["Recency"]
    frequency = profile_row["Frequency"]
    monetary = profile_row["Monetary"]

    # Identify dominant category
    cat_cols_list = [col for col in profile_row.index if col.startswith("Category_")]
    if cat_cols_list:
        dominant_category = max(cat_cols_list, key=lambda x: profile_row[x])
        dominant_cat_name = dominant_category.replace("Category_", "")
        cat_strength = profile_row[dominant_category]
    else:
        dominant_cat_name = "General"
        cat_strength = 0

    # Identify dominant mall
    mall_cols_list = [col for col in profile_row.index if col.startswith("Mall_")]
    if mall_cols_list:
        dominant_mall = max(mall_cols_list, key=lambda x: profile_row[x])
        mall_name = dominant_mall.replace("Mall_", "")
    else:
        mall_name = "Mixed"

    # Classify based on RFM
    is_high_value = monetary > profile_df["Monetary"].median()
    is_frequent = frequency > profile_df["Frequency"].median()
    is_recent = recency < profile_df["Recency"].median()

    # Generate name
    if is_high_value and is_frequent and is_recent:
        tier = "VIP"
    elif is_high_value and is_frequent:
        tier = "Loyal High-Spender"
    elif is_frequent and is_recent:
        tier = "Frequent Shopper"
    elif is_high_value:
        tier = "High-Value"
    elif is_recent and frequency > 1:
        tier = "Active"
    else:
        tier = "Casual"

    # Add category descriptor if strong preference (>40%)
    if cat_strength > 0.4:
        category_desc = f"{dominant_cat_name}-Focused"
    else:
        category_desc = "Omni-Category"

    # Build full name
    cluster_name = f"{tier} {category_desc}"

    return cluster_name


# Generate names and characteristics
cluster_names = {}
cluster_characteristics = {}

print("\nCluster Names & Key Characteristics:\n")
for cluster_id in sorted(profile_df.index):
    profile_row = profile_df.loc[cluster_id]
    size = cluster_sizes[cluster_id]
    size_pct = (size / len(cluster_data)) * 100

    name = name_cluster(cluster_id, profile_row, size)
    cluster_names[cluster_id] = name

    # Extract key characteristics
    chars = {
        "Size": f"{size} customers ({size_pct:.1f}%)",
        "Avg Spend": f"HKD {profile_row['Monetary']:.0f}",
        "Avg Frequency": f"{profile_row['Frequency']:.1f} visits",
        "Recency": f"{profile_row['Recency']:.0f} days",
        "Avg Age": f"{profile_row['Age']:.0f} years",
    }

    # Top category
    cat_cols_list = [col for col in profile_row.index if col.startswith("Category_")]
    if cat_cols_list:
        top_cat = max(cat_cols_list, key=lambda x: profile_row[x])
        chars["Top Category"] = (
            f"{top_cat.replace('Category_', '')} ({profile_row[top_cat] * 100:.0f}%)"
        )

    # Preferred mall
    mall_cols_list = [col for col in profile_row.index if col.startswith("Mall_")]
    if mall_cols_list:
        top_mall = max(mall_cols_list, key=lambda x: profile_row[x])
        chars["Preferred Mall"] = (
            f"{top_mall.replace('Mall_', '')} ({profile_row[top_mall] * 100:.0f}%)"
        )

    cluster_characteristics[cluster_id] = chars

    print(f"📊 Cluster {cluster_id}: {name}")
    print(f"   └─ {chars['Size']}")
    print(
        f"   └─ {chars['Avg Spend']} | {chars['Avg Frequency']} | Last visit: {chars['Recency']}"
    )
    print(
        f"   └─ {chars.get('Top Category', 'N/A')} | {chars.get('Preferred Mall', 'N/A')}"
    )
    print()

# Add cluster names to main dataframe
cluster_data["Cluster_Name"] = cluster_data["Cluster"].map(cluster_names)

# ===========================
# 17. STRATEGIC RECOMMENDATIONS
# ===========================
print("=" * 60)
print("STRATEGIC RECOMMENDATIONS")
print("=" * 60)

# Identify target segment (highest CLV potential)
profile_df["CLV_Score"] = (
    profile_df["Monetary"] * profile_df["Frequency"] / (profile_df["Recency"] + 1)
)
target_cluster = profile_df["CLV_Score"].idxmax()
target_name = cluster_names[target_cluster]

print(f"\n🎯 TARGET SEGMENT: Cluster {target_cluster} - {target_name}")
print(f"   Why: Highest CLV potential (Spend × Frequency / Recency)")
print(f"   Size: {cluster_characteristics[target_cluster]['Size']}")
print(f"   Value: {cluster_characteristics[target_cluster]['Avg Spend']}")
print()

# Generate acquisition & retention tactics for each cluster
print("\n💡 RECOMMENDED TACTICS BY SEGMENT:\n")

for cluster_id in sorted(profile_df.index):
    name = cluster_names[cluster_id]
    profile_row = profile_df.loc[cluster_id]
    chars = cluster_characteristics[cluster_id]

    # Get payment preference
    payment_pref = payment_dist.loc[cluster_id].idxmax()
    payment_pct = payment_dist.loc[cluster_id].max() * 100

    # Get top category
    cat_cols_list = [col for col in profile_row.index if col.startswith("Category_")]
    if cat_cols_list:
        top_cat = max(cat_cols_list, key=lambda x: profile_row[x])
        top_cat_name = top_cat.replace("Category_", "")
        top_cat_pct = profile_row[top_cat] * 100
    else:
        top_cat_name = "General"
        top_cat_pct = 0

    # Get mall preference
    mall_cols_list = [col for col in profile_row.index if col.startswith("Mall_")]
    if mall_cols_list:
        top_mall = max(mall_cols_list, key=lambda x: profile_row[x])
        top_mall_name = top_mall.replace("Mall_", "")
    else:
        top_mall_name = "Mixed"

    print(
        f"{'🎯 ' if cluster_id == target_cluster else ''}Cluster {cluster_id}: {name}"
    )
    print(f"{'=' * 60}")

    # ACQUISITION TACTIC
    print("📈 ACQUISITION:")
    if payment_pref == "Mobile Payment" and payment_pct > 50:
        print(f"   → App-First Campaign: {payment_pct:.0f}% use mobile payment")
        print(
            f"      • Offer: Download app for HKD 50 welcome voucher on {top_cat_name}"
        )
        print(f"      • Channel: Social media ads targeting {top_mall_name} area")
    elif top_cat_pct > 40:
        print(f"   → Category-Targeted Ads: {top_cat_pct:.0f}% prefer {top_cat_name}")
        print(f"      • Offer: 15% off first {top_cat_name} purchase + free shipping")
        print(f"      • Channel: Google Shopping & Facebook lookalike audiences")
    else:
        print(f"   → Omnichannel Campaign: Diverse shopping behavior")
        print(f"      • Offer: HKD 100 off first purchase over HKD 500")
        print(f"      • Channel: Broad digital + in-mall signage at {top_mall_name}")

    print()

    # RETENTION TACTIC
    print("🔄 RETENTION:")
    if (
        profile_row["Frequency"] > profile_df["Frequency"].median()
        and profile_row["Monetary"] > profile_df["Monetary"].median()
    ):
        print(
            f"   → VIP Loyalty Program: High frequency ({profile_row['Frequency']:.1f}) & spend (HKD {profile_row['Monetary']:.0f})"
        )
        print(f"      • Offer: Early access to sales + free gift after 3 purchases")
        print(
            f"      • Trigger: Automated after reaching HKD {profile_row['Monetary'] * 0.8:.0f} spend"
        )
    elif profile_row["Recency"] > profile_df["Recency"].median():
        print(
            f"   → Win-Back Campaign: {profile_row['Recency']:.0f} days since last visit"
        )
        print(f"      • Offer: 'We miss you' - 20% off {top_cat_name} valid 7 days")
        print(f"      • Channel: Email + SMS if opted in")
    elif top_cat_pct > 40:
        print(
            f"   → Category Bundle Deals: Strong {top_cat_name} preference ({top_cat_pct:.0f}%)"
        )
        print(f"      • Offer: Buy 2 {top_cat_name} items, get 10% off total basket")
        print(f"      • Timing: Monthly on {payment_pref} paydays")
    else:
        print(f"   → Points-Based Loyalty: Moderate engagement")
        print(f"      • Offer: 1 point per HKD 10, redeem at 100 points")
        print(f"      • Channel: {payment_pref}-enabled digital wallet integration")

    print()

# ===========================
# 18. POSITIONING MAP
# ===========================
print("\n" + "=" * 60)
print("POSITIONING MAP VISUALIZATION")
print("=" * 60)

# Create positioning map: Spend vs Frequency
fig_positioning = px.scatter(
    profile_df.reset_index(),
    x="Frequency",
    y="Monetary",
    text="Cluster",
    size=[cluster_sizes[i] for i in profile_df.index],
    color="Cluster",
    title="Customer Segment Positioning Map<br><sub>Bubble size = Segment size | Strategic positioning: Spend vs. Visit Frequency</sub>",
    labels={
        "Frequency": "Average Purchase Frequency (visits)",
        "Monetary": "Average Monetary Value (HKD)",
    },
    color_discrete_sequence=px.colors.qualitative.Bold,
    size_max=60,
)

# Add cluster names as annotations
for idx, row in profile_df.reset_index().iterrows():
    cluster_id = row["Cluster"]
    fig_positioning.add_annotation(
        x=row["Frequency"],
        y=row["Monetary"],
        text=f"<b>{cluster_names[cluster_id]}</b><br>{cluster_sizes[cluster_id]} customers",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="gray",
        ax=0,
        ay=-50 if idx % 2 == 0 else 50,
        font=dict(size=10, color="black"),
        bgcolor="white",
        opacity=0.9,
    )

fig_positioning.update_traces(
    marker=dict(line=dict(width=2, color="white")),
    textposition="top center",
    textfont=dict(size=14, color="white"),
)

fig_positioning.update_layout(
    template="plotly_white", height=700, font=dict(size=12), showlegend=False
)

# Add quadrant lines
median_freq = profile_df["Frequency"].median()
median_mon = profile_df["Monetary"].median()

fig_positioning.add_hline(
    y=median_mon, line_dash="dash", line_color="gray", opacity=0.5
)
fig_positioning.add_vline(
    x=median_freq, line_dash="dash", line_color="gray", opacity=0.5
)

fig_positioning.show()

# ===========================
# 19. SAVE FINAL RESULTS
# ===========================
print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

# Save clustered data
clustered_file = os.path.join(script_dir, "clustered_customers.csv")
cluster_data.to_csv(clustered_file, index=False)
print(f"\n✓ Clustered customer data saved to {clustered_file}")

# Save cluster profiles with names
profile_df["Cluster_Name"] = profile_df.index.map(cluster_names)
profile_df["Size"] = profile_df.index.map(cluster_sizes)
profile_df["Size_Pct"] = (profile_df["Size"] / len(cluster_data) * 100).round(1)

profile_file = os.path.join(script_dir, "cluster_profiles.csv")
profile_df.to_csv(profile_file)
print(f"✓ Cluster profiles saved to {profile_file}")

# Save detailed recommendations
recommendations = []
for cluster_id in sorted(profile_df.index):
    recommendations.append(
        {
            "Cluster": cluster_id,
            "Name": cluster_names[cluster_id],
            "Size": cluster_characteristics[cluster_id]["Size"],
            "Avg_Spend": cluster_characteristics[cluster_id]["Avg Spend"],
            "Avg_Frequency": cluster_characteristics[cluster_id]["Avg Frequency"],
            "Is_Target_Segment": "✓" if cluster_id == target_cluster else "",
        }
    )

recommendations_df = pd.DataFrame(recommendations)
recommendations_file = os.path.join(script_dir, "segment_recommendations.csv")
recommendations_df.to_csv(recommendations_file, index=False)
print(f"✓ Segment recommendations saved to {recommendations_file}")

print("\n" + "=" * 60)
print("CLUSTERING COMPLETE!")
print("=" * 60)
print("\n📁 Files generated:")
print(f"  1. {clustered_file}")
print(f"  2. {profile_file}")
print(f"  3. {recommendations_file}")
print(f"\n🎯 Target Segment: Cluster {target_cluster} - {target_name}")
print("📊 Ready for presentation! Interactive visualizations displayed above.")
