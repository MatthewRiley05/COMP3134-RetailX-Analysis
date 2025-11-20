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

# Load Data
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

print("Loading data...")
customers = pd.read_csv(os.path.join(project_root, "customers_15.csv"))
sales = pd.read_csv(os.path.join(project_root, "sales_15.csv"))
products = pd.read_csv(os.path.join(project_root, "products_15.csv"))

# Separate Sections
print("Exploding product baskets...")
sales_exploded = sales.copy()
sales_exploded["Product id"] = sales_exploded["Product id list"].str.split(",")
sales_exploded = sales_exploded.explode("Product id")
sales_exploded["Product id"] = sales_exploded["Product id"].str.strip()

sales_with_products = sales_exploded.merge(products, on="Product id", how="left")

# Calculate Invoice Totals
print("Calculating invoice totals...")
invoice_totals = (
    sales_with_products.groupby(
        ["Invoice no", "Customer id", "Invoice date", "Shopping mall"]
    )
    .agg({"Price": "sum"})
    .reset_index()
)
invoice_totals.rename(columns={"Price": "Invoice Value"}, inplace=True)

# Compute Recency
print("Computing recency...")
invoice_totals["Invoice date"] = pd.to_datetime(
    invoice_totals["Invoice date"], format="%d/%m/%Y"
)

max_date = invoice_totals["Invoice date"].max()

customer_recency = (
    invoice_totals.groupby("Customer id").agg({"Invoice date": "max"}).reset_index()
)
customer_recency["Recency"] = (max_date - customer_recency["Invoice date"]).dt.days

# Location Proportions
print("Calculating location proportions...")
location_counts = (
    invoice_totals.groupby(["Customer id", "Shopping mall"])
    .size()
    .reset_index(name="Count")
)

location_pivot = location_counts.pivot(
    index="Customer id", columns="Shopping mall", values="Count"
).fillna(0)

location_proportions = location_pivot.div(location_pivot.sum(axis=1), axis=0)
location_proportions.columns = [f"Mall_{col}" for col in location_proportions.columns]

# Category Spend Proportions
print("Calculating category spend proportions...")
category_spend = (
    sales_with_products.groupby(["Customer id", "Category"])
    .agg({"Price": "sum"})
    .reset_index()
)

category_pivot = category_spend.pivot(
    index="Customer id", columns="Category", values="Price"
).fillna(0)

category_proportions = category_pivot.div(category_pivot.sum(axis=1), axis=0)
category_proportions.columns = [
    f"Category_{col}" for col in category_proportions.columns
]

# RFM Metrics
print("Computing RFM metrics...")
rfm = (
    invoice_totals.groupby("Customer id")
    .agg(
        {
            "Invoice no": "count",
            "Invoice Value": "sum",
        }
    )
    .reset_index()
)

rfm.rename(
    columns={"Invoice no": "Frequency", "Invoice Value": "Monetary"}, inplace=True
)

rfm = rfm.merge(
    customer_recency[["Customer id", "Recency"]], on="Customer id", how="left"
)

# Combine All Features
print("Combining all features...")
cluster_data = customers[["Customer id", "Gender", "Age", "Payment method"]].copy()

cluster_data = cluster_data.merge(rfm, on="Customer id", how="left")

cluster_data = cluster_data.merge(location_proportions, on="Customer id", how="left")

cluster_data = cluster_data.merge(category_proportions, on="Customer id", how="left")

customers_before = len(cluster_data)
cluster_data = cluster_data[cluster_data["Frequency"].notna()].copy()
customers_after = len(cluster_data)
customers_removed = customers_before - customers_after

if customers_removed > 0:
    print(f"\nFiltered out {customers_removed} customers with no purchase history")
    print(f"Clustering will be performed on {customers_after} active customers")

cluster_data.fillna(0, inplace=True)

# Select Clustering Features
print("\nSelecting clustering features...")

rfm_features = ["Recency", "Frequency", "Monetary"]

category_cols = [col for col in cluster_data.columns if col.startswith("Category_")]
category_features = category_cols[:-1]
dropped_category = category_cols[-1] if category_cols else None

mall_cols = [col for col in cluster_data.columns if col.startswith("Mall_")]
mall_features = mall_cols[:-1]
dropped_mall = mall_cols[-1] if mall_cols else None

age_scaler = StandardScaler()
cluster_data["Age_scaled"] = age_scaler.fit_transform(cluster_data[["Age"]])
demographic_features = ["Age_scaled"]

clustering_features = (
    rfm_features + category_features + mall_features + demographic_features
)

holdout_features = ["Gender", "Payment method"]

print(f"\nClustering features selected ({len(clustering_features)} total):")
print(f"  - Core RFM: {rfm_features}")
print(f"  - Category mix: {category_features}")
if dropped_category:
    print(f"    (dropped {dropped_category} to avoid collinearity)")
print(f"  - Location mix: {mall_features}")
if dropped_mall:
    print(f"    (dropped {dropped_mall} to avoid collinearity)")
print(f"  - Demographics: {demographic_features}")
print("\nHold-out features for post-clustering profiling:")
print(f"  {holdout_features}")

X_cluster = cluster_data[clustering_features].copy()

print(f"\nClustering matrix shape: {X_cluster.shape}")

# Save Prepared Data
output_file = os.path.join(script_dir, "cluster_ready_data.csv")
cluster_data.to_csv(output_file, index=False)
print(f"\nFull dataset saved to {output_file}")

clustering_matrix_file = os.path.join(script_dir, "clustering_features.csv")
clustering_output = cluster_data[["Customer id"] + clustering_features].copy()
clustering_output.to_csv(clustering_matrix_file, index=False)
print(f"Clustering features saved to {clustering_matrix_file}")

print("\nDataset summary:")
print(f"  Total customers: {len(cluster_data)}")
print(f"  Clustering features: {len(clustering_features)}")
print(f"  Holdout features: {len(holdout_features)}")

# Transform Features
print("\n" + "=" * 60)
print("FEATURE TRANSFORMATION")
print("=" * 60)

X_transformed = X_cluster.copy()

print("\n1. Log-transforming RFM features (log1p)...")
X_transformed["Recency"] = np.log1p(X_transformed["Recency"])
X_transformed["Frequency"] = np.log1p(X_transformed["Frequency"])
X_transformed["Monetary"] = np.log1p(X_transformed["Monetary"])

print("2. Winsorizing extreme values (top 1%)...")
for col in ["Recency", "Frequency", "Monetary"]:
    X_transformed[col] = mstats.winsorize(X_transformed[col], limits=[0, 0.01])

print("3. Standardizing all features (z-score)...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_transformed)
X_scaled_df = pd.DataFrame(
    X_scaled, columns=X_transformed.columns, index=X_transformed.index
)

print(f"\nTransformed feature matrix shape: {X_scaled_df.shape}")
print("\nTransformed feature statistics:")
print(X_scaled_df.describe().round(2))

# K-Selection
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

optimal_k = list(k_range)[np.argmax(ch_scores)]
print(f"\nRecommended K based on Calinski-Harabasz: {optimal_k}")

# K-Means Clustering
print("\n" + "=" * 60)
print(f"K-MEANS CLUSTERING (K={optimal_k})")
print("=" * 60)

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
cluster_data["Cluster"] = kmeans_final.fit_predict(X_scaled)

print("\nClustering complete!")
print("\nCluster sizes:")
print(cluster_data["Cluster"].value_counts().sort_index())

# PCA Visualization
print("\n" + "=" * 60)
print("PCA VISUALIZATION")
print("=" * 60)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

cluster_data["PCA1"] = X_pca[:, 0]
cluster_data["PCA2"] = X_pca[:, 1]

print("\nPCA variance explained:")
print(f"  PC1: {pca.explained_variance_ratio_[0]:.1%}")
print(f"  PC2: {pca.explained_variance_ratio_[1]:.1%}")
print(f"  Total: {pca.explained_variance_ratio_.sum():.1%}")

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

# Cluster Profiling
print("\n" + "=" * 60)
print("CLUSTER PROFILING")
print("=" * 60)

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

# Positioning Map
print("\n" + "=" * 60)
print("POSITIONING MAP VISUALIZATION")
print("=" * 60)

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

# Save Results
print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

clustered_file = os.path.join(script_dir, "clustered_customers.csv")
cluster_data.to_csv(clustered_file, index=False)
print(f"\nClustered customer data saved to {clustered_file}")

profile_df["Size"] = profile_df.index.map(cluster_sizes)
profile_df["Size_Pct"] = (profile_df["Size"] / len(cluster_data) * 100).round(1)

profile_file = os.path.join(script_dir, "cluster_profiles.csv")
profile_df.to_csv(profile_file)
print(f"Cluster profiles saved to {profile_file}")

print("\n" + "=" * 60)
print("CLUSTERING COMPLETE!")
print("=" * 60)
print("\nFiles generated:")
print(f"  1. {clustered_file}")
print(f"  2. {profile_file}")
