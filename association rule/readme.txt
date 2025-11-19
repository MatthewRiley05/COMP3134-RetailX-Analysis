# RetailX Market Basket Analysis: Association Rules

## Overview
This repository contains a Python script for market basket analysis on RetailX's sales data using association rules. It computes all possible category combinations (itemsets) with supports, highlights those with good support (>=0.1), and generates rules with good confidence (>=0.5). This informs promotion strategies, per COMP3134 project (associations for data relationships).

Key features:
- Simplified transaction processing for readability ("human-coded" style).
- Manual implementation (no mlxtend) for broad compatibility.
- All 31 itemsets shown with highlights.
- Rules filtered/sorted, with clean indexing.
- Visualizations: Bar (colored by support quality), scatter, network graph.
- Outputs CSVs and PNGs.

## Dependencies
- Python 3.x
- Libraries: `pandas`, `itertools` (standard), `matplotlib`, `networkx`
  
Install extras if needed:
pip install matplotlib networkx
text## Dataset
Uses `sales_15.csv` (transactions) and `products_15.csv` (categories). Place in script directory.

## How to Run
1. Clone:
git clone <repo-url>
cd <repo-folder>
text2. Run:
python association_rules_analysis.py
textOutputs:
- Console: All itemsets (highlighted), rules (highlighted, sorted).
- `all_itemsets.csv`: All combinations with supports/highlights.
- `association_rules.csv`: Filtered rules.
- PNGs for visuals.

Example (truncated):
All Possible Itemsets Sorted by Support:
itemsets_str  support            highlight
Clothing     0.581784 Good Support (>=0.1)
...
Association Rules Sorted by Lift (with Confidence >=0.5):
antecedents consequents   support  confidence      lift                highlight
Groceries   Clothing    0.277158    0.732933  1.259801  Good Confidence (>=0.5)
...
text## Outputs
- **all_itemsets.csv & association_rules.csv**: Data for analysis.
- **all_itemsets.png**: Bar of top 20 itemsets (green: good support, red: low).
  ![Top Itemsets](all_itemsets.png)
- **rules_scatter.png**: Scatter of rules.
  ![Rules Scatter](rules_scatter.png)
- **rules_network.png**: Network of rules.
  ![Rules Network](rules_network.png)

## Interpretation
- **Itemsets**: 14/31 have good support; singles like Clothing dominate, pairs like Books-Toys common.
- **Rules**: 6 strong rules; high-lift like Groceries → Clothing for bundles.
- **Visuals**: Bar highlights viable combos; scatter/network show rule strengths/connections.