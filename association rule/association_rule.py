import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import networkx as nx

#Load the sales and products datasets
sales = pd.read_csv('sales_15.csv')
products = pd.read_csv('products_15.csv')

#Create a dictionary mapping product IDs to categories
product_to_categories = dict(zip(products['Product id'], products['Category']))

#Prepare transactions for each invoice and get unique categories from the product list
transactions = []
for row in sales['Product id list']:
    prods = row.split(',')  #Split the product IDs string into a list
    categories = []  
    seen_categories = set()  #Use a set to track unique categories
    for p in prods:
        p_clean = p.strip()  
        if p_clean in product_to_categories:  
            cat = product_to_categories[p_clean]  #Get the category
            if cat not in seen_categories:  #Add only if not already seen (unique)
                categories.append(cat)
                seen_categories.add(cat)
    if categories:  #If there are categories add to transactions
        transactions.append(sorted(categories))  #Sort

#Use TransactionEncoder to convert transactions into a one-hot encoded DataFrame
te = TransactionEncoder()
te_change = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_change, columns=te.columns_)

#Applying Apriori algorithm to find frequent itemsets with minimum support = 0.1
frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)  # Sort for better visualization

#generating and updating association rules with minimum confidence =0.5 (using confidence as the metric)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

rules = rules.sort_values(by='lift', ascending=False)

#Print the rules (antecedents, consequents, support, confidence, lift)
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

#Save rules to CSV for further analysis or reporting
rules.to_csv('association_rules.csv', index=False)

# Visualization 1: Bar chart for top 10 frequent itemsets by support
frequent_itemsets['itemsets_str'] = frequent_itemsets['itemsets'].apply(lambda x: ', '.join(list(x)))
plt.figure(figsize=(12, 8))
plt.barh(frequent_itemsets['itemsets_str'][:10], frequent_itemsets['support'][:10])
plt.xlabel('Support')
plt.title('Top 10 Frequent Itemsets by Support')
plt.gca().invert_yaxis()  # Highest support on top
plt.savefig('frequent_itemsets.png')
plt.close()

# Visualization 2: Scatter plot for rules (confidence vs lift, size by support)
plt.figure(figsize=(10, 6))
plt.scatter(rules['confidence'], rules['lift'], s=rules['support'] * 10000, alpha=0.7, color='b')
plt.xlabel('Confidence')
plt.ylabel('Lift')
plt.title('Association Rules: Confidence vs Lift (Bubble Size = Support)')
plt.grid(True)
plt.savefig('rules_scatter.png')
plt.close()

# Visualization 3: Network graph for rules (nodes as itemsets, edges as rules with lift weights)
plt.figure(figsize=(12, 12))
graph = nx.DiGraph()
for row in rules.itertuples():
    ante = ', '.join(list(row.antecedents))
    cons = ', '.join(list(row.consequents))
    graph.add_edge(ante, cons, weight=row.lift)
position = nx.spring_layout(graph)
nx.draw(graph, position, with_labels=True, node_color='lightblue', node_size=3000, font_size=10, arrows=True)
edge_labels = nx.get_edge_attributes(graph, 'weight')
nx.draw_networkx_edge_labels(graph, position, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()})
plt.title('Association Rules Network Graph')
plt.savefig('rules_network.png')
plt.close()

print("Visualizations saved: frequent_itemsets.png, rules_scatter.png, rules_network.png")
