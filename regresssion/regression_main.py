import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# Load csv data
sales_data = pd.read_csv('datas/sales_15.csv')
customers_data = pd.read_csv('datas/customers_15.csv')
products_data = pd.read_csv('datas/products_15.csv')

print(sales_data.head())

# Preprocess Sales Data
# Split product ids into separate rows
sales_data['Product id list'] = sales_data['Product id list'].str.split(',')
sales_data = sales_data.explode('Product id list')


# Merge data
merged_data = sales_data.merge(customers_data, on='Customer id')
merged_data = merged_data.merge(products_data, left_on='Product id list', right_on='Product id')
print(merged_data.head())

# 1st test: dependent = product id list, independent: gender, age, payment method, price, shopping mall
y = merged_data['Product id list']  # Dependent variable
X = merged_data[['Gender', 'Age', 'Payment method', 'Price', 'Shopping mall']]  # Independent variables

# OneHotEncoder for categorical features
categorical_features = ['Gender', 'Payment method', 'Shopping mall']
numeric_features = ['Age', 'Price']

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# Create a pipeline with preprocessing and classification model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)

print(f'Accuracy: {accuracy}')
print('Classification Report:\n', report)

# Save model results if needed (e.g., feature importances)
importances = pipeline.named_steps['classifier'].feature_importances_
features = pipeline.named_steps['preprocessor'].get_feature_names_out()

feature_importance_data = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance_data.to_csv('regression_feature_importances/feature_importances_1st_test.csv', index=False)

