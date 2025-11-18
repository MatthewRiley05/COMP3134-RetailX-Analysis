import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load csv data
sales_data = pd.read_csv('sales_15.csv')
customers_data = pd.read_csv('customers_15.csv')
products_data = pd.read_csv('products_15.csv')

# Preprocess Sales Data
# Split product ids into separate rows
sales_data['Product id list'] = sales_data['Product id list'].str.split(',')
sales_data = sales_data.explode('Product id list')

# Convert Invoice Date to datetime format
sales_data['Invoice date'] = pd.to_datetime(sales_data['Invoice date'], format='%d/%m/%Y')

# Extract date features
sales_data['Year'] = sales_data['Invoice date'].dt.year
sales_data['Month'] = sales_data['Invoice date'].dt.month
sales_data['Day'] = sales_data['Invoice date'].dt.day
sales_data['Day of Week'] = sales_data['Invoice date'].dt.day_name()

# Drop (delete) the original Invoice Date column
sales_data.drop(columns=['Invoice date'], inplace=True)

# Merge data
merged_data = sales_data.merge(customers_data, on='Customer id')
merged_data = merged_data.merge(products_data, left_on='Product id list', right_on='Product id')
print(merged_data.head())

# # # # # # # # # 
# 1st regression: 
# # # # # # # # # 
# dependent = product id list, 
# independent = gender, age, payment method, price, shopping mall, invoice date
y_classification  = merged_data['Product id list']  # Dependent variable
X_classification  = merged_data[['Gender', 'Age', 'Payment method', 'Price', 'Shopping mall', 'Year', 'Month', 'Day', 'Day of Week']]  # Independent variables

# OneHotEncoder for categorical features
categorical_features = ['Gender', 'Payment method', 'Shopping mall', 'Day of Week']
numeric_features = ['Age', 'Price', 'Year', 'Month', 'Day']

# Create a preprocessing pipeline
class_preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# Create a pipeline with preprocessing and the Random Forest Classifier
classification_pipeline = Pipeline(steps=[
    ('preprocessor', class_preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Split the data into training and test sets for classification
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_classification, y_classification, test_size=0.2, random_state=42)

# Fit the model for classification
classification_pipeline.fit(X_train_class, y_train_class)

# Make predictions for classification
y_pred_class = classification_pipeline.predict(X_test_class)

# Evaluate the classification model
accuracy = accuracy_score(y_test_class, y_pred_class)
report = classification_report(y_test_class, y_pred_class, zero_division=0)

print(f'Classification Accuracy: {accuracy}')
print('Classification Report:\n', report)

# # # # # # # # # 
# 2nd regression: 
# # # # # # # # # 
# Define independent and dependent variables for the second regression
y_regression = merged_data['Price']  # Dependent variable
X_regression = merged_data[['Gender', 'Age', 'Payment method', 'Year', 'Month', 'Day', 'Shopping mall', 'Day of Week']]  # Independent variables (excluding Product ID list)

# Create a preprocessing pipeline for regression
regression_preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', ['Age', 'Year', 'Month', 'Day']),
        ('cat', OneHotEncoder(drop='first'), ['Gender', 'Payment method', 'Shopping mall', 'Day of Week'])
    ])

# Create a pipeline with preprocessing and the Linear Regression model
regression_pipeline = Pipeline(steps=[
    ('preprocessor', regression_preprocessor),
    ('regressor', LinearRegression())
])

# Split the data into training and test sets for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_regression, y_regression, test_size=0.2, random_state=42)

# Fit the model for regression
regression_pipeline.fit(X_train_reg, y_train_reg)

# Make predictions for regression
y_pred_reg = regression_pipeline.predict(X_test_reg)

# Evaluate the regression model
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)


# # # # # # # # # 
# 3rd regression: 
# # # # # # # # # 
# Count the occurrences of each Product ID
product_counts = merged_data['Product id list'].value_counts().reset_index()
product_counts.columns = ['Product id', 'Quantity Sold']

# Merge count data back into merged_data
merged_data = merged_data.merge(product_counts, left_on='Product id list', right_on='Product id', how='left')

# Calculate total revenue generated from each product sold (Quantity Sold * Price)
merged_data['Total Revenue'] = merged_data['Quantity Sold'] * merged_data['Price']

# Define independent and dependent variables for the new regression
y_regression_age = merged_data['Age']  # Dependent variable
X_regression_age = merged_data[['Total Revenue']]  # Independent variable based on product sales

# Create a preprocessing pipeline for age regression
regression_pipeline_age = Pipeline(steps=[
    ('preprocessor', ColumnTransformer(
        transformers=[('num', 'passthrough', ['Total Revenue'])]
    )),
    ('regressor', LinearRegression())
])

# Split the data into training and test sets for age regression
X_train_reg_age, X_test_reg_age, y_train_reg_age, y_test_reg_age = train_test_split(X_regression_age, y_regression_age, test_size=0.2, random_state=42)

# Fit the model for age regression
regression_pipeline_age.fit(X_train_reg_age, y_train_reg_age)

# Make predictions for age regression
y_pred_reg_age = regression_pipeline_age.predict(X_test_reg_age)

# Evaluate the age regression model
mse_age = mean_squared_error(y_test_reg_age, y_pred_reg_age)
r2_age = r2_score(y_test_reg_age, y_pred_reg_age)

print(f'Regression Mean Squared Error (Age): {mse_age}')
print(f'Regression R-squared (Age): {r2_age}')

# Visualization of the relationship
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test_reg_age['Total Revenue'], y=y_test_reg_age, label='Actual Age', color='blue', alpha=0.6)
sns.scatterplot(x=X_test_reg_age['Total Revenue'], y=y_pred_reg_age, label='Predicted Age', color='orange', alpha=0.6)

# Fit a regression line
sns.regplot(x=X_test_reg_age['Total Revenue'], y=y_pred_reg_age, scatter=False, color='red', label='Regression Line')

plt.title('Customer Age Prediction Based on Total Revenue')
plt.xlabel('Total Revenue')
plt.ylabel('Customer Age')
plt.legend()
plt.grid(True)
plt.show()


# Save feature importances for (1st) classification model
importances_class = classification_pipeline.named_steps['classifier'].feature_importances_
features_class = classification_pipeline.named_steps['preprocessor'].get_feature_names_out()

feature_importance_class_df = pd.DataFrame({'Feature': features_class, 'Importance': importances_class})
feature_importance_class_df.to_csv('regression/feature_importances_classification.csv', index=False)

# Save coefficients for (2nd) price regression model
coefficients_reg = pd.Series(regression_pipeline.named_steps['regressor'].coef_, index=regression_pipeline.named_steps['preprocessor'].get_feature_names_out())
coefficients_reg.to_csv('regression/regression_coefficients_price.csv')

# Save coefficients for (3rd) amount regression model
coefficients_reg_age = pd.Series(regression_pipeline_age.named_steps['regressor'].coef_, 
                                  index=regression_pipeline_age.named_steps['preprocessor'].get_feature_names_out())
coefficients_reg_age.to_csv('regression/regression_coefficients_age.csv', index=True)
