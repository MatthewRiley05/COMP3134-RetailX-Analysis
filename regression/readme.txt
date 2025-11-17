How to run:
1. Install Python 3, pandas and scikit-learn
2. Run the regression_main.py file
3. Results will be printed to the console and output files will be made in the "regression" folder

Output file explanation:
1. feature_importances_classification.csv (Predicting product ID list)
num__Price (0.4739): Price has the highest importance, indicating it is the main driver influencing which product is bought. Clear strategies related to pricing can directly affect sales.
num__Age (0.1231): Age has a moderate importance, indicating that the demographic profile of the customer can inform product recommendations.
num__Month (0.0830): May suggest that purchasing patterns vary according to the month, which may inform seasonal marketing strategies.
num__Day (0.1077): Shows variations in purchasing behavior based on the day of the month. 

cat__Gender_M (0.0253): Gender might not significantly affect the product ID predictions when compared to other numeric features.
cat__Payment method_x (e.g., Credit Card, Mobile Payment):  Relatively low importance scores, indicating that payment method doesn't strongly drive product choice.
cat__Shopping mall_x (e.g., MK, ST, TKO), cat__Day of Week_x (e.g., Monday, Saturday, etc.): These scores are among the lowest, suggesting the day of the week and shopping location does not drastically impact product choice.

2. regression_coefficients_price.csv (Predicting price)
A positive value indicates that the dependent variable (product price) is expected to increase, while a negative value means it is expected to decrease.

Numeric Features
num__Age (0.0487):
For every additional year of customer age, the product price is expected to increase by approximately 0.0487 units. This minor effect suggests that older customers may be associated with slightly higher-priced purchases, potentially due to their preferences or purchasing power.

num__Year (19.79):
Each additional year results in an increase in product price by about 19.79 units. This substantial effect could imply that either product prices have increased over the years due to inflation or that the products purchased in later years are generally of higher value.

num__Month (2.39):
For each additional month, the price is expected to rise by about 2.39 units, indicating seasonal pricing trends or perhaps that later months offer higher-value products.

num__Day (-0.93):
Each additional day in the month leads to a decrease in predicted price by approximately 0.93 units. This could suggest that as the month progresses, prices tend to drop, possibly due to sales strategies or stock clearance.
Categorical Features

cat__Gender_M (-8.28):
Being male is associated with a decrease in product price by about 8.28 units compared to females. This could imply that products targeted at male customers are generally priced lower.

cat__Payment method_Credit Card (2.51):
Using a credit card is associated with an increase in price of approximately 2.51 units, which might reflect that higher-end products are marketed more frequently to customers who can leverage credit.

cat__Payment method_Mobile Payment (14.75):
Customers using mobile payment methods are expected to pay about 14.75 units more. This significant positive coefficient suggests that mobile payments might be associated with premium or more convenience-driven purchases.

cat__Shopping mall_MK (-20.10):
Shopping at the Mong Kok mall is linked to a decrease in price by around 20.10 units. This indicates that products in this mall are priced lower compared to others.

cat__Shopping mall_ST (-20.58):
Shopping at the Sha Tin mall results in an even larger decrease in expected price (approximately 20.58 units), further indicating that this mall caters to more budget-conscious consumers.

cat__Shopping mall_TKO (-9.52):
The Tseung Kwan O mall is associated with a smaller price decrease of about 9.52 units compared to others, indicating that it has a relatively higher price point than MK and ST.

cat__Day of Week_Monday (-24.59):
Products purchased on Monday are expected to be priced 24.59 units lower. This sizable negative coefficient may suggest discounts or special pricing strategies implemented at the start of the week.

cat__Day of Week_Saturday (-8.28):
Saturday purchases are linked to a decrease of about 8.28 units, indicating potential sales or lower prices during weekend shopping.

cat__Day of Week_Sunday (-40.18):
Sunday shows the most significant decrease of 40.18 units, implying that products bought on Sundays are much cheaper, possibly due to end-of-week discounts.

cat__Day of Week_Thursday (-40.29):
Similar to Sunday, shopping on Thursday also predicts a significant reduction in price by about 40.29 units, indicating potential promotional pricing strategies on this day.

cat__Day of Week_Tuesday (-28.23):
Products purchased on Tuesday are expected to be about 28.23 units cheaper, suggesting that this day also promotes lower pricing.

cat__Day of Week_Wednesday (-8.76):
Shopping on Wednesday results in a slight decrease in price (approximately 8.76 units), but much less significant than the other days.


3. regression_coefficients_age.csv (Predicting age)
Numeric Features
num__Price (2.82e-05):
A very small positive coefficient indicates that for each unit increase in product price, the predicted age increases by approximately 0.0000282 years (or about 0.01 days). This suggests that higher-priced products might be associated, albeit very weakly, with slightly older customers. However, the effect is negligible.

num__Year (0.3353):
For each additional year in the dataset's timeline, the predicted age increases by about 0.3353 years. This could reflect that over time, customers in the dataset have aged, or it could indicate that the dataset's customer base has shifted to older demographics in more recent years.

num__Month (-0.0718):
Each additional month results in a slight decrease in predicted age of about 0.0718 years. This suggests a potential trend where, in later months, the customers may be younger, possibly due to seasonal purchasing patterns.

num__Day (-0.0201):
A one-day increase corresponds to a small decrease in predicted age of approximately 0.0201 years. This aligns with the earlier point indicating that shopping later in the month could coincide with younger customers.


Categorical Features
cat__Payment method_Credit Card (0.3454):
Customers who pay with a credit card are predicted to be approximately 0.3454 years older than those using other payment methods. This could suggest that older individuals are more likely to use credit cards for purchases, possibly reflecting their financial habits or access to credit.

cat__Payment method_Mobile Payment (-1.32):
Using mobile payment methods is associated with a decrease in predicted age by about 1.32 years. This indicates that mobile payment users are generally younger, which may reflect familiarity with technology among younger demographics.

cat__Shopping mall_MK (-0.1942):
Shopping at mall MK correlates with a decrease in predicted age of approximately 0.1942 years. This implies that younger shoppers frequent this mall compared to others.

cat__Shopping mall_ST (-0.8134):
Shopping at mall ST is associated with a decrease in predicted age by about 0.8134 years. This mall likely attracts a younger demographic as well.

cat__Shopping mall_TKO (-0.8314):
Shopping at TKO shows an even greater predicted decrease in age (about 0.8314 years), suggesting this mall has a significantly younger clientele.

cat__Day of Week_Monday (-0.1530):
Shopping on Monday is linked to a predicted decrease in age of about 0.1530 years, indicating that younger customers may be more active in shopping at the beginning of the week.

cat__Day of Week_Saturday (1.0892):
Conversely, shopping on Saturday is associated with an increase in predicted age of about 1.0892 years. This suggests that older customers are more likely to shop on weekends.

cat__Day of Week_Sunday (-0.1826):
Purchasing on Sunday results in a predicted decrease in age of about 0.1826 years, indicating a younger customer base shopping on Sundays.

cat__Day of Week_Thursday (-0.6560):
Shopping on Thursday predicts a decrease in age of approximately 0.6560 years, showing that this day tends to attract younger shoppers as well.

cat__Day of Week_Tuesday (-0.7698):
Similar to Thursday, Tuesday also predicts a decrease in age of about 0.7698 years, once again showing that younger customers are shopping on this day.

cat__Day of Week_Wednesday (-1.7742):
This day shows the most considerable decrease in predicted age, suggesting that customers shopping on Wednesdays are significantly younger, with a predicted decrease of about 1.77 years.