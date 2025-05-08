## Use case

Build discount optimization for a manufacturing company that sells drinks. Here is my approach:

1. Predict the quantity sold with a regression model using random forest
2. Build a discount optimization function using scipy.optimization's minimize() for each wholesaler and product
3. Once we get the optimized discount from scipy, apply those discounts to Random Forest model to predict quantity, revenue, and profit with those optimized discounts

End-user could:
1. Simulate a discount by hardcoding (the user chooses what number of the discount they want to apply) and directly applies the hardcoded discount to the Random Forest model
2. Observed the optimized discount for each wholesaler and product
3. Apply the optimized discount to the Random Forest model

## Stacks

- Language: Python
- Stacks: RandomForestRegressor, Scipy.optimization with Minimize() function


## Project flow

### 1. Feature engineering
  - Create wholesaler-product column pair wholesaler-product column pair because we want to introduce the sales pattern for each wholesaler and product
  - Create profit margin column
  - Filter data so each wholesaler-product pair have more than 10 POs (Purchase Order) to introduce more variations to the ML model
  - Create price elasticity feature

### 2. Predict Quantity Sold with RandomForestRegressor.

I got 99.98% prediction accuracy on training set and 99.97% on testing set. We get the predicted quantity with the historical discount. After that, we want to predict the quantity sold when we change the discount.
   
### 3. Build discount optimization
  - using minimize() function from scipy to find the discount that maximizes profit. Since we want to maximize profit, we convert Maximize profit to Minimize (–profit)
  - Leveraging Linear Regression and Minimize() function to get the optimized discount
  - Input to minimize(): discount value x[0]
  - Objective function: f(x) to minimize -profit by calculating predicted quantity using regression coefficients
  - Computes final price from discount: Calculates profit = (final price – cost) × quantity then returns –profit (because we want to maximize it)
  - Constraints:
    
    a. Discount must be between 0.5% and 45%
    
    b. Final price must be ≥ cost
    
  - The result is a table with these information: past price, past quantity, past discount, past profit, then optimized discount

### 4. Apply discount to random forest model

We could either hard coding the discount percentage, or applying directly through the optimized discount
