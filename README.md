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

I got 99.99% prediction accuracy on training set and 99.98% on testing set
   
### 3. Build discount optimization
  - using minimize() function from scipy to find the discount that maximizes profit. Since we want to maximize profit, we convert Maximize profit to Minimize (–profit)
  - Input to minimize(): discount value x[0]
  - Objective function: f(x) to minimize -profit by calculating predicted quantity using regression coefficients
  - Computes final price from discount: Calculates profit = (final price – cost) × quantity then returns –profit (because we want to maximize it)
  - Constraints:
    
    a. Discount must be between 0.5% and 45%
    
    b. Final price must be ≥ cost

### 4. Apply discount to random forest model
