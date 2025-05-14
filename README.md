## Use case

Build discount optimization for a manufacturing company that sells drinks. Here is my approach:

1. Predict the quantity sold with a regression model using random forest
2. Build a discount optimization function using scipy.optimization's minimize() for each wholesaler and product
3. Once we get the optimized discount from scipy, apply those discounts to Random Forest model to predict quantity, revenue, and profit with those optimized discounts

End-user could:
1. Simulate a discount by hardcoding (the user chooses what number of the discount they want to apply) and directly applies the hardcoded discount to the Random Forest model
2. Observed the optimized discount for each wholesaler and product
3. Apply the optimized discount to the Random Forest model

## Notable insights

I also applied customer segmentation to segment all wholesalers into 3 clusters, and here are my findings:

### **Cluster 0**

before applying discount optimization:
1. low sales, inconsistent revenue, indicating **small to medium-scale buyers**
2. some transactions are poorly priced leads to negative profit margin
3. low discounts, but there are huge discounts = 51% for such small sales

after applying discount optimization:
1. discounts decreased slightly (mean: 4% → 2%)
2. quantity dropped slightly, but profit increased significantly (~43% increase)
3. for this cluster, the model reduces discounts without hurting sales volume significantly and boosts profit, suggesting optimal margin management

### **Cluster 1**

before applying discount optimization:
1. high sales, massive and consistent revenue and profit, around 79M and 4M respectively, indicating **large wholesalers**
2. moderate discount, some don't get a discount
3. There are cases a wholesaler doesn't get a discount, but the minimum purchase is still high ~36.000 pcs

after applying discount optimization:
1. discounts are already optimized (14%), and further discounting doesn’t help (past discount & optimized discount are same)
2. slightly drop in quantity (~3k sales) and profit (~Rp 2.000 lower than past profit)

*? same discount but drops happened, because that is only a prediction, and the discount decimal might be different*

3. high std devs suggest very diverse customers or pricing structures.
4. this cluster seems like already in the maximum efficiency. Trying further optimization doesn't hurt, and might need further discussion regarding the amount of discount should be given to wholesalers in this cluster

### **Cluster 2**

before applying discount optimization:
1. Total revenue is around 22M and consistent, indicating **mid-volume buyers**
2. Has the highest discount % average  among all clusters, under 75% of data reach almost 11% discount
3. This cluster indicates that discount is needed in order to drive more sales

after applying discount optimization:
1. mall quantity drop (~300 units), but massive profit gain: ~30% increase
2. despite lowering discounts, random forest predicted better profit and almost the same sales.
3. this cluster responds well to careful discount control, and profit can be significantly boosted by small pricing tweaks.


## Stacks

- Language: Python
- Stacks: RandomForestRegressor, Scipy.optimization with Minimize() function


## Project flow


### 1. Feature engineering
  - Create wholesaler-product column pair wholesaler-product column pair because we want to introduce the sales pattern for each wholesaler and product
  - Create profit margin column
  - Filter data so each wholesaler-product pair have more than 10 POs (Purchase Order) to introduce more variations to the ML model
  - Create price elasticity feature
  - Segment wholesaleres into 3 clusters, and use the cluster column as a new feature


### 2. Predict Quantity Sold with RandomForestRegressor.

I got 99.98% prediction accuracy on training set and 99.97% on testing set, calculated by R² adjusted. We get the predicted quantity with the historical discount. After that, we want to predict the quantity sold when we change the discount.

   
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
