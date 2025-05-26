import pandas as pd
import numpy as np
from category_encoders import TargetEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import joblib

class discountOptimV2(object):
    def __init__(self, filepath):
        self.filepath = filepath
    
    def training_data(self):
        data_purchased = pd.read_csv(self.filepath)
        data_purchased['Product_ID'] = data_purchased['Product_ID'].astype('str')
        self.data_purchased = data_purchased
        self.data_wholesaler = data_purchased.drop_duplicates(subset=['Wholesaler_Product_ID'])[
            ['Wholesaler_ID','Product_ID','Wholesaler_Product_ID','Cluster']]

    def input_new_data(self, wholesaler_id, product_id, base_price, ppn, cost, discount_pct):
        final_price = base_price * (1 - discount_pct)
        margin = final_price - cost
        wholesaler_product_id = f"{wholesaler_id}-{product_id}"
        
        return pd.DataFrame([{
            'Wholesaler_ID': wholesaler_id,
            'Product_ID': product_id,
            'Base_Price': base_price,
            'PPN': ppn,
            'modal_per_pcs_inc_PPN': cost,
            'Final_Price_New': final_price,
            'discount_pct': discount_pct,
            'Wholesaler_Product_ID': wholesaler_product_id,
            'Margin': margin,
            'is_discounted': 1
        }])
    
    def get_predict_quantity(self, trained_pipeline, data):
        data = data.merge(self.data_wholesaler[['Wholesaler_Product_ID','Cluster']], on='Wholesaler_Product_ID', how='left')
        X_pred = data.drop(['Wholesaler_ID', 'Product_ID'], axis=1)
        predicted_quantity = trained_pipeline.predict(X_pred)[0]
        return max(0, predicted_quantity)
    
    def constraint1(self, D_array, base_price, cost):
        D = D_array[0]
        max_discount = max(0, 1 - cost / base_price)
        return min(0.50, max_discount) - D

    def constraint2(self, D_array, base_price, cost):
        D = D_array[0]
        return base_price * (1 - D) - cost

    def objective_function(self, D_array, base_price, cost, data_input, trained_pipeline):
        D = D_array[0]
        
        new_data = self.input_new_data(
            wholesaler_id=data_input['Wholesaler_ID'].values[0],
            product_id=data_input['Product_ID'].values[0],
            base_price=base_price,
            ppn=data_input['PPN'].values[0],
            cost=cost,
            discount_pct=D
        )

        predicted_quantity = self.get_predict_quantity(trained_pipeline, new_data)
        final_price = base_price * (1 - D)
        profit = (final_price - cost) * predicted_quantity

        return -profit  # We negate to maximize
    
    def optimization(self, wholesaler_id, product_id, base_price, ppn, cost, trained_pipeline):
        subset_data_history = self.data_purchased[(self.data_purchased['Wholesaler_ID']==wholesaler_id) & (self.data_purchased['Product_ID']==product_id)]
        subset_data_history = subset_data_history.sort_values(by="Transaction_Date",ascending=True)
        latest_data = subset_data_history.iloc[-1]
        initial_discount = subset_data_history['discount_pct'].mean()

        latest_past_discount = latest_data['discount_pct']
        latest_past_final_price = latest_data['Final_Price_New']
        latest_qty = latest_data['Quantity_Sold']
        latest_profit = latest_data['Total_Profit']
        latest_profit_margin = latest_data['Profit_Margin_%']


        # === Create initial data template ===
        data_input = self.input_new_data(
            wholesaler_id, product_id, base_price, ppn, cost, initial_discount
        )

        # === Constraints ===
        constraints = [
            {"type": "ineq", "fun": lambda x: self.constraint1(x, base_price, cost)},
            {"type": "ineq", "fun": lambda x: self.constraint2(x, base_price, cost)}
        ]

        # === Optimization ===
        result = minimize(
            self.objective_function,
            x0=[initial_discount],
            args=(base_price, cost, data_input, trained_pipeline),
            method="SLSQP",
            bounds=[(0.0, 0.50)],
            constraints=constraints
        )

        # === Results ===
        if result.success:
            optimal_discount = result.x[0]
            optimal_final_price = base_price * (1 - optimal_discount)

            final_data = self.input_new_data(
                wholesaler_id, product_id, base_price, ppn, cost, optimal_discount
            )
            predicted_quantity = self.get_predict_quantity(trained_pipeline, final_data)
            max_profit = (optimal_final_price - cost) * predicted_quantity


            return {
            "optimal_discount": optimal_discount,
            "optimal_final_price": optimal_final_price,
            "predicted_quantity": predicted_quantity,
            "predicted_profit": max_profit,
            "predicted_profit_margin": (optimal_final_price - cost )/ optimal_final_price,
            "latest_past_discount":latest_past_discount,
            "latest_past_final_price":latest_past_final_price,
            "latest_qty":latest_qty,
            "latest_profit":latest_profit,
            "latest_profit_margin":latest_profit_margin
            }

        else:
            print("❌ Optimization failed:", result.message)
    
    def calculate_price_elasticity(self, original_price, original_quantity, optimized_price, optimized_quantity):
        if original_price == 0 or original_quantity == 0:
            return None

        pct_change_quantity = (optimized_quantity - original_quantity) / original_quantity
        pct_change_price = (optimized_price - original_price) / original_price

        if pct_change_price == 0:
            return None

        elasticity = pct_change_quantity / pct_change_price
        return elasticity