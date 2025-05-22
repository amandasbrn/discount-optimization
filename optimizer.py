import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.optimize import minimize

class discountOptimizer(object):
    def __init__(self, filepath):
        self.filepath = filepath
    
    def data_prep(self):
        data = pd.read_csv(self.filepath)
        data_purchased = data.dropna()
        data_purchased['Product_ID'] = data_purchased['Product_ID'].astype('str')

        # if amount = 0, discount_pct must be 0 as well
        data_purchased['discount_pct'] = data_purchased.apply(lambda row: 0 if row['Discount_Amount'] == 0 else row['discount_pct'], axis=1)

        # create a new column wholesaler & product pair
        data_purchased['Wholesaler_Product_ID'] = data_purchased['Wholesaler_ID'] + "-" + data_purchased['Product_ID']

        # create profit columns
        data_purchased['Profit_Margin_%'] = (
            (data_purchased['Final_Price_New'] - data_purchased['modal_per_pcs_inc_PPN'])
            / data_purchased['Final_Price_New'] )

        data_purchased['Profit_Per_Unit'] = data_purchased['Final_Price_New'] - data_purchased['modal_per_pcs_inc_PPN']

        data_purchased['Total_Profit'] = data_purchased['Profit_Per_Unit'] * data_purchased['Quantity_Sold']

        # filter data so each pair has more than 10 purchase orders
        data_purchased = data_purchased.groupby('Wholesaler_Product_ID').filter(lambda x : len(x)>=10)

        # Calculate percentage changes in Quantity Sold and Final Price
        data_purchased['Quantity_Change_%'] = data_purchased.groupby(['Wholesaler_ID', 'Product_ID'])['Quantity_Sold'].pct_change() * 100
        data_purchased['Price_Change_%'] = data_purchased.groupby(['Wholesaler_ID', 'Product_ID'])['Final_Price_New'].pct_change() * 100

        # Calculate Elasticity
        data_purchased['Elasticity'] = data_purchased.apply(
            lambda row: row['Quantity_Change_%'] / row['Price_Change_%'] if row['Price_Change_%'] != 0 else np.nan,
            axis=1
        )

        # Fill NaN values (this will typically happen for the first row in each group)
        data_purchased['Quantity_Change_%'] = data_purchased['Quantity_Change_%'].fillna(0)
        data_purchased['Price_Change_%'] = data_purchased['Price_Change_%'].fillna(0)

        # Cap elasticity to handle extreme outliers
        data_purchased['Elasticity'] = data_purchased['Elasticity'].clip(-10, 10)
        data_purchased['Elasticity'] = data_purchased['Elasticity'].fillna(0)
        data_purchased = data_purchased.drop(['Quantity_Change_%', 'Price_Change_%'],axis=1)

        self.data_ready = data_purchased
        
    
    def modeling(self):
        drop_col = ['Product_Description', 'BOSnetszUomId', 'BOSnetdecUom', 'decUomConversion1', 'decUomConversion2','is_conversion_equal','Product_Desc', 'PO_ID','Wholesaler_ID','Product_ID','depoid', 'Depo_Prod_Unique_ID', 'Discount_Type','Nama_Depo', 'Transaction_Date','Convert_to_box', 'unique_ID', 'Product_Nickname', 'Profit_Margin_%', 'Profit_Per_Unit', 'Final_Price_New','Total_Revenue_New']
        target = ['Quantity_Sold']
        X = self.data_ready.drop(drop_col+target,axis=1)
        y = self.data_ready[target]

        # split
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
        
        # encode
        encoder = TargetEncoder(cols=['Wholesaler_Product_ID'])  # or Wholesaler_Product_ID if you concatenate
        encoder.fit(X_train, y_train)

        X_train = encoder.transform(X_train)
        X_test = encoder.transform(X_test)

        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # r2 = r2_score(y_test, y_pred)
        # r2_adjusted_train = 1 - (1-model.score(X_train, y_train))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)
        # r2_adjusted_test = 1 - (1-model.score(X_test, y_test))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)

        self.drop_col = drop_col
        self.target = target
        self.model = model
        self.encoder = encoder
    
    def optimize_discounts(self):
        results = []

        grouped = self.data_ready.groupby('Wholesaler_Product_ID')

        for key, group in grouped:
            X = group.drop(columns=self.drop_col + self.target)
            y = group['Quantity_Sold']

            if 'Wholesaler_Product_ID' in X.columns:
                encoder = TargetEncoder(cols=['Wholesaler_Product_ID'])
                X = encoder.fit_transform(X, y)

            # use linreg to capture relationship between discount and quantity increase/decrease
            reg = LinearRegression()
            reg.fit(X, y)
            intercept = reg.intercept_
            coef = dict(zip(X.columns, reg.coef_))

            # self.group_models = {}  # New dictionary to store model per group

            # # Inside the loop after fitting:
            # self.group_models[key] = {
            #     'intercept': intercept,
            #     'coef': coef,
            #     'base_price': base_price,
            #     'cost': cost
            # }

            # Extract latest values
            latest_features = X.iloc[-1].to_dict()
            base_price = group['Base_Price'].iloc[-1]
            cost = group['modal_per_pcs_inc_PPN'].iloc[-1]

            # Define profit objective
            def objective(x):
                D = x[0]
                features = latest_features.copy()
                features['discount_pct'] = D
                predicted_quantity = intercept + sum(coef[k] * features[k] for k in coef)
                final_price = base_price * (1 - D)
                profit = (final_price - cost) * predicted_quantity
                return -profit

            # Constraint: max discount and final price >= cost
            max_discount = max(0, 1 - cost / base_price)
            constraints = [
                {"type": "ineq", "fun": lambda x: min(0.50, max_discount) - x[0]},
                {"type": "ineq", "fun": lambda x: base_price * (1 - x[0]) - cost}
            ]

            x0 = [group['discount_pct'].min()]
            bounds = [(0.00, 0.50)]

            opt_result = minimize(objective, x0=x0, constraints=constraints, bounds=bounds, method='SLSQP')

            if opt_result.success:
                wholesaler_id = group['Wholesaler_ID'].iloc[0]
                product_id = group['Product_ID'].iloc[0]
                opt_disc = round(opt_result.x[0], 3)

                features = latest_features.copy()
                features['discount_pct'] = opt_disc
                predicted_quantity = intercept + sum(coef[k] * features[k] for k in coef)

                past_price = group['Final_Price_New'].iloc[0]
                past_discount = group['discount_pct'].iloc[0]
                past_qty = group['Quantity_Sold'].iloc[0]
                past_profit = group['Total_Profit'].iloc[0]
                past_profit_margin = group['Profit_Margin_%'].iloc[0]

                results.append({
                    'Wholesaler_ID': wholesaler_id,
                    'Product_ID': product_id,
                    'past_price': past_price,
                    'past_discount': past_discount,
                    'past_qty': past_qty,
                    'past_profit':past_profit,
                    'past_profit_margin':past_profit_margin,
                    'optimized_discount': opt_disc
                })

        self.optimized_df = pd.DataFrame(results)
        return self.optimized_df
    
    def apply_optimization_to_new_data(self, new_data):
        results = []

        for _, row in new_data.iterrows():
            key = f"{row['Wholesaler_ID']}_{row['Product_ID']}"
            if key not in self.group_models:
                print(f"Missing model for {key}. Skipping.")
                continue

            model_info = self.group_models[key]
            intercept = model_info['intercept']
            coef = model_info['coef']
            base_price = model_info['base_price']
            cost = model_info['cost']

            # Latest features from new data
            features = row.to_dict()

            def objective(x):
                D = x[0]
                features_copy = features.copy()
                features_copy['discount_pct'] = D
                predicted_quantity = intercept + sum(coef.get(k, 0) * features_copy.get(k, 0) for k in coef)
                final_price = base_price * (1 - D)
                profit = (final_price - cost) * predicted_quantity
                return -profit

            max_discount = max(0, 1 - cost / base_price)
            constraints = [
                {"type": "ineq", "fun": lambda x: min(0.50, max_discount) - x[0]},
                {"type": "ineq", "fun": lambda x: base_price * (1 - x[0]) - cost}
            ]
            x0 = [row.get('discount_pct', 0.1)]
            bounds = [(0.0, 0.5)]

            opt_result = minimize(objective, x0=x0, constraints=constraints, bounds=bounds, method='SLSQP')

            if opt_result.success:
                opt_disc = round(opt_result.x[0], 3)
                final_price = base_price * (1 - opt_disc)
                predicted_quantity = intercept + sum(coef.get(k, 0) * features.get(k, 0) for k in coef)
                profit = (final_price - cost) * predicted_quantity

                results.append({
                    'Wholesaler_ID': row['Wholesaler_ID'],
                    'Product_ID': row['Product_ID'],
                    'optimized_discount': opt_disc,
                    'new_final_price': final_price,
                    'predicted_quantity': predicted_quantity,
                    'predicted_profit': profit
                })

            else:
                print(f"Optimization failed for new row {key}: {opt_result.message}")

        return pd.DataFrame(results)

    
    def disc_simulation(self, data, drop_col, target, discount, encoder, model):
        data['discount_pct'] = discount

        # calculate discount amount & final price manually because of new discount %
        data['Discount_Amount'] = (data['Base_Price'] * data['Quantity_Sold']) * data['discount_pct']
        data['Final_Price_New'] = ((data['Base_Price'] * data['Quantity_Sold']) - data['Discount_Amount']) / data['Quantity_Sold']
        data['Profit_Margin_%'] = (data['Final_Price_New'] - data['modal_per_pcs_inc_PPN']) / data['Final_Price_New']
        
        # Apply encodings
        new_data = data.drop(drop_col+ target, axis=1)
        new_data_encoded = encoder.transform(new_data)

        predicted_quantity = model.predict(new_data_encoded)
        data['Predicted_Quantity'] = predicted_quantity
        data['Predicted_Revenue'] = data['Predicted_Quantity'] * data['Final_Price_New']
        data['Predicted_Profit'] = data['Predicted_Quantity'] * (
            data['Final_Price_New'] - data['modal_per_pcs_inc_PPN']
        )

        data = data.rename(columns={
                                        'Quantity_Sold': 'Past_Quantity_Sold',
                                        'Total_Revenue_New': 'Past_Revenue',
                                        'Total_Profit': 'Past_Profit'
                                    })
        data['New_Discount'] = discount

        show_cols = [
        'Wholesaler_ID', 'PO_ID', 'Product_ID', 'Product_Desc',
        'Base_Price',
        'Past_Quantity_Sold', 'Past_Revenue', 'Past_Profit',
        'New_Discount', 'Discount_Amount','Predicted_Quantity', 'Predicted_Revenue', 'Predicted_Profit', 'Profit_Margin_%'
        ]
        return data[show_cols]
    
    def compare_optimized(self, data):
        results = []

        for i in range(data.shape[0]):
            row = data.iloc[i]
            wholesaler_id = row['Wholesaler_ID']
            product_id = row['Product_ID']
            optim_disc = row['optimized_discount']

            subset = self.data_ready[
                (self.data_ready['Wholesaler_ID'] == wholesaler_id) &
                (self.data_ready['Product_ID'] == product_id)
            ]

            if subset.empty:
                continue

            res = self.disc_simulation(
                data=subset.copy(),
                drop_col=self.drop_col,
                target=self.target,
                discount=optim_disc,
                encoder=self.encoder,
                model=self.model
            )

            if res is None or res.empty:
                continue

            predicted_quantity = res['Predicted_Quantity'].iloc[0]
            base_price = subset['Base_Price'].iloc[-1]
            cost = subset['modal_per_pcs_inc_PPN'].iloc[-1]
            past_discount_amt = subset['Discount_Amount'].iloc[-1]

            optimized_discount_amt = (base_price * predicted_quantity) * optim_disc
            new_final_price = (base_price * predicted_quantity - optimized_discount_amt) / predicted_quantity
            profit_per_unit = new_final_price - cost
            profit_margin_pct = profit_per_unit / new_final_price if new_final_price != 0 else 0
            total_profit = profit_per_unit * predicted_quantity

            past_price = row['past_price']
            past_profit = row['past_profit']
            past_discount = row['past_discount']
            past_profit_margin = (past_price - cost) / past_price if past_price != 0 else 0

            discount_diff = (optim_disc - past_discount) / past_discount if past_discount != 0 else 0
            profit_diff = (total_profit - past_profit) / past_profit if past_profit != 0 else 0

            results.append({
                **row.to_dict(),
                'past_profit_margin': past_profit_margin,
                'past_discount_amt': past_discount_amt,
                'predicted_quantity': predicted_quantity,
                'predicted_profit': total_profit,
                'predicted_profit_margin': profit_margin_pct,
                'optimized_discount_amt': optimized_discount_amt,
                'optimized_price': new_final_price
            })

        self.rf_optimized_df = pd.DataFrame(results)
        return self.rf_optimized_df if not self.rf_optimized_df.empty else None


