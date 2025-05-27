import streamlit as st
import pandas as pd
import numpy as np
import joblib
from optim_backend import discountOptimV2 

st.header('AI Discount Optimization')
optimizer = discountOptimV2('training_data.csv')
optimizer.training_data()
data = optimizer.data_purchased
model = joblib.load('trained_pipeline.joblib')

product_mapping = data[['Product_ID', 'Product_Desc']].drop_duplicates().set_index('Product_ID')['Product_Desc'].to_dict()
wholesaler_ids = sorted(data['Wholesaler_ID'].unique())
depo_ids = sorted(data['Kode_Depo'].unique())

tab1, tab2, tab3, tab4 = st.tabs(["Discount Simulation", "Discount Structure", "Simulation History", "Check Discount"])

with tab1:

    # '''
    #     Objective: discount simulation
    #     this is where user can get the most optimal discount based on their own simulation

    # '''

    #st.dataframe(data.head())
    # ----------------- USER INPUT ----------------------
    st.markdown("*Generate the most optimal discount based on the simulation here.*")
    # input depo
    selected_depo = str(st.selectbox("Select Depo", depo_ids, key='depo_optim'))

    # input wholesaler based on depo
    wholesaler_depo = data[data['Kode_Depo'] == selected_depo][['Wholesaler_ID']].drop_duplicates()
    selected_wholesaler = str(st.selectbox("Select Wholesaler", wholesaler_depo, key='wholesaler_optim'))

    # input product based on wholesaler
    products_for_wholesaler = data[data['Wholesaler_ID'] == selected_wholesaler][['Product_ID', 'Product_Desc']].drop_duplicates()
    product_display_to_id = {f"{row.Product_Desc} ({row.Product_ID})": row.Product_ID for _, row in products_for_wholesaler.iterrows()}
    
    selected_display = str(st.selectbox("Select Product", sorted(product_display_to_id.keys()),key='product_optim'))
    selected_product = product_display_to_id[selected_display]  # this will give the actual Product_ID

    subset_data = data[(data['Wholesaler_ID']==selected_wholesaler) & (data['Product_ID']==selected_product)]

    # other inputs
    user_base_price = int(st.number_input("Enter Base Price:", min_value=1, value=1000))
    unique_base = sorted(subset_data['Base_Price'].round(2).unique())
    st.caption(f"Past base price: range from {min(unique_base)} to {max(unique_base)}")

    user_ppn = int(st.number_input("Enter PPN:", min_value=1, value=1000))
    unique_ppn = sorted(subset_data['PPN'].round(2).unique())
    st.caption(f"Past PPN: range from {min(unique_ppn)} to {max(unique_ppn)}")

    user_cost = int(st.number_input("Enter Cost:", min_value=1, value=1000))
    unique_cost = sorted(subset_data['modal_per_pcs_inc_PPN'].round(2).unique())
    st.caption(f"Past cost: range from {min(unique_cost)} to {max(unique_cost)}")

    #user_init_discount = float(st.number_input("Enter the initial discount you want to give:", min_value=0.0, max_value=1.0, value=0.05))
    # user_is_discounted = st.checkbox("Do you want to give a discount?", value=True)
    # if user_is_discounted:
    #     status = 1
    # else:
    #     status = 0

    if "results_df" not in st.session_state:
        st.session_state.results_df = pd.DataFrame(columns=[
            "Wholesaler_ID", "Product_ID", "Product_Desc", "Input_Base_Price", "Input_PPN",
            "Input_Modal", "Past_Discount", "Past_Final_Price","Past_Quantity","Past_Profit","Past_Profit_Margin",
            "Optimized_Discount", "Optimized_Final_Price",
            "Predicted_Quantity", "Predicted_Profit", "Profit_Margin"
        ])
        
    # ----------------------------------------------------
    # GENERATE RESULT
    # ----------------------------------------------------
    if st.button('🔍 Generate the most optimal discount', type='primary'):
        with st.spinner('Please wait...'):
            result = optimizer.optimization(selected_wholesaler, selected_product, user_base_price, user_ppn, user_cost, model)
            if result is not None and result.get("optimal_discount") is not None:
                st.markdown("**BEFORE OPTIMIZATION**")
                st.text(f"✅ Past Discount: {round(result['latest_past_discount'],4)*100} %")
                st.text(f"💰 Past Final Price: {result['latest_past_final_price']:.2f}")
                st.text(f"💰 Past Quantity: {result['latest_qty']:.0f}")
                st.text(f"💰 Past Profit: {result['latest_profit']:.2f}")
                st.text(f"💰 Past Profit Margin %: {round(result['latest_profit_margin'],2)*100} %")
                st.text("")
                st.markdown("**OPTIMIZATION RESULT**")
                st.text(f"✅ Optimal Discount: {round(result['optimal_discount'],4)*100} %")
                st.text(f"💰 Final Price: {round(result['optimal_final_price'], 2)}")
                st.text(f"📦 Predicted Quantity: {round(result['predicted_quantity'])}")
                st.text(f"📈 Predicted Profit: {round(result['predicted_profit'], 2)}")
                st.text(f"📈 Predicted Profit Margin %: {round(result['predicted_profit_margin']*100,2)} %")

                new_row = {
                    "Wholesaler_ID": selected_wholesaler,
                    "Product_ID": selected_product,
                    "Product_Desc": selected_display,
                    "Input_Base_Price": user_base_price,
                    "Input_PPN": user_ppn,
                    "Input_Modal": user_cost,
                    "Past_Discount": round(result['latest_past_discount'], 4),
                    "Past_Final_Price": round(result['latest_past_final_price'], 2),
                    "Past_Quantity": round(result['latest_qty']),
                    "Past_Profit": round(result['latest_profit'], 2),
                    "Past_Profit_Margin": round(result['latest_profit_margin'], 2),
                    "Optimized_Discount": round(result['optimal_discount'], 4),
                    "Optimized_Final_Price": round(result['optimal_final_price'], 2),
                    "Predicted_Quantity": round(result['predicted_quantity']),
                    "Predicted_Profit": round(result['predicted_profit'], 2),
                    "Profit_Margin": round(result['predicted_profit_margin'], 2)
                }

                st.session_state.results_df = pd.concat(
                    [st.session_state.results_df, pd.DataFrame([new_row])],
                    ignore_index=True
                )
                st.success('Successfully saved the simulation!')
            else:
                st.warning('⚠️ No valid result could be generated.')


with tab2:
    # '''
    #     Objective: discount structure
    #     This stores the most optimal discount for historical data, for each wholesaler and product.
    #     User can get summary & recommendations
    # '''

    # ----------------------------------------
    # USER INPUT
    # ----------------------------------------
    st.markdown("*Generate the most optimal discount based on the simulation here.*")
    # input depo
    selected_depo = str(st.selectbox("Select Depo", depo_ids, key='depo_str'))

    # input wholesaler based on depo
    wholesaler_depo = data[data['Kode_Depo'] == selected_depo][['Wholesaler_ID']].drop_duplicates()
    selected_wholesaler = str(st.selectbox("Select Wholesaler", wholesaler_depo, key='wholesaler_str'))

    # input product based on wholesaler
    products_for_wholesaler = data[data['Wholesaler_ID'] == selected_wholesaler][['Product_ID', 'Product_Desc']].drop_duplicates()
    product_display_to_id = {f"{row.Product_Desc} ({row.Product_ID})": row.Product_ID for _, row in products_for_wholesaler.iterrows()}
    
    selected_display = str(st.selectbox("Select Product", sorted(product_display_to_id.keys()),key='product_str'))
    selected_product = product_display_to_id[selected_display]  # this will give the actual Product_ID

    # ----------------------------------------
    # SHOW OPTIMIZATION RESULT
    # ----------------------------------------

    optimized_df = pd.read_csv('optimized_df.csv')
    optimized_df['Product_ID'] = optimized_df['Product_ID'].astype('str')

    if st.button('🔍 Show discount structure', type='primary'):
        opt_subset_data = optimized_df[(optimized_df['Wholesaler_ID']==selected_wholesaler) & (optimized_df['Product_ID']==selected_product)]
        opt_subset_data = opt_subset_data.drop(['unique_ID','optimized_discount_str'],axis=1)

        clusters = opt_subset_data['Cluster'].unique()
        texts = []

        if clusters[0] == 0:
            texts.append("📦 This wholesaler typically places low-volume orders. Their purchasing behavior is steady but modest.")
        elif clusters[0] == 1:
            texts.append("🚚 This wholesaler is a relatively moderate volume buyers. They contribute significantly to revenue but operate on thinner margins.")
        elif clusters[0] == 2:
            texts.append("🏆 This wholesaler may be a strategic or key account that consistently place very large orders. They receive discounts regularly and are vital for long-term growth.")
        else:
            texts.append("❓ This wholesaler's behavior does not align with known clusters and may require further analysis.")

        elasticity_dominant = np.sign(opt_subset_data['price_elasticity']).mode()[0]

        if elasticity_dominant < 0:
            elasticity_text = "🟢 For this product, this wholesaler is **sensitive to price changes**."
            elasticity_action = "💡 You can **experiment with small discount changes** to drive higher sales volume."
        elif elasticity_dominant > 0:
            elasticity_text = "🔴 For this product, this wholesaler shows a sign that lower prices may not increase demand."
            elasticity_action = "💡 Consider focusing on **value-added services, exclusivity, or loyalty incentives** instead of discounts."
        else:
            elasticity_text = "⚪ For this product, this wholesaler is **not affected by price changes**."
            elasticity_action = "💡 Focus on **non-price factors** like delivery time, service quality, or bundling."

        st.markdown("#### **📊 Summary for selected Wholesaler & Product**")
        st.markdown(f"""
                    **Wholesaler ID**: {opt_subset_data['Wholesaler_ID'].iloc[0]}

                    **Product ID:** {opt_subset_data['Product_ID'].iloc[0]}

                    """)
        st.markdown("**This wholesaler shows some behaviors:**")
        st.markdown(", ".join(texts))
        st.markdown(elasticity_text)
        st.markdown("**Recommendations for this wholesaler & product:**")
        st.markdown(elasticity_action)
        st.divider()
        st.markdown("#### **📈 Discount structure for selected Wholesaler & Product**")
        st.dataframe(opt_subset_data)

with tab3:

    # '''
    #     Objective: discount simulation history
    #     After doing simulation on tab1, the input goes here

    # '''

    st.dataframe(st.session_state.results_df)

with tab4:

    # '''
    #     Objective: build their own discount
    #     This is where user can get quantity and profit prediction based on their base price, PPN, cost, and discount input.
    #     User can freely test their desired discount here.

    # '''

    st.markdown("*Build simulation of your own discount and price here.*")
    # input wholesaler
    selected_wholesaler = str(st.selectbox("Select Wholesaler", wholesaler_ids, key='wholesaler_disc'))
    products_for_wholesaler = data[data['Wholesaler_ID'] == selected_wholesaler][['Product_ID', 'Product_Desc']].drop_duplicates()
    product_display_to_id = {f"{row.Product_Desc} ({row.Product_ID})": row.Product_ID for _, row in products_for_wholesaler.iterrows()}
    
    # input product
    selected_display = str(st.selectbox("Select Product", sorted(product_display_to_id.keys()),key='product_disc'))
    selected_product = product_display_to_id[selected_display]  # this will give the actual Product_ID

    subset_data = data[(data['Wholesaler_ID']==selected_wholesaler) & (data['Product_ID']==selected_product)]

    # other inputs
    user_base_price = int(st.number_input("Enter Base Price:", min_value=1, value=1000,key='base_disc'))
    unique_base = sorted(subset_data['Base_Price'].round(2).unique())
    st.caption(f"Past base price: range from {min(unique_base)} to {max(unique_base)}")

    user_ppn = int(st.number_input("Enter PPN:", min_value=1, value=1000,key='ppn_disc'))
    unique_ppn = sorted(subset_data['PPN'].round(2).unique())
    st.caption(f"Past PPN: range from {min(unique_ppn)} to {max(unique_ppn)}")

    user_cost = int(st.number_input("Enter Cost:", min_value=1, value=1000,key='cost_disc'))
    unique_cost = sorted(subset_data['modal_per_pcs_inc_PPN'].round(2).unique())
    st.caption(f"Past cost: range from {min(unique_cost)} to {max(unique_cost)}")

    user_disc = float(st.number_input("Enter Desired Discount:", min_value=0.0, max_value=0.5, value=0.2,key='try_disc'))
    unique_discounts = sorted(subset_data['discount_pct'].round(2).unique())
    st.caption(f"Past discounts: range from {min(unique_discounts)} to {max(unique_discounts)}")

    if st.button('🔍 Generate prediction with your discount', type='primary'):
        input_data = optimizer.input_new_data(selected_wholesaler, selected_product, user_base_price, user_ppn, user_cost, user_disc)
        result = optimizer.get_predict_quantity(model, input_data)
        input_data['Predicted_Quantity'] = result
        input_data['Discount_Amount'] = (input_data['Base_Price'] * input_data['Predicted_Quantity']) * input_data['discount_pct']
        input_data['New_Final_Price'] = ((input_data['Base_Price'] * input_data['Predicted_Quantity']) - input_data['Discount_Amount']) / input_data['Predicted_Quantity']
        input_data['Predicted_Profit_Margin_%'] = ((input_data['New_Final_Price'] - input_data['modal_per_pcs_inc_PPN']) / input_data['New_Final_Price'] )
        input_data['Profit_Per_Unit'] = input_data['New_Final_Price'] - input_data['modal_per_pcs_inc_PPN']
        input_data['Predicted_Total_Profit'] = input_data['Profit_Per_Unit'] * input_data['Predicted_Quantity']
        input_data = input_data.drop(['Profit_Per_Unit','New_Final_Price'],axis=1)
        #st.dataframe(input_data)

        # ---- text output ----
        st.text(f"✅ Your Discount: {user_disc*100} %")
        st.text(f"💰 Predicted Final Price: {round(input_data['Final_Price_New'].iloc[0], 2)}")
        st.text(f"📦 Predicted Quantity: {round(input_data['Predicted_Quantity'].iloc[0])}")
        st.text(f"📈 Predicted Profit: {round(input_data['Predicted_Total_Profit'].iloc[0], 2)}")
        st.text(f"📈 Predicted Profit Margin %: {round(input_data['Predicted_Profit_Margin_%'].iloc[0]*100,2)} %")