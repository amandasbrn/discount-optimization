import streamlit as st
from optimizer import discountOptimizer

st.header('AI Discount Optimization')
tab1, tab2, tab3 = st.tabs(["Discount Simulation", "Discount Optimization","Discount Structure"])
optimizer = discountOptimizer(filepath="final_data.csv")

optimizer.data_prep()
opt_df = optimizer.data_ready
product_mapping = opt_df[['Product_ID', 'Product_Desc']].drop_duplicates().set_index('Product_ID')['Product_Desc'].to_dict()

wholesaler_ids = sorted(opt_df['Wholesaler_ID'].unique())

with tab1:
    selected_wholesaler = st.selectbox("Select Wholesaler", wholesaler_ids, key='wholesaler_sim')
    # get products for the selected wholesaler
    products_for_wholesaler = opt_df[opt_df['Wholesaler_ID'] == selected_wholesaler][['Product_ID', 'Product_Desc']].drop_duplicates()

    # display product name but map back to Product_ID
    product_display_to_id = {f"{row.Product_Desc} ({row.Product_ID})": row.Product_ID for _, row in products_for_wholesaler.iterrows()}
    selected_display = st.selectbox("Select Product", sorted(product_display_to_id.keys()), key='product_sim')
    selected_product = product_display_to_id[selected_display]  # this will give the actual Product_ID
    

    subset_df = opt_df[
            (opt_df['Wholesaler_ID'] == selected_wholesaler) &
            (opt_df['Product_ID'] == selected_product)
        ]
    st.subheader('Your data:')
    show_cols = ['Wholesaler_ID','Product_ID','Product_Desc','PO_ID','Transaction_Date','Base_Price','PPN','Quantity_Sold','Discount_Amount','discount_pct','Final_Price_New','Total_Revenue_New','Total_Profit','Profit_Margin_%']
    st.dataframe(subset_df[show_cols])

    discount_sml = st.slider("Input your desired discount (%)", 0, 50, 20) / 100.0
    if st.button("See simulation result", type="primary"):
        optimizer.modeling()

        # test data


        result = optimizer.disc_simulation(
                            data=subset_df,
                            drop_col=optimizer.drop_col,
                            target=optimizer.target,
                            discount=discount_sml,
                            encoder=optimizer.encoder,
                            model=optimizer.model
                        )
        st.write('Discount:', discount_sml)
        st.dataframe(result)

with tab2:
    selected_wholesaler = st.selectbox("Select Wholesaler", wholesaler_ids, key='wholesaler_optim')
    # get products for the selected wholesaler
    products_for_wholesaler = opt_df[opt_df['Wholesaler_ID'] == selected_wholesaler][['Product_ID', 'Product_Desc']].drop_duplicates()

    # display product name but map back to Product_ID
    product_display_to_id = {f"{row.Product_Desc} ({row.Product_ID})": row.Product_ID for _, row in products_for_wholesaler.iterrows()}
    selected_display = st.selectbox("Select Product", sorted(product_display_to_id.keys()),key='product_optim')
    selected_product = product_display_to_id[selected_display]  # this will give the actual Product_ID

    if st.button("Get the optimized discount", type="primary"):
        with st.spinner('Let the system finds the optimal discount for you...'):
            optimizer.modeling()
            opt_res = optimizer.optimize_discounts()
            data_opt = opt_res[
                    (opt_res['Wholesaler_ID'] == selected_wholesaler) & 
                    (opt_res['Product_ID'] == selected_product)
                ]
            result_df = optimizer.compare_optimized(data_opt)
            if result_df is None or result_df.empty:
                st.warning("This wholesaler and product already have the most optimal discount.")
            else:
                st.subheader("Optimized Discount Info")
                st.info(f"Optimized discount for this wholesaler and product: {result_df['optimized_discount'].values[0]:.2%}")
                st.info(f"Predicted profit margin: {result_df['predicted_profit_margin'].values[0]:.2%}")

                past_cols = ['Wholesaler_ID','Product_ID','past_price','past_discount','past_discount_amt','past_qty','past_profit','past_profit_margin']
                new_cols = ['Wholesaler_ID','Product_ID','optimized_price','optimized_discount','optimized_discount_amt','predicted_quantity','predicted_profit','predicted_profit_margin']

                st.markdown('Before discount optimization')
                st.dataframe(result_df[past_cols])

                st.markdown('After discount optimization')
                st.dataframe(result_df[new_cols])

with tab3:
    with st.spinner('Please wait...'):
        optimizer.modeling()
        opt_res = optimizer.optimize_discounts()
        st.dataframe(optimizer.compare_optimized(opt_res))