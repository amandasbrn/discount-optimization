import streamlit as st
from optimizer import discountOptimizer

st.header('AI Discount Optimization')
tab1, tab2 = st.tabs(["Discount Simulation", "Discount Optimization"])
optimizer = discountOptimizer(filepath="final_data.csv")

with tab1:
    discount_sml = st.slider("Input your desired discount (%)", 0, 50, 20) / 100.0
    if st.button("See simulation result", type="primary"):
        optimizer.data_prep()
        optimizer.modeling()

        # test data
        first_row = optimizer.data_ready.iloc[0]
        subset_df = optimizer.data_ready[
            (optimizer.data_ready['Wholesaler_ID'] == first_row['Wholesaler_ID']) &
            (optimizer.data_ready['Product_ID'] == first_row['Product_ID'])
        ]

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
    with st.spinner('Please wait...'):
        optimizer.data_prep()

    # Load initial data for dropdowns
    opt_df = optimizer.data_ready
    product_mapping = opt_df[['Product_ID', 'Product_Desc']].drop_duplicates().set_index('Product_ID')['Product_Desc'].to_dict()
    
    wholesaler_ids = sorted(opt_df['Wholesaler_ID'].unique())
    selected_wholesaler = st.selectbox("Select Wholesaler", wholesaler_ids)

    # get products for the selected wholesaler
    products_for_wholesaler = opt_df[opt_df['Wholesaler_ID'] == selected_wholesaler][['Product_ID', 'Product_Desc']].drop_duplicates()

    # display product name but map back to Product_ID
    product_display_to_id = {f"{row.Product_Desc} ({row.Product_ID})": row.Product_ID for _, row in products_for_wholesaler.iterrows()}
    selected_display = st.selectbox("Select Product", sorted(product_display_to_id.keys()))
    selected_product = product_display_to_id[selected_display]  # this will give the actual Product_ID

    if st.button("Get the optimized discount", type="primary"):
        with st.spinner('Let the system finds the optimal discount for you...'):
            optimizer.modeling()
            optimizer.optimize_discounts()
            result_df = optimizer.compare_optimized(selected_wholesaler, selected_product)

            if result_df is None or result_df.empty:
                st.warning("This wholesaler and product already have the most optimal discount.")
            else:
                result_row = result_df[
                    (result_df['Wholesaler_ID'] == selected_wholesaler) & 
                    (result_df['Product_ID'] == selected_product)
                ]

                st.subheader("Optimized Discount Info")
                st.info(f"Optimized discount for this wholesaler and product: {result_row['optimized_discount'].values[0]:.2%}")
                st.info(f"Predicted profit margin: {result_row['rf_predicted_profit_margin'].values[0]:.2%}")

                past_cols = ['Wholesaler_ID','Product_ID','past_price','past_discount','past_qty','past_profit','past_profit_margin']
                new_cols = ['Wholesaler_ID','Product_ID','new_final_price','optimized_discount','rf_predicted_quantity','rf_predicted_profit']

                st.markdown('Before discount optimization')
                st.dataframe(result_row[past_cols])

                st.markdown('After discount optimization')
                st.dataframe(result_row[new_cols])


