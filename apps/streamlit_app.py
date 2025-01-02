import streamlit as st
import pandas as pd
import altair as alt
import fyp  # Import the custom data cleansing and model code

# Set the page configuration
st.set_page_config(
    page_title="Drug Inventory Tracker",
    page_icon=":pill:",
)

# Streamlit file uploader
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    # Use fyp.load_and_prepare_data to process the uploaded file
    try:
        data = fyp.load_and_prepare_data(uploaded_file)
        st.success("File successfully loaded and cleansed.")
        st.write("Available columns in the data:", data.columns.tolist())
    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")
        st.stop()
else:
    st.info("Please upload an Excel file to proceed.")
    st.stop()

# Define relevant columns for display
columns_to_display = [
    "brand_name",  # Updated to match cleansed DataFrame column names
    "active_pharmaceutical_ingredient_strength_mg",
    "dosage_form",
    "number_of_tablets",
    "total_weight_of_tablets_without_mixed_packaging",
    "packaging_complexity_score",
    "weight_per_unit_box_or_strip",
    "weight_to_strength_ratio",
]

# Check if all required columns exist
missing_columns = [col for col in columns_to_display if col not in data.columns]
if missing_columns:
    st.warning(f"The following columns are missing and will not be displayed: {missing_columns}")
    columns_to_display = [col for col in columns_to_display if col in data.columns]

# Editable data table
edited_data = st.data_editor(
    data[columns_to_display],
    num_rows="dynamic",
    disabled=[],  # Allow editing of all columns
)

# Add a save button
if st.button("Save Changes"):
    st.success("Changes saved successfully! (Implement saving logic)")

# Visualizations
st.subheader("Inventory Insights")

# 1. Total weight by drug
if "total_weight_of_tablets_without_mixed_packaging" in data.columns:
    st.altair_chart(
        alt.Chart(data).mark_bar().encode(
            x="total_weight_of_tablets_without_mixed_packaging",
            y=alt.Y("brand_name", sort="-x"),
            color="brand_name",
        ),
        use_container_width=True,
    )

# 2. Tablets per drug
if "number_of_tablets" in data.columns:
    st.altair_chart(
        alt.Chart(data).mark_bar(color="orange").encode(
            x="number_of_tablets",
            y=alt.Y("brand_name", sort="-x"),
        ),
        use_container_width=True,
    )

# Alerts for low stock
st.subheader("Low Stock Alerts")
if "number_of_tablets" in data.columns:
    low_stock_threshold = st.slider("Set low stock threshold", min_value=0, max_value=100, value=10)
    low_stock_items = data[data["number_of_tablets"] <= low_stock_threshold]

    if not low_stock_items.empty:
        st.error("The following items are running low:")
        st.dataframe(low_stock_items)
    else:
        st.success("All items are sufficiently stocked.")
