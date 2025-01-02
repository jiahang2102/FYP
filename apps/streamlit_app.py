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
    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")
        st.stop()
else:
    st.info("Please upload an Excel file to proceed.")
    st.stop()

# Display the title and introductory information
"""
# 🌐 Drug Inventory Tracker

**Track your drug inventory with ease!**
This dashboard displays inventory data directly from the cleansed dataset.
"""

st.info("Below is the current inventory data. You can edit, add, or remove entries as needed.")

# Define relevant columns for display
columns_to_display = [
    "Brand Name",
    "Active Pharmeutical Ingredient Strength (mg)",
    "Dosage Form",
    "number of tablets",
    "Total weight of tablets without mixed packaging",
    "Packaging Complexity Score",
    "Weight per Unit Box or Strip",
    "Weight-to-Strength Ratio",
]

# Editable data table
edited_data = st.data_editor(
    data[columns_to_display],
    num_rows="dynamic",
    disabled=[],  # Allow editing of all columns
    column_config={
        "Active Pharmeutical Ingredient Strength (mg)": st.column_config.NumberColumn(format="%.2f mg"),
        "Total weight of tablets without mixed packaging": st.column_config.NumberColumn(format="%.2f g"),
        "Weight-to-Strength Ratio": st.column_config.NumberColumn(format="%.2f"),
    },
    key="inventory_table",
)

# Add a save button
if st.button("Save Changes"):
    # Placeholder for saving functionality
    st.success("Changes saved successfully! (Implement saving logic)")

"""
---
"""

# Visualizations
st.subheader("Inventory Insights")

# 1. Total weight by drug
st.altair_chart(
    alt.Chart(data).mark_bar().encode(
        x="Total weight of tablets without mixed packaging",
        y=alt.Y("Brand Name", sort="-x"),
        color="Brand Name",
    ),
    use_container_width=True,
)

# 2. Tablets per drug
st.altair_chart(
    alt.Chart(data).mark_bar(color="orange").encode(
        x="number of tablets",
        y=alt.Y("Brand Name", sort="-x"),
    ),
    use_container_width=True,
)

# Alerts for low stock
st.subheader("Low Stock Alerts")
low_stock_threshold = st.slider("Set low stock threshold", min_value=0, max_value=100, value=10)
low_stock_items = data[data["number of tablets"] <= low_stock_threshold]

if not low_stock_items.empty:
    st.error("The following items are running low:")
    st.dataframe(low_stock_items)
else:
    st.success("All items are sufficiently stocked.")
