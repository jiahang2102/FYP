import streamlit as st
import pandas as pd
import altair as alt

# Set the page configuration
st.set_page_config(
    page_title="Drug Inventory Tracker",
    page_icon=":pill:",
)

# Load the dataset
DATA_PATH = "data/cleansed_data.xlsx"  # Updated to point to the correct file
def load_data(file_path):
    try:
        df = pd.read_excel(file_path, sheet_name=0)
        df.columns = df.columns.str.strip()  # Clean column names (remove leading/trailing spaces)
        return df
    except FileNotFoundError:
        st.error(f"Data file not found. Please ensure the file is located at '{file_path}'.")
        return pd.DataFrame()

data = load_data(DATA_PATH)

# Check if data loaded successfully
if data.empty:
    st.stop()

# Display the title and introductory information
"""
# 🌐 Drug Inventory Tracker
**Track your drug inventory with ease!**
This dashboard displays inventory data directly from the uploaded datasheet.
"""

st.info("Below is the current inventory data. You can edit, add, or remove entries as needed.")

# Define relevant columns for display
columns_to_display = [
    "Brand Name",
    "Drug Code",
    "Active Pharmaceutical Ingredients",
    "Active Pharmeutical Ingredient Strength (mg)",
    "Dosage Form",
    "Combination Drug (Y/N)",
    "Special Formulation (Y/N)",
    "number of tablets",
    "Total weight of counted drug with mixed packing",
]

# Keep only existing columns in the dataset
for col in columns_to_display:
    if col not in data.columns:
        data[col] = None  # Add missing columns with null values (NaN)

if not columns_to_display:
    st.error("No valid columns found in the dataset. Please check the data file.")
    st.stop()

# Replace null values with a placeholder for better visualization
data_display = data[columns_to_display].fillna("N/A")

# Editable data table
edited_data = st.data_editor(
    data[columns_to_display],  # Select columns for display
    num_rows="dynamic",
    disabled=[],  # Allow editing of all columns
    column_config={
        "Active Pharmeutical Ingredient Strength (mg)": st.column_config.NumberColumn(format="%.2f mg"),
        "Total weight of counted drug with mixed packing": st.column_config.NumberColumn(format="%.2f g"),
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
        x="Total weight of counted drug with mixed packing",
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
low_stock_items = data[data["number of tablets"] <= low_stock_threshold] if "number of tablets" in data.columns else pd.DataFrame()
if not low_stock_items.empty:
    st.error("The following items are running low:")
    st.dataframe(low_stock_items)
else:
    st.success("All items are sufficiently stocked.")
