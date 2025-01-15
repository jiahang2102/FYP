import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import altair as alt
import pickle

# Set the page configuration
st.set_page_config(
    page_title="Drug Inventory Tracker",
    page_icon=":pill:",
)

# Load and preprocess dataset
def load_and_prepare_data(file_path):
    try:
        df = pd.read_excel(file_path, sheet_name=0)
        df.columns = df.columns.str.strip()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()
    
    # Drop unnecessary columns and preprocess
    columns_to_drop = [
        'no. of tablets/unit per box', 'average weight of one box of medication with PIL (g)',
        'Drug Code', 'Active Pharmaceutical Ingredients', 'Combination Drug (Y/N)',
        'Special Formulation (Y/N)', 'ZipLock bag (S)', 'ZipLock bag (M)', 'ZipLock bag (L)',
        'ZipLock bag (XL)', 'Drug Bin Weight (g)', 'Rubber band'
    ]
    df.drop(columns=columns_to_drop, errors='ignore', inplace=True)
    df.replace(to_replace=r'(?i)\bN\.?A\.?\b', value=np.nan, regex=True, inplace=True)
    df.dropna(subset=['number of tablets', 'Average weight of one loose cut tablet', 
                      'Active Pharmeutical Ingredient Strength (mg)'], inplace=True)

    # Feature engineering
    df['Total weight of tablets without mixed packaging'] = (
        df['number of tablets'] * df['Average weight of one loose cut tablet']
    )
    df['Weight-to-Strength Ratio'] = (
        df['Total weight of tablets without mixed packaging'] / 
        df['Active Pharmeutical Ingredient Strength (mg)']
    )
    return df

# Train models
def train_model(data, model_type):
    try:
        # Ensure target variable is present
        if 'number of tablets' not in data.columns:
            st.error("The target column 'number of tablets' is missing from the dataset.")
            return None, None, None

        # Separate features and target
        X = data.drop(columns=['number of tablets'], errors='ignore')
        y = data['number of tablets']

        # Convert categorical variables to dummies
        X = pd.get_dummies(X, drop_first=True)

        # Check for NaN or infinite values
        if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
            st.error("Dataset contains missing values. Please clean the data and try again.")
            st.write("Missing values in features:", X.isnull().sum().sum())
            st.write("Missing values in target:", y.isnull().sum())
            return None, None, None

        if not np.isfinite(X).all().all() or not np.isfinite(y).all():
            st.error("Dataset contains non-finite values (e.g., Inf or -Inf). Please clean the data.")
            return None, None, None

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Select and train the model
        if model_type == 'Linear Regression':
            model = LinearRegression()
        elif model_type == 'Gradient Boosting':
            model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.09, max_depth=3, random_state=42)
        elif model_type == 'XGBoost':
            model = XGBRegressor(n_estimators=400, learning_rate=0.06, max_depth=4, random_state=42)
        elif model_type == 'Random Forest':
            model = RandomForestRegressor(n_estimators=200, random_state=42)
        else:
            st.error("Invalid model type selected.")
            return None, None, None

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions and calculate metrics
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return model, mse, r2
    except Exception as e:
        st.error(f"An error occurred during model training: {e}")
        return None, None, None
# File upload
st.title("Drug Inventory Tracker")
uploaded_file = st.file_uploader("Upload your dataset (Excel format)", type=["xlsx"])
if uploaded_file:
    data = load_and_prepare_data(uploaded_file)
    if not data.empty:
        st.success("Data loaded and processed successfully!")
        st.dataframe(data.head())

        # Model selection and training
        st.subheader("Train a Model")
        model_type = st.selectbox("Select a model type", 
                                  ["Linear Regression", "Gradient Boosting", "XGBoost", "Random Forest"])
        if st.button("Train Model"):
            model, mse, r2 = train_model(data, model_type)
            if model:
                st.write(f"Model: {model_type}")
                st.write(f"Mean Squared Error (MSE): {mse:.2f}")
                st.write(f"R-squared (R2): {r2:.2f}")
                # Save the model
                model_filename = f"{model_type.replace(' ', '_').lower()}_model.pkl"
                with open(model_filename, 'wb') as file:
                    pickle.dump(model, file)
                st.success(f"Model trained and saved as {model_filename}")
else:
    st.warning("Please upload a dataset to proceed.")

# Display the title and introductory information
"""
# 🌐 Drug Inventory Tracker
**Track your drug inventory with ease!**
This dashboard displays inventory data directly from the uploaded datasheet.
"""

st.info("Below is the current inventory data. You can edit, add, or remove entries as needed.")

# Dynamically set columns_to_display to include all columns from the cleansed dataset
columns_to_display = list(data.columns)  # Use all columns from the dataset

# Editable data table
edited_data = st.data_editor(
    data[columns_to_display],
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
