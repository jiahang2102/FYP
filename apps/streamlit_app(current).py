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
    
    # Drop unnecessary columns
    columns_to_drop = [
        'no. of tablets/unit per box', 'average weight of one box of medication with PIL (g)',
        'Drug Code', 'Active Pharmaceutical Ingredients', 'Combination Drug (Y/N)',
        'Special Formulation (Y/N)', 'ZipLock bag (S)', 'ZipLock bag (M)', 'ZipLock bag (L)',
        'ZipLock bag (XL)', 'Drug Bin Weight (g)', 'Rubber band'
    ]
    df.drop(columns=columns_to_drop, errors='ignore', inplace=True)
    
    # Remove unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Remove duplicate rows
    df.drop_duplicates(inplace=True)
    
    # Replace "N.A." values with NaN
    df.replace(to_replace=r'(?i)\bN\.?A\.?\b', value=np.nan, regex=True, inplace=True)
    
    # Drop rows with NaN in essential columns
    df.dropna(subset=['number of tablets', 'Average weight of one loose cut tablet', 
                      'Active Pharmeutical Ingredient Strength (mg)'], inplace=True)
    
    # Calculate total weight of tablets without mixed packaging
    df['Total weight of tablets without mixed packaging'] = (
        df['number of tablets'] * df['Average weight of one loose cut tablet']
    )
    
    # Remove outliers using the IQR method
    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    columns_to_check_outliers = [
        'number of tablets',
        'Average weight of one loose cut tablet',
        'Active Pharmeutical Ingredient Strength (mg)'
    ]
    for col in columns_to_check_outliers:
        df = remove_outliers(df, col)
    
    # Calculate Packaging Complexity Score
    def calculate_complexity_score(row):
        score = 1  # Base score if the drug has only one packaging type
        if row.get('Box (Y/N)', 'N') == 'Y':
            score += 1
        if row.get('Full strips (Y/N)', 'N') == 'Y':
            score += 1
        if row.get('Loose Cut (Y/N)', 'N') == 'Y':
            score -= 1
        if row.get('rubber band (Y/N)', 'N') == 'Y':
            score += 1
        return score
    df['Packaging Complexity Score'] = df.apply(calculate_complexity_score, axis=1)
    
    # Calculate Number of Packaging
    df['Number of Packaging'] = (
        (df['Full strips (Y/N)'] == 'Y').astype(int) +
        (df['Box (Y/N)'] == 'Y').astype(int) +
        df['Number of rubber band']
    )
    
    # Calculate weight per unit box or strip
    df['Weight per Unit Box or Strip'] = (
        df['Average weight of one strip/unit of medication (g)'] / df['no. of tablet/unit per strip']
    )
    
    # Mean Weight by Packaging Type
    df['Mean Weight Box'] = df.apply(
        lambda row: row['Total weight of tablets without mixed packaging'] / row['number of tablets']
                    if row.get('Box (Y/N)', 'N') == 'Y' else 0, 
        axis=1
    )
    df['Mean Weight Strip'] = df.apply(
        lambda row: row['Total weight of tablets without mixed packaging'] / row['number of tablets']
                    if row.get('Full strips (Y/N)', 'N') == 'Y' else 0, 
        axis=1
    )
    df['Mean Weight Loose Cut'] = df.apply(
        lambda row: row['Total weight of tablets without mixed packaging'] / row['number of tablets']
                    if row.get('Loose Cut (Y/N)', 'N') == 'Y' else 0, 
        axis=1
    )
    
    # Calculate Weight-to-Strength Ratio
    df['Weight-to-Strength Ratio'] = df.apply(
        lambda row: row['Total weight of tablets without mixed packaging'] / row['Active Pharmeutical Ingredient Strength (mg)']
        if row['Active Pharmeutical Ingredient Strength (mg)'] > 0 else np.nan,
        axis=1
    )
    
    # Polynomial interactions: squared terms and interaction terms
    df['Total Weight Squared'] = df['Total weight of tablets without mixed packaging'] ** 2
    df['Tablet Count Squared'] = df['number of tablets'] ** 2
    df['Weight x Tablet Count'] = df['Total weight of tablets without mixed packaging'] * df['number of tablets']
    
    # Z-Score Normalization for key numerical columns
    columns_to_normalize = [
        'Total weight of tablets without mixed packaging',
        'Weight per Unit Box or Strip',
        'number of tablets',
        'Active Pharmeutical Ingredient Strength (mg)',
        'Packaging Complexity Score'
    ]
    for col in columns_to_normalize:
        col_mean = df[col].mean()
        col_std = df[col].std()
        df[f'Normalized {col}'] = (df[col] - col_mean) / col_std
    
    # One-hot encoding for categorical columns
    columns_to_encode = [col for col in df.select_dtypes(include=['object']).columns if col != 'Brand Name']
    df = pd.get_dummies(df, columns=columns_to_encode, drop_first=True)
    
    # Final cleanup: drop any remaining NaN rows introduced during encoding or transformations
    df.dropna(inplace=True)
    
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

# Section: Add New Data
st.subheader("Add New Data")

# Create a form for user input
with st.form("add_data_form", clear_on_submit=True):
    st.write("Enter new data below:")

    # User inputs for each column (replace column names and types as needed)
    brand_name = st.text_input("Brand Name")
    number_of_tablets = st.number_input("Number of Tablets", min_value=0, step=1)
    avg_weight_tablet = st.number_input("Average Weight of One Loose Cut Tablet (g)", min_value=0.0, step=0.01)
    strength_mg = st.number_input("Active Pharmaceutical Ingredient Strength (mg)", min_value=0.0, step=0.1)
    packaging_box = st.selectbox("Box (Y/N)", options=["Y", "N"])
    packaging_strip = st.selectbox("Full Strips (Y/N)", options=["Y", "N"])
    packaging_loose = st.selectbox("Loose Cut (Y/N)", options=["Y", "N"])
    rubber_band = st.number_input("Number of Rubber Bands", min_value=0, step=1)

    # Submit button for the form
    submitted = st.form_submit_button("Add Data")

    if submitted:
        # Validate the input
        errors = []
        if not brand_name.strip():
            errors.append("Brand Name cannot be empty.")
        if number_of_tablets <= 0:
            errors.append("Number of Tablets must be greater than zero.")
        if avg_weight_tablet <= 0:
            errors.append("Average Weight must be a positive number.")
        if strength_mg <= 0:
            errors.append("Active Pharmaceutical Ingredient Strength must be a positive number.")
        if rubber_band < 0:
            errors.append("Number of Rubber Bands cannot be negative.")

        # If validation errors exist, show them to the user
        if errors:
            for error in errors:
                st.error(error)
        else:
            # Add the valid data to the DataFrame
            new_row = {
                "Brand Name": brand_name,
                "number of tablets": number_of_tablets,
                "Average weight of one loose cut tablet": avg_weight_tablet,
                "Active Pharmeutical Ingredient Strength (mg)": strength_mg,
                "Box (Y/N)": packaging_box,
                "Full strips (Y/N)": packaging_strip,
                "Loose Cut (Y/N)": packaging_loose,
                "Number of rubber band": rubber_band,
            }

            # Append the new row to the dataset
            st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
            st.success("Data added successfully!")
            st.dataframe(st.session_state.data.tail(5))  # Show the last few rows as confirmation


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
