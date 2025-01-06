import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path

# Streamlit app configuration
st.set_page_config(page_title="Drug Data Processing and Modeling", page_icon=":pill:")

st.title("Drug Data Processing and Modeling")
st.sidebar.header("Navigation")
app_mode = st.sidebar.selectbox("Choose an action", ["Data Processing", "Run Models", "View Data"])

# Load and prepare data
def load_and_prepare_data(file):
    # Load data from Excel file
    data = pd.read_excel(file)

    # Drop unnecessary columns
    columns_to_drop = [
        'no. of tablets/unit per box',
        'average weight of one box of medication with PIL (g)',
        'Drug Code',
        'Active Pharmaceutical Ingredients',
        'Combination Drug (Y/N)',
        'Special Formulation (Y/N)',
        'ZipLock bag (S)', 'ZipLock bag (M)', 'ZipLock bag (L)', 'ZipLock bag (XL)',
        'Drug Bin Weight (g)',
        'Rubber band',
    ]
    data = data.drop(columns=columns_to_drop, errors='ignore')
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

    # Remove duplicate rows
    data = data.drop_duplicates()

    # Replace "N.A." values with NaN
    data.replace(to_replace=r"(?i)\bN\.?A\.?\b", value=np.nan, regex=True, inplace=True)

    # Drop rows with NaN in essential columns
    data = data.dropna(subset=[
        'number of tablets',
        'Average weight of one loose cut tablet',
        'Active Pharmeutical Ingredient Strength (mg)',
    ])

    # Calculate total weight of tablets without mixed packaging
    data['Total weight of tablets without mixed packaging'] = (
        data['number of tablets'] * data['Average weight of one loose cut tablet']
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
        'Active Pharmeutical Ingredient Strength (mg)',
    ]
    for col in columns_to_check_outliers:
        data = remove_outliers(data, col)

    # Add custom features
    data['Weight-to-Strength Ratio'] = data['Total weight of tablets without mixed packaging'] / data[
        'Active Pharmeutical Ingredient Strength (mg)']

    data = pd.get_dummies(data, drop_first=True)
    return data

# Export function
def export_to_excel(data, filename="cleansed_data.xlsx"):
    download_path = Path.home() / "Downloads" / filename
    data.to_excel(download_path, index=False)
    st.success(f"Data has been exported to {download_path}")

# Model functions
def run_linear_regression_model(data):
    st.subheader("Linear Regression Results")
    target_column = 'number of tablets'
    data_lr = data.dropna(subset=[target_column, 'Average weight of one loose cut tablet'])
    X = data_lr[['Average weight of one loose cut tablet']]
    y = data_lr[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"Mean Squared Error (MSE): {mse}")
    st.write(f"R-squared (R2): {r2}")

def run_gradient_boost_model(data):
    st.subheader("Gradient Boosting Results")
    target_column = 'number of tablets'
    data_gb = data.dropna()
    X = data_gb.drop(columns=[target_column])
    y = data_gb[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.09, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"Mean Squared Error (MSE): {mse}")
    st.write(f"R-squared (R2): {r2}")

# Streamlit app functionality
if app_mode == "Data Processing":
    st.header("Data Processing")
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])
    if uploaded_file:
        data = load_and_prepare_data(uploaded_file)
        st.write("Processed Data:")
        st.write(data.head())
        if st.button("Export Processed Data"):
            export_to_excel(data)

elif app_mode == "Run Models":
    st.header("Run Models")
    uploaded_file = st.file_uploader("Upload your Excel file for modeling", type=["xlsx", "xls"])
    if uploaded_file:
        data = load_and_prepare_data(uploaded_file)
        model_choice = st.radio("Choose a model to run", ["Linear Regression", "Gradient Boosting"])
        if model_choice == "Linear Regression":
            run_linear_regression_model(data)
        elif model_choice == "Gradient Boosting":
            run_gradient_boost_model(data)

elif app_mode == "View Data":
    st.header("View Data")
    st.write("Upload data in the 'Data Processing' section to view details.")
