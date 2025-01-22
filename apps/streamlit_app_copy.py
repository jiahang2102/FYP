import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import altair as alt
from io import BytesIO
from github import Github
import pickle

# Set the page configuration
st.set_page_config(
    page_title="Drug Inventory Tracker",
    page_icon=":pill:",
)

# Load data from GitHub
def load_and_prepare_data_from_github(file_url):
    try:
        response = requests.get(file_url)
        response.raise_for_status()
        df = pd.read_excel(BytesIO(response.content), sheet_name=0)
        df.columns = df.columns.str.strip()
    except Exception as e:
        st.error(f"Error loading data from GitHub: {e}")
        return pd.DataFrame()
    
    # Data cleaning and preprocessing
    columns_to_drop = [
        'no. of tablets/unit per box', 'average weight of one box of medication with PIL (g)',
        'Drug Code', 'Active Pharmaceutical Ingredients', 'Combination Drug (Y/N)',
        'Special Formulation (Y/N)', 'ZipLock bag (S)', 'ZipLock bag (M)', 'ZipLock bag (L)',
        'ZipLock bag (XL)', 'Drug Bin Weight (g)', 'Rubber band'
    ]
    df.drop(columns=columns_to_drop, errors='ignore', inplace=True)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.drop_duplicates(inplace=True)
    df.replace(to_replace=r'(?i)\bN\.?A\.?\b', value=np.nan, regex=True, inplace=True)
    df.dropna(subset=['number of tablets', 'Average weight of one loose cut tablet', 
                      'Active Pharmeutical Ingredient Strength (mg)'], inplace=True)

    df['Total weight of tablets without mixed packaging'] = (
        df['number of tablets'] * df['Average weight of one loose cut tablet']
    )

    # Outlier removal
    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    columns_to_check_outliers = [
        'number of tablets', 'Average weight of one loose cut tablet', 
        'Active Pharmeutical Ingredient Strength (mg)'
    ]
    for col in columns_to_check_outliers:
        df = remove_outliers(df, col)
    
    df['Packaging Complexity Score'] = df.apply(lambda row: (
        1 + (row.get('Box (Y/N)', 'N') == 'Y') + 
        (row.get('Full strips (Y/N)', 'N') == 'Y') - 
        (row.get('Loose Cut (Y/N)', 'N') == 'Y') +
        (row.get('rubber band (Y/N)', 'N') == 'Y')), axis=1)
    
    df['Number of Packaging'] = (
        (df['Full strips (Y/N)'] == 'Y').astype(int) +
        (df['Box (Y/N)'] == 'Y').astype(int) +
        df['Number of rubber band']
    )
    
    df['Weight per Unit Box or Strip'] = (
        df['Average weight of one strip/unit of medication (g)'] / df['no. of tablet/unit per strip']
    )

    df['Weight-to-Strength Ratio'] = df.apply(
        lambda row: row['Total weight of tablets without mixed packaging'] / row['Active Pharmeutical Ingredient Strength (mg)']
        if row['Active Pharmeutical Ingredient Strength (mg)'] > 0 else np.nan, axis=1)
    
    df['Total Weight Squared'] = df['Total weight of tablets without mixed packaging'] ** 2
    df['Tablet Count Squared'] = df['number of tablets'] ** 2
    df['Weight x Tablet Count'] = df['Total weight of tablets without mixed packaging'] * df['number of tablets']
    
    columns_to_normalize = [
        'Total weight of tablets without mixed packaging', 
        'Weight per Unit Box or Strip', 'number of tablets', 
        'Active Pharmeutical Ingredient Strength (mg)', 'Packaging Complexity Score'
    ]
    for col in columns_to_normalize:
        col_mean = df[col].mean()
        col_std = df[col].std()
        df[f'Normalized {col}'] = (df[col] - col_mean) / col_std

    columns_to_encode = [col for col in df.select_dtypes(include=['object']).columns if col != 'Brand Name']
    df = pd.get_dummies(df, columns=columns_to_encode, drop_first=True)
    df.dropna(inplace=True)
    return df

# Save data back to GitHub
def save_changes_to_github(data, repo_name, file_path, access_token):
    try:
        g = Github(access_token)
        repo = g.get_repo(repo_name)
        contents = repo.get_contents(file_path)

        excel_buffer = BytesIO()
        data.to_excel(excel_buffer, index=False)

        repo.update_file(
            path=file_path,
            message="Updated Excel file via Streamlit app",
            content=excel_buffer.getvalue(),
            sha=contents.sha,
        )
        st.success("Changes saved to GitHub successfully!")
    except Exception as e:
        st.error(f"Error saving changes to GitHub: {e}")

# Train regression models
def train_model(data, model_type):
    try:
        if 'number of tablets' not in data.columns:
            st.error("Target column 'number of tablets' is missing.")
            return None, None, None

        X = data.drop(columns=['number of tablets'], errors='ignore')
        y = data['number of tablets']
        X = pd.get_dummies(X, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        if model_type == 'Linear Regression':
            model = LinearRegression()
        elif model_type == 'Gradient Boosting':
            model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.09, max_depth=3, random_state=42)
        elif model_type == 'XGBoost':
            model = XGBRegressor(n_estimators=400, learning_rate=0.06, max_depth=4, random_state=42)
        elif model_type == 'Random Forest':
            model = RandomForestRegressor(n_estimators=200, random_state=42)
        else:
            st.error("Invalid model type.")
            return None, None, None

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return model, mse, r2
    except Exception as e:
        st.error(f"Error during model training: {e}")
        return None, None, None

# Streamlit app
st.title("Drug Inventory Tracker")

repo_name = "your_username/your_repo"
file_path = "FYP/data/SOI database_cleaned.xlsx"
github_file_url = f"https://raw.githubusercontent.com/{repo_name}/main/{file_path}"
access_token = st.text_input("Enter GitHub Access Token (for saving changes)", type="password")

data = load_and_prepare_data_from_github(github_file_url)
if not data.empty:
    st.success("Data loaded successfully!")
    st.dataframe(data.head())

    edited_data = st.data_editor(data, num_rows="dynamic", disabled=[])

    if st.button("Save Changes"):
        if access_token:
            save_changes_to_github(
                data=edited_data,
                repo_name=repo_name,
                file_path=file_path,
                access_token=access_token,
            )
        else:
            st.error("Access Token is required to save changes.")

    st.subheader("Train a Regression Model")
    model_type = st.selectbox("Select a model type", ["Linear Regression", "Gradient Boosting", "XGBoost", "Random Forest"])
    if st.button("Train Model"):
        model, mse, r2 = train_model(data, model_type)
        if model:
            st.write(f"Model: {model_type}")
            st.write(f"MSE: {mse:.2f}")
            st.write(f"R2: {r2:.2f}")
else:
    st.warning("Failed to load data.")
