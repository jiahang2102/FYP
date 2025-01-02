from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import tkinter as tk
from tkinter import filedialog


def load_and_prepare_data(file_path):
    # Load data from Excel file
    data = pd.read_excel(file_path)

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

    # Replace "N.A." values and its variations with NaN
    data.replace(to_replace=r'(?i)\bN\.?A\.?\b', value=np.nan, regex=True, inplace=True)

    # Drop rows with NaN in essential columns
    data = data.dropna(subset=['number of tablets',
                               'Average weight of one loose cut tablet', 
                               'Active Pharmeutical Ingredient Strength (mg)'
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

    # Define columns where outliers need to be removed
    columns_to_check_outliers = [
        'number of tablets',
        'Average weight of one loose cut tablet',
        'Active Pharmeutical Ingredient Strength (mg)'
    ]
    for col in columns_to_check_outliers:
        data = remove_outliers(data, col)

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
    data['Packaging Complexity Score'] = data.apply(calculate_complexity_score, axis=1)

    # Calculate Number of Packaging
    data['Number of Packaging'] = (
        (data['Full strips (Y/N)'] == 'Y').astype(int) +  # Adds 1 if "Y"
        (data['Box (Y/N)'] == 'Y').astype(int) +         # Adds 1 if "Y"
        data['Number of rubber band']                   # Adds the number directly
    )

    # Calculate weight per unit box or strip
    data['Weight per Unit Box or Strip'] = (
        data['Average weight of one strip/unit of medication (g)'] / data['no. of tablet/unit per strip']
    )

    # Mean Weight by Packaging Type
    data['Mean Weight Box'] = data.apply(
        lambda row: row['Total weight of tablets without mixed packaging'] / row['number of tablets'] 
                    if row.get('Box (Y/N)', 'N') == 'Y' else 0, 
        axis=1
    )
    data['Mean Weight Strip'] = data.apply(
        lambda row: row['Total weight of tablets without mixed packaging'] / row['number of tablets'] 
                    if row.get('Full strips (Y/N)', 'N') == 'Y' else 0, 
        axis=1
    )
    data['Mean Weight Loose Cut'] = data.apply(
        lambda row: row['Total weight of tablets without mixed packaging'] / row['number of tablets'] 
                    if row.get('Loose Cut (Y/N)', 'N') == 'Y' else 0, 
        axis=1
    )

    # Calculate Weight-to-Strength Ratio
    data['Weight-to-Strength Ratio'] = data.apply(
        lambda row: row['Total weight of tablets without mixed packaging'] / row['Active Pharmeutical Ingredient Strength (mg)']
        if row['Active Pharmeutical Ingredient Strength (mg)'] > 0 else np.nan,
        axis=1
    )

    # Polynomial interactions: squared terms and interaction terms
    data['Total Weight Squared'] = data['Total weight of tablets without mixed packaging'] ** 2
    data['Tablet Count Squared'] = data['number of tablets'] ** 2
    data['Weight x Tablet Count'] = data['Total weight of tablets without mixed packaging'] * data['number of tablets']

    # Z-Score Normalization for key numerical columns
    columns_to_normalize = [
        'Total weight of tablets without mixed packaging',
        'Weight per Unit Box or Strip',
        'number of tablets',
        'Active Pharmeutical Ingredient Strength (mg)',
        'Packaging Complexity Score'
    ]
    for col in columns_to_normalize:
        col_mean = data[col].mean()
        col_std = data[col].std()
        data[f'Normalized {col}'] = (data[col] - col_mean) / col_std

    # Specify only the columns with Y/N values and other non-numeric columns for one-hot encoding
    columns_to_encode = [col for col in data.select_dtypes(include=['object']).columns if col != 'Brand Name']
    data = pd.get_dummies(data, columns=columns_to_encode, drop_first=True)
        
    # Final cleanup: drop any remaining NaN rows introduced during encoding or transformations
    data = data.dropna()

    return data

# File selection utility
def select_file():
    print("Select the Excel file to load the data.")
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    file_path = filedialog.askopenfilename(
        title="Select Excel File",
        filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
    )
    if not file_path:
        print("No file selected. Exiting...")
        exit(1)
    return file_path

# Get file path dynamically
file_path = select_file()
data = load_and_prepare_data(file_path)


# Export function
def export_to_excel(data, filename="cleansed_data.xlsx"):
    # Define download path in the user's Downloads folder
    download_path = Path.home() / "Downloads" / filename
    data.to_excel(download_path, index=False)
    print(f"\nData has been exported to {download_path}")




#--------------Start of Linear Regression------------------
def run_linear_regression_model(data):
    # Set the target column explicitly
    target_column = 'number of tablets'

    # Drop rows with NaN in relevant columns for the regression model
    data_lr = data.dropna(subset=[target_column, 'Average weight of one loose cut tablet'])

    # Define the target variable (y) and feature(s) (X)
    X = data_lr[['Average weight of one loose cut tablet']]
    y = data_lr[target_column]

    # Split the data into training and testing sets (70% training, 30% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Display results
    print("Results of Linear Regression")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared (R2): {r2}")
    print("Intercept:", model.intercept_)
    print("Coefficient:", model.coef_)


#Results for Linear Regression
#Mean Squared Error (MSE): 8370.790097517753
#R-squared (R2): 0.008893335865744367
#Intercept: 106.89725259002493
#Coefficient: [-43.34671131]
# ---------------------------------------------


# ----------------- Start of Gradient Boost -----------------
def run_gradient_boost_model(data):
    # Set the target column explicitly
    target_column = 'number of tablets'

    # Drop rows with NaN in relevant columns for the gradient boosting model
    data_gb = data.dropna()

    # Define the target variable (y) and feature(s) (X) for gradient boosting
    y_gb = data_gb[target_column]
    X_gb = data_gb.drop(columns=[target_column], errors='ignore')

    # Convert categorical variables to numeric if needed (one-hot encoding)
    X_gb = pd.get_dummies(X_gb, drop_first=True)

    # Split the data into training and testing sets (80% training, 20% testing)
    X_train_gb, X_test_gb, y_train_gb, y_test_gb = train_test_split(X_gb, y_gb, test_size=0.2, random_state=42)

    # Initialize and train the Gradient Boosting model with specified parameters
    gb_model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.09,
        max_depth=3,
        subsample=0.8,
        random_state=42
    )
    gb_model.fit(X_train_gb, y_train_gb)

    # Make predictions on the test set
    y_pred_gb = gb_model.predict(X_test_gb)

    # Evaluate the Gradient Boosting model
    mse_gb = mean_squared_error(y_test_gb, y_pred_gb)
    r2_gb = r2_score(y_test_gb, y_pred_gb)

    # Display results for Gradient Boosting
    print("Gradient Boosting Regression Results")
    print(f"Mean Squared Error (MSE): {mse_gb}")
    print(f"R-squared (R2): {r2_gb}")

    
#Gradient Boosting Regression Results
#Mean Squared Error (MSE): 1.225167923565701
#R-squared (R2): 0.9995220142666976
#Best Parameters for Gradient Boosting: {'learning_rate': 0.09, 'max_depth': 3, 'n_estimators': 300, 'subsample': 0.8}
# ---------------------------------------------


#----------------- Start of XGBoost -----------------
def run_xgboost_model(data):
    # Set the target column explicitly
    target_column = 'number of tablets'

    # Drop rows with NaN in relevant columns for the XGBoost model
    data_xgb = data.dropna()

    # Define the target variable (y) and feature(s) (X) for XGBoost
    y_xgb = data_xgb[target_column]
    X_xgb = data_xgb.drop(columns=[target_column], errors='ignore')

    # Convert categorical variables to numeric if needed (one-hot encoding)
    X_xgb = pd.get_dummies(X_xgb, drop_first=True)

    # Split the data into training and testing sets (70% training, 30% testing)
    X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(X_xgb, y_xgb, test_size=0.2, random_state=42)

    # Initialize and train the XGBoost model
    xgb_model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.06,
        max_depth=4,
        subsample=0.72,
        random_state=42
    )
    xgb_model.fit(X_train_xgb, y_train_xgb)

    # Make predictions on the test set
    y_pred_xgb = xgb_model.predict(X_test_xgb)

    # Evaluate the XGBoost model
    mse_xgb = mean_squared_error(y_test_xgb, y_pred_xgb)
    r2_xgb = r2_score(y_test_xgb, y_pred_xgb)

    # Display results for XGBoost
    print("XGBoost Regression Results")
    print(f"Mean Squared Error (MSE): {mse_xgb}")
    print(f"R-squared (R2): {r2_xgb}")
    
#XGBoost Regression Results
#Mean Squared Error (MSE): 1.7244545044523312
#R-squared (R2): 0.9993272231218246
#Best Parameters for XGBoost: {'learning_rate': 0.06, 'max_depth': 4, 'n_estimators': 400, 'subsample': 0.72}
# ---------------------------------------------


#----------------- Start of Random Forest -----------------
def run_random_forest_model(data):
    # Set the target column explicitly
    target_column = 'number of tablets'

    # Drop rows with NaN in relevant columns for the Random Forest model
    data_rf = data.dropna()

    # Define the target variable (y) and feature(s) (X) for Random Forest
    y_rf = data_rf[target_column]
    X_rf = data_rf.drop(columns=[target_column], errors='ignore')

    # Convert categorical variables to numeric if needed (one-hot encoding)
    X_rf = pd.get_dummies(X_rf, drop_first=True)

    # Split the data into training and testing sets (70% training, 30% testing)
    X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.3, random_state=42)

    # Initialize and train the Random Forest model with specified parameters
    rf_model = RandomForestRegressor(
        n_estimators=200,         # Number of trees
        max_depth=None,           # Maximum depth of the tree
        min_samples_split=2,      # Minimum number of samples required to split an internal node
        min_samples_leaf=1,       # Minimum number of samples required to be at a leaf node
        random_state=42
    )
    rf_model.fit(X_train_rf, y_train_rf)

    # Make predictions on the test set
    y_pred_rf = rf_model.predict(X_test_rf)

    # Evaluate the Random Forest model
    mse_rf = mean_squared_error(y_test_rf, y_pred_rf)
    r2_rf = r2_score(y_test_rf, y_pred_rf)

    # Display results for Random Forest
    print("Random Forest Regression Results")
    print(f"Mean Squared Error (MSE): {mse_rf}")
    print(f"R-squared (R2): {r2_rf}")
    
#Random Forest Regression Results
#Mean Squared Error (MSE): 24.811290765765783
#R-squared (R2): 0.9972835171994464
#Best Parameters for Random Forest: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
# ---------------------------------------------



#------------ Linear Regression Grid Search ------------
def run_linear_regression_grid(data):
    print("Running Linear Regression with Polynomial Features Grid Search...")

    # Set target column
    target_column = 'number of tablets'

    # Drop rows with NaN in the relevant columns
    data_lr = data.dropna(subset=[target_column, 'Average weight of one loose cut tablet'])

    # Define target variable (y) and features (X)
    X = data_lr[['Average weight of one loose cut tablet']]
    y = data_lr[target_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define the parameter grid for polynomial degrees
    param_grid_lr = {
        'polynomialfeatures__degree': [1, 2, 3, 4]  # Try different degrees for polynomial features
    }

    # Create a pipeline with PolynomialFeatures and LinearRegression
    pipeline = make_pipeline(PolynomialFeatures(), LinearRegression())
    grid_search_lr = GridSearchCV(pipeline, param_grid_lr, cv=5, scoring='neg_mean_squared_error')
    grid_search_lr.fit(X_train, y_train)

    # Best model results
    y_pred = grid_search_lr.best_estimator_.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Display results
    print("Best Parameters for Linear Regression with Polynomial Features:", grid_search_lr.best_params_)
    print(f"Tuned Mean Squared Error (MSE): {mse}")
    print(f"Tuned R-squared (R2): {r2}")


#------------ Gradient Boost Grid Search ------------
def run_gradient_boost_grid(data):
    print("Running Gradient Boosting Grid Search...")

    # Define target and features for Gradient Boosting
    target_column = 'number of tablets'
    
    # Drop rows with NaN values in the target column
    data_gb = data.dropna(subset=[target_column])
    y_gb = data_gb[target_column]
    
    # Drop Brand Name from features only for model training
    X_gb = data_gb.drop(columns=[target_column, 'Brand Name'], errors='ignore')

    # Split the data into training and testing sets
    X_train_gb, X_test_gb, y_train_gb, y_test_gb = train_test_split(X_gb, y_gb, test_size=0.3, random_state=42)

    # Parameter grid for Gradient Boosting
    param_grid_gb = {
        'n_estimators': [300, 500, 700],
        'learning_rate': [0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
        'max_depth': [3, 4, 5, 6, 7],
        'subsample': [0.7, 0.75, 0.8]
    }

    # Set up GridSearchCV with GradientBoostingRegressor
    grid_search_gb = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_grid_gb, cv=5, scoring='neg_mean_squared_error'
    )
    grid_search_gb.fit(X_train_gb, y_train_gb)

    # Make predictions using the best model found in grid search
    y_pred_gb = grid_search_gb.best_estimator_.predict(X_test_gb)
    mse_gb = mean_squared_error(y_test_gb, y_pred_gb)
    r2_gb = r2_score(y_test_gb, y_pred_gb)

    # Display best parameters and performance metrics
    print("Best Parameters for Gradient Boosting:", grid_search_gb.best_params_)
    print(f"Tuned Mean Squared Error (MSE): {mse_gb}")
    print(f"Tuned R-squared (R2): {r2_gb}")


#------------ XG Boost Grid Search ------------
def run_xgboost_grid(data):
    print("Running XGBoost Grid Search...")

    # Set target and features
    target_column = 'number of tablets'
    y_xgb = data[target_column]
    X_xgb = data.drop(columns=[target_column, 'Brand Name'], errors='ignore')  # Exclude Brand Name from features

    # Split data into training and testing sets
    X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(X_xgb, y_xgb, test_size=0.3, random_state=42)

    # Define parameter grid for Grid Search
    param_grid_xgb = {
        'n_estimators': [100, 200, 300, 400, 500, 600, 700],
        'learning_rate': [0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
        'max_depth': [3, 4, 5, 6],
        'subsample': [0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.78, 0.79, 0.8,]
    }

    # Initialize GridSearchCV with XGBRegressor
    grid_search_xgb = GridSearchCV(
        XGBRegressor(random_state=42, objective='reg:squarederror'),
        param_grid_xgb, cv=5, scoring='neg_mean_squared_error'
    )
    grid_search_xgb.fit(X_train_xgb, y_train_xgb)

    # Make predictions using the best model
    y_pred_xgb = grid_search_xgb.best_estimator_.predict(X_test_xgb)
    mse_xgb = mean_squared_error(y_test_xgb, y_pred_xgb)
    r2_xgb = r2_score(y_test_xgb, y_pred_xgb)

    # Display results
    print("Best Parameters for XGBoost:", grid_search_xgb.best_params_)
    print(f"Tuned Mean Squared Error (MSE): {mse_xgb}")
    print(f"Tuned R-squared (R2): {r2_xgb}")


#------------ Random Forest Grid Search ------------
def run_random_forest_grid(data):
    print("Running Random Forest Grid Search...")

    target_column = 'number of tablets'
    data_rf = data.dropna()

    y_rf = data_rf[target_column]
    X_rf = data_rf.drop(columns=[target_column], errors='ignore')
    X_rf = pd.get_dummies(X_rf, drop_first=True)

    X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.3, random_state=42)

    param_grid_rf = {
        'n_estimators': [100, 200, 300, 500, 700],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=5, scoring='neg_mean_squared_error')
    grid_search_rf.fit(X_train_rf, y_train_rf)

    y_pred_rf = grid_search_rf.best_estimator_.predict(X_test_rf)
    mse_rf = mean_squared_error(y_test_rf, y_pred_rf)
    r2_rf = r2_score(y_test_rf, y_pred_rf)

    print("Best Parameters for Random Forest:", grid_search_rf.best_params_)
    print(f"Tuned Mean Squared Error (MSE): {mse_rf}")
    print(f"Tuned R-squared (R2): {r2_rf}")



    print("Running Support Vector Regression (SVR) Grid Search...")

    target_column = 'number of tablets'
    data_svr = data.dropna()

    y_svr = data_svr[target_column]
    X_svr = data_svr.drop(columns=[target_column], errors='ignore')
    X_svr = pd.get_dummies(X_svr, drop_first=True)

    X_train_svr, X_test_svr, y_train_svr, y_test_svr = train_test_split(X_svr, y_svr, test_size=0.3, random_state=42)

    param_grid_svr = {
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 0.5, 1],
        'kernel': ['linear', 'rbf', 'poly']
    }

    grid_search_svr = GridSearchCV(SVR(), param_grid_svr, cv=5, scoring='neg_mean_squared_error')
    grid_search_svr.fit(X_train_svr, y_train_svr)

    y_pred_svr = grid_search_svr.best_estimator_.predict(X_test_svr)
    mse_svr = mean_squared_error(y_test_svr, y_pred_svr)
    r2_svr = r2_score(y_test_svr, y_pred_svr)

    print("Best Parameters for SVR:", grid_search_svr.best_params_)
    print(f"Tuned Mean Squared Error (MSE): {mse_svr}")
    print(f"Tuned R-squared (R2): {r2_svr}")



# Interface for selecting whether to run a basic model test, a grid search, or view data
def main():
    while True:
        print("\nMain Menu:")
        print("1 - Run Basic Model Test")
        print("2 - Run Grid Search for Hyperparameters")
        print("3 - View Data")
        print("0 - Exit")
        mode_choice = input("Enter your choice (1, 2, 3, or 0): ")

        if mode_choice == '0':
            print("Exiting program. Goodbye!")
            break

        if mode_choice not in ['1', '2', '3']:
            print("Invalid choice. Please select 1, 2, 3, or 0.")
            continue

        if mode_choice == '3':
            while True:
                print("\nData Viewing Options:")
                print("1 - View All Column Names")
                print("2 - View Cleansed and Manipulated Data (First Few Rows)")
                print("3 - Download Cleansed Data to Downloads Folder")
                print("0 - Back to Main Menu")
                view_choice = input("Enter the number of the view option you want to use: ")

                if view_choice == '0':
                    print("Returning to the Main Menu...")
                    break

                if view_choice == '1':
                    # View all column names
                    print("\nColumn Names:")
                    for col in data.columns:
                        print(col)
                elif view_choice == '2':
                    # View the first few rows of the cleansed and manipulated data
                    print("\nData after all cleansing and manipulation (first few rows):")
                    print(data.head())
                elif view_choice == '3':
                    # Save the cleansed data to an Excel file in the Downloads folder
                    downloads_folder = Path.home() / "Downloads"
                    output_file = downloads_folder / "cleansed_data.xlsx"
                    data.to_excel(output_file, index=False)
                    print(f"Data has been downloaded to {output_file}")
                else:
                    print("Invalid view option. Please select a number from 1 to 3, or 0 to go back.")
                
                # After completing any data viewing action, print a message and return to the main menu
                print("\nData viewing action completed. Returning to the Main Menu...")
                break  # Exit the data viewing loop after any action is completed

            continue  # Go back to the main menu after exiting the Data Viewing mode

        # Set the title message based on the chosen mode
        title_message = "Select Model to Run" if mode_choice == '1' else "Select Model to Run Grid Boost"
        
        while True:
            print(f"\n{title_message}")
            print("1 - Linear Regression")
            print("2 - Gradient Boosting")
            print("3 - XGBoost")
            print("4 - Random Forest")
            print("5 - Support Vector Regression (SVR)")
            print("0 - Back to Main Menu")
            model_choice = input("Enter the number of the model you want to run: ")

            if model_choice == '0':
                print("Returning to the Main Menu...")
                break

            if model_choice not in ['1', '2', '3', '4']:
                print("Invalid model choice. Please select a number from 1 to 4, or 0 to go back.")
                continue

            if model_choice == '1':
                if mode_choice == '1':
                    run_linear_regression_model(data)
                else:
                    run_linear_regression_grid(data)
            elif model_choice == '2':
                if mode_choice == '1':
                    run_gradient_boost_model(data)
                else:
                    run_gradient_boost_grid(data)
            elif model_choice == '3':
                if mode_choice == '1':
                    run_xgboost_model(data)
                else:
                    run_xgboost_grid(data)
            elif model_choice == '4':
                if mode_choice == '1':
                    run_random_forest_model(data)
                else:
                    run_random_forest_grid(data)

            # After running the model, print a message and exit the model selection menu to return to the main menu
            print("\nModel run completed. Returning to the Main Menu...")
            break  # Exit the model selection menu

if __name__ == "__main__":
    main()