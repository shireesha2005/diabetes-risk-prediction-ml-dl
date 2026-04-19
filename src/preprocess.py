import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load Dataset
def load_data():
    # Using raw string (r"") to avoid path errors
    path = r"C:\Projects\Healthcare_NLP_Project\data\diabetes.csv"
    df = pd.read_csv(path)
    return df


# Explore Data
def explore_data(df):
    print("\nFirst 5 rows:\n", df.head())
    print("\nDataset Info:\n")
    print(df.info())
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nStatistical Summary:\n", df.describe())


# 3 Preprocess Data
def preprocess_data(df):
    # Replace invalid zero values with median
    columns_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    for col in columns_with_zero:
        df[col] = df[col].replace(0, df[col].median())

    # Split features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-Test Split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, scaler
