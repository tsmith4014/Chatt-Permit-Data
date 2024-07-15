# import pandas as pd

# def read_csv_headers(file_path):
#     """
#     Reads the column headers from a CSV file.

#     Parameters:
#     file_path (str): The path to the CSV file.

#     Returns:
#     list: A list of column headers.
#     """
#     try:
#         # Read only the first row to get the headers
#         df = pd.read_csv(file_path, nrows=0)
#         headers = df.columns.tolist()
#         return headers
#     except pd.errors.ParserError as e:
#         print(f"Error parsing file: {e}")
#         return []

# # Example usage
# file_path = 'Chatt-2024-Permit-Data.csv'
# headers = read_csv_headers(file_path)
# print(headers)



# import pandas as pd

# def infer_column_types(file_path, sample_size=1000):
#     """
#     Infers the data types of each column in a CSV file.

#     Parameters:
#     file_path (str): The path to the CSV file.
#     sample_size (int): The number of rows to read for inferring data types.

#     Returns:
#     dict: A dictionary mapping column names to their inferred data types.
#     """
#     try:
#         # Read a sample of the data
#         df = pd.read_csv(file_path, nrows=sample_size)
        
#         # Infer data types
#         column_types = df.dtypes.apply(lambda x: x.name).to_dict()
        
#         return column_types
#     except pd.errors.ParserError as e:
#         print(f"Error parsing file: {e}")
#         return {}

# # Example usage
# file_path = 'Chatt-2024-Permit-Data.csv'
# column_types = infer_column_types(file_path)
# print(column_types)




import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer

def load_data(file_path):
    dtype_spec = {
        'column_4': 'str',
        'column_27': 'str',
        'column_30': 'str',
        'column_33': 'str',
        'column_34': 'str',
        'column_35': 'str',
        'column_36': 'str',
        'column_37': 'str',
        'column_41': 'str',
        'column_42': 'str',
        'column_50': 'str',
        'column_59': 'str',
        'column_61': 'str'
    }
    return pd.read_csv(file_path, dtype=dtype_spec, low_memory=False)

def preprocess_data(df):
    # Drop columns with all missing values
    df.dropna(axis=1, how='all', inplace=True)
    
    # Handle missing values for categorical columns
    df.fillna(method='ffill', inplace=True)
    
    # Convert date columns to datetime and extract features
    date_columns = ['AppliedDate', 'IssuedDate', 'CompletedDate', 'StatusDate', 'ExpiresDate', 'COIssuedDate', 'HoldDate', 'VoidDate']
    for col in date_columns:
        if (col in df.columns) and (df[col].dtype == 'object'):
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            df.drop(columns=[col], inplace=True)
    
    # Encode categorical variables
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le
    
    # Handle missing values for numerical columns
    imputer = SimpleImputer(strategy='mean')
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
    
    return df, label_encoders

def feature_selection(df):
    # Check if 'StatusCurrent' exists in the DataFrame
    if 'StatusCurrent' not in df.columns:
        print("Available columns:", df.columns)
        raise KeyError("'StatusCurrent' column not found in the DataFrame. Please check the column name.")
    
    # Select relevant features (example: predicting 'StatusCurrent')
    features = df.drop(columns=['StatusCurrent'])
    target = df['StatusCurrent']
    return features, target

def split_data(features, target):
    return train_test_split(features, target, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    # Identify numerical columns
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, numerical_cols

def evaluate_model(model, scaler, X_test, y_test, numerical_cols):
    X_test_scaled = X_test.copy()
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
    y_pred = model.predict(X_test_scaled)
    
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

def main():
    file_path = 'Chatt-2024-Permit-Data.csv'
    
    # Load data
    df = load_data(file_path)
    
    # Preprocess data
    df, label_encoders = preprocess_data(df)
    
    # Print columns to debug
    print("Columns in DataFrame:", df.columns)
    
    # Feature selection
    features, target = feature_selection(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(features, target)
    
    # Train model
    model, scaler, numerical_cols = train_model(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, scaler, X_test, y_test, numerical_cols)
    
    # Decode and print class labels
    status_encoder = label_encoders['StatusCurrent']
    print("Class labels and their corresponding original values:")
    for class_label in sorted(status_encoder.classes_):
        print(f"Class {status_encoder.transform([class_label])[0]}: {class_label}")

if __name__ == "__main__":
    main()