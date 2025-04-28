import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os
from datetime import datetime
from tqdm import tqdm

def load_data(file_path):
    """Load and preprocess the dataset."""
    print("Loading dataset...")
    df = pd.read_csv(os.path.join('..', file_path))
    return df

def extract_time_features(df):
    """Extract time-based features from transaction timestamp."""
    print("Extracting time features...")
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
    df['month'] = df['trans_date_trans_time'].dt.month
    return df

def calculate_distance(df):
    """Calculate distance between customer and merchant locations."""
    print("Calculating distances...")
    # Using Haversine formula for distance calculation
    R = 6371  # Earth's radius in kilometers
    
    lat1, lon1 = np.radians(df['lat']), np.radians(df['long'])
    lat2, lon2 = np.radians(df['merch_lat']), np.radians(df['merch_long'])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    df['distance_km'] = R * c
    return df

def create_transaction_features(df):
    """Create transaction-based features."""
    print("Creating transaction features...")
    
    # Group by customer and calculate statistics
    customer_stats = df.groupby('cc_num').agg({
        'amt': ['mean', 'std', 'max', 'min'],
        'is_fraud': 'count'
    }).reset_index()
    
    customer_stats.columns = ['cc_num', 'avg_trans_amt', 'std_trans_amt', 
                            'max_trans_amt', 'min_trans_amt', 'trans_count']
    
    # Merge back to original dataframe
    df = df.merge(customer_stats, on='cc_num', how='left')
    
    # Calculate relative transaction amount
    df['amt_relative_to_avg'] = df['amt'] / df['avg_trans_amt']
    df['amt_relative_to_max'] = df['amt'] / df['max_trans_amt']
    
    return df

def preprocess_data(df):
    """Main preprocessing function."""
    print("Starting data preprocessing...")
    
    # Extract time features
    df = extract_time_features(df)
    
    # Calculate distance
    df = calculate_distance(df)
    
    # Create transaction features
    df = create_transaction_features(df)
    
    # Define features
    numeric_features = ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long',
                       'distance_km', 'hour', 'day_of_week', 'month',
                       'avg_trans_amt', 'std_trans_amt', 'max_trans_amt', 'min_trans_amt',
                       'trans_count', 'amt_relative_to_avg', 'amt_relative_to_max']
    
    categorical_features = ['category', 'gender', 'state']
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Separate features and target
    X = df[numeric_features + categorical_features]
    y = df['is_fraud']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Fit and transform the data
    print("Fitting preprocessing pipeline...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Save the preprocessor
    os.makedirs('models', exist_ok=True)
    joblib.dump(preprocessor, 'models/preprocessor.joblib')
    
    # Save feature names
    feature_names = (numeric_features + 
                    list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)))
    joblib.dump(feature_names, 'models/feature_names.joblib')
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor

def main():
    """Main function to run preprocessing."""
    # Load data
    df = load_data('data/fraudTrain.csv')
    
    # Preprocess data
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    joblib.dump({
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }, 'data/processed/processed_data.joblib')
    
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    main() 