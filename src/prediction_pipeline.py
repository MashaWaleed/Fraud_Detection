import pandas as pd
import numpy as np
import joblib
import argparse
from datetime import datetime
import os
from tqdm import tqdm

def load_models():
    """Load the preprocessor and best model."""
    print("Loading models...")
    preprocessor = joblib.load('models/preprocessor.joblib')
    model = joblib.load('models/saved_models/random_forest_tuned.joblib')
    feature_names = joblib.load('models/feature_names.joblib')
    return preprocessor, model, feature_names

def preprocess_new_data(df, preprocessor):
    """Preprocess new data using the saved preprocessor."""
    print("Preprocessing new data...")
    
    # Extract time features
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
    df['month'] = df['trans_date_trans_time'].dt.month
    
    # Calculate distance
    R = 6371  # Earth's radius in kilometers
    lat1, lon1 = np.radians(df['lat']), np.radians(df['long'])
    lat2, lon2 = np.radians(df['merch_lat']), np.radians(df['merch_long'])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    df['distance_km'] = R * c
    
    # Create transaction features
    customer_stats = df.groupby('cc_num').agg({
        'amt': ['mean', 'std', 'max', 'min'],
    }).reset_index()
    
    customer_stats.columns = ['cc_num', 'avg_trans_amt', 'std_trans_amt', 
                            'max_trans_amt', 'min_trans_amt']
    
    df = df.merge(customer_stats, on='cc_num', how='left')
    df['amt_relative_to_avg'] = df['amt'] / df['avg_trans_amt']
    df['amt_relative_to_max'] = df['amt'] / df['max_trans_amt']
    
    # Define features
    numeric_features = ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long',
                       'distance_km', 'hour', 'day_of_week', 'month',
                       'avg_trans_amt', 'std_trans_amt', 'max_trans_amt', 'min_trans_amt',
                       'amt_relative_to_avg', 'amt_relative_to_max']
    
    categorical_features = ['category', 'gender', 'state']
    
    # Transform the data
    X = df[numeric_features + categorical_features]
    X_processed = preprocessor.transform(X)
    
    return X_processed, df

def make_predictions(X_processed, model, df):
    """Make predictions using the trained model."""
    print("Making predictions...")
    
    # Get probabilities and predictions
    probabilities = model.predict_proba(X_processed)[:, 1]
    predictions = model.predict(X_processed)
    
    # Create results dataframe
    results = pd.DataFrame({
        'transaction_id': df.index,
        'cc_num': df['cc_num'],
        'amount': df['amt'],
        'merchant': df['merchant'],
        'category': df['category'],
        'fraud_probability': probabilities,
        'is_fraud': predictions
    })
    
    return results

def save_predictions(results, output_file):
    """Save predictions to a CSV file."""
    print(f"Saving predictions to {output_file}...")
    results.to_csv(output_file, index=False)

def main():
    """Main function to run the prediction pipeline."""
    parser = argparse.ArgumentParser(description='Make predictions on new transaction data')
    parser.add_argument('--input', required=True, help='Path to input CSV file')
    parser.add_argument('--output', default='predictions.csv', help='Path to output CSV file')
    args = parser.parse_args()
    
    # Load models
    preprocessor, model, feature_names = load_models()
    
    # Load and preprocess new data
    new_data = pd.read_csv(args.input)
    X_processed, df = preprocess_new_data(new_data, preprocessor)
    
    # Make predictions
    results = make_predictions(X_processed, model, df)
    
    # Save predictions
    save_predictions(results, args.output)
    
    print("\nPrediction pipeline completed successfully!")
    print(f"\nSummary of predictions:")
    print(f"Total transactions: {len(results)}")
    print(f"Predicted fraud cases: {results['is_fraud'].sum()}")
    print(f"Fraud rate: {(results['is_fraud'].sum() / len(results) * 100):.2f}%")

if __name__ == "__main__":
    main() 