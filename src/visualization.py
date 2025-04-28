import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data():
    """Load the data and model results."""
    print("Loading data...")
    data = pd.read_csv(os.path.join('..', 'data', 'fraudTrain.csv'))
    results = pd.read_csv(os.path.join('..', 'data', 'predictions.csv'))
    return data, results

def plot_fraud_distribution(data):
    """Plot the distribution of fraud vs non-fraud transactions."""
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x='is_fraud')
    plt.title('Distribution of Fraud vs Non-Fraud Transactions')
    plt.xlabel('Is Fraud (0: No, 1: Yes)')
    plt.ylabel('Number of Transactions')
    plt.savefig(os.path.join('..', 'visualizations', 'fraud_distribution.png'))
    plt.close()

def plot_amount_distribution(data):
    """Plot the distribution of transaction amounts for fraud vs non-fraud."""
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data, x='is_fraud', y='amt')
    plt.title('Transaction Amount Distribution by Fraud Status')
    plt.xlabel('Is Fraud (0: No, 1: Yes)')
    plt.ylabel('Transaction Amount')
    plt.savefig(os.path.join('..', 'visualizations', 'amount_distribution.png'))
    plt.close()

def plot_fraud_by_category(data):
    """Plot fraud rates by merchant category."""
    fraud_by_category = data.groupby('category')['is_fraud'].mean().sort_values(ascending=False)
    plt.figure(figsize=(15, 6))
    fraud_by_category.plot(kind='bar')
    plt.title('Fraud Rate by Merchant Category')
    plt.xlabel('Category')
    plt.ylabel('Fraud Rate')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join('..', 'visualizations', 'fraud_by_category.png'))
    plt.close()

def plot_hourly_fraud_rate(data):
    """Plot fraud rates by hour of day."""
    data['hour'] = pd.to_datetime(data['trans_date_trans_time']).dt.hour
    hourly_fraud = data.groupby('hour')['is_fraud'].mean()
    
    plt.figure(figsize=(12, 6))
    hourly_fraud.plot(kind='line', marker='o')
    plt.title('Fraud Rate by Hour of Day')
    plt.xlabel('Hour')
    plt.ylabel('Fraud Rate')
    plt.grid(True)
    plt.savefig(os.path.join('..', 'visualizations', 'hourly_fraud_rate.png'))
    plt.close()

def plot_geographical_distribution(data):
    """Plot geographical distribution of fraud transactions."""
    plt.figure(figsize=(15, 10))
    plt.scatter(data[data['is_fraud']==0]['long'].sample(1000), 
               data[data['is_fraud']==0]['lat'].sample(1000), 
               alpha=0.5, label='Non-Fraud', c='blue')
    plt.scatter(data[data['is_fraud']==1]['long'], 
               data[data['is_fraud']==1]['lat'], 
               alpha=0.5, label='Fraud', c='red')
    plt.title('Geographical Distribution of Transactions')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.savefig(os.path.join('..', 'visualizations', 'geographical_distribution.png'))
    plt.close()

def save_visualizations():
    """Create and save all visualizations."""
    print("Creating and saving visualizations...")
    
    # Load data
    data, results = load_data()
    
    # Create visualizations directory if it doesn't exist
    os.makedirs(os.path.join('..', 'visualizations'), exist_ok=True)
    
    # Generate visualizations
    plot_fraud_distribution(data)
    plot_amount_distribution(data)
    plot_fraud_by_category(data)
    plot_hourly_fraud_rate(data)
    plot_geographical_distribution(data)
    
    print("Visualizations saved successfully!")

def main():
    """Main function to run the visualization pipeline."""
    save_visualizations()

if __name__ == "__main__":
    main() 