import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output
import joblib
import os
from datetime import datetime

def load_data():
    """Load the processed data and model results."""
    print("Loading data...")
    data = joblib.load('data/processed/processed_data.joblib')
    results = joblib.load('model_results.csv')
    return data, results

def create_fraud_distribution_plot(df):
    """Create a plot showing the distribution of fraud cases."""
    fig = px.pie(df, names='is_fraud', title='Distribution of Fraudulent vs Non-Fraudulent Transactions')
    return fig

def create_amount_distribution_plot(df):
    """Create a plot showing the distribution of transaction amounts."""
    fig = px.histogram(df, x='amt', color='is_fraud', 
                      title='Distribution of Transaction Amounts by Fraud Status',
                      log_x=True)
    return fig

def create_time_analysis_plot(df):
    """Create a plot showing fraud patterns over time."""
    df['hour'] = pd.to_datetime(df['trans_date_trans_time']).dt.hour
    fraud_by_hour = df.groupby(['hour', 'is_fraud']).size().reset_index(name='count')
    
    fig = px.line(fraud_by_hour, x='hour', y='count', color='is_fraud',
                 title='Transaction Frequency by Hour and Fraud Status')
    return fig

def create_category_analysis_plot(df):
    """Create a plot showing fraud patterns by category."""
    fraud_by_category = df.groupby(['category', 'is_fraud']).size().reset_index(name='count')
    
    fig = px.bar(fraud_by_category, x='category', y='count', color='is_fraud',
                title='Transaction Frequency by Category and Fraud Status')
    return fig

def create_distance_analysis_plot(df):
    """Create a plot showing fraud patterns by distance."""
    fig = px.box(df, x='is_fraud', y='distance_km',
                title='Distribution of Transaction Distances by Fraud Status')
    return fig

def create_model_comparison_plot(results):
    """Create a plot comparing model performance."""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC']
    models = results.index
    
    fig = go.Figure()
    for metric in metrics:
        fig.add_trace(go.Bar(
            name=metric,
            x=models,
            y=results[metric],
            text=results[metric].round(3),
            textposition='auto',
        ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Model',
        yaxis_title='Score',
        barmode='group'
    )
    return fig

def create_dashboard():
    """Create an interactive dashboard using Dash."""
    app = Dash(__name__)
    
    # Load data
    data, results = load_data()
    df = pd.DataFrame(data['X_train'], columns=joblib.load('models/feature_names.joblib'))
    df['is_fraud'] = data['y_train']
    
    app.layout = html.Div([
        html.H1('Credit Card Fraud Detection Analysis'),
        
        html.Div([
            html.H2('Transaction Overview'),
            dcc.Graph(figure=create_fraud_distribution_plot(df)),
            dcc.Graph(figure=create_amount_distribution_plot(df)),
        ]),
        
        html.Div([
            html.H2('Temporal Analysis'),
            dcc.Graph(figure=create_time_analysis_plot(df)),
        ]),
        
        html.Div([
            html.H2('Category Analysis'),
            dcc.Graph(figure=create_category_analysis_plot(df)),
        ]),
        
        html.Div([
            html.H2('Geographical Analysis'),
            dcc.Graph(figure=create_distance_analysis_plot(df)),
        ]),
        
        html.Div([
            html.H2('Model Performance'),
            dcc.Graph(figure=create_model_comparison_plot(results)),
        ]),
    ])
    
    return app

def save_visualizations():
    """Save static visualizations as HTML files."""
    print("Creating and saving visualizations...")
    data, results = load_data()
    df = pd.DataFrame(data['X_train'], columns=joblib.load('models/feature_names.joblib'))
    df['is_fraud'] = data['y_train']
    
    os.makedirs('visualizations', exist_ok=True)
    
    # Save individual plots
    create_fraud_distribution_plot(df).write_html('visualizations/fraud_distribution.html')
    create_amount_distribution_plot(df).write_html('visualizations/amount_distribution.html')
    create_time_analysis_plot(df).write_html('visualizations/time_analysis.html')
    create_category_analysis_plot(df).write_html('visualizations/category_analysis.html')
    create_distance_analysis_plot(df).write_html('visualizations/distance_analysis.html')
    create_model_comparison_plot(results).write_html('visualizations/model_comparison.html')

def main():
    """Main function to run visualization pipeline."""
    # Save static visualizations
    save_visualizations()
    
    # Create and run interactive dashboard
    app = create_dashboard()
    app.run_server(debug=True)

if __name__ == "__main__":
    main() 