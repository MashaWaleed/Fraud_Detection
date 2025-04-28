# Credit Card Fraud Detection

A comprehensive machine learning project for detecting fraudulent credit card transactions using various classification models and advanced feature engineering techniques.

## Project Overview

This project implements a robust fraud detection system using supervised machine learning techniques. The system is trained on a dataset of over 1.2 million transactions, with a focus on handling class imbalance and optimizing model performance. The project includes a complete pipeline from data preprocessing to model deployment, with emphasis on interpretability and real-world applicability.

## Key Features

### Data Processing
- Advanced feature engineering including:
  - Transaction time analysis (hour, day, month)
  - Merchant category encoding
  - Transaction amount transformations
  - Geographical feature processing
- Robust data cleaning and preprocessing pipeline
- Class imbalance handling using RandomUnderSampler
- Comprehensive data validation and error handling

### Model Implementation
- Multiple model implementations:
  - Logistic Regression (baseline)
  - Decision Tree (interpretable)
  - Random Forest (ensemble)
  - Gradient Boosting (advanced ensemble)
- Hyperparameter tuning using GridSearchCV
- Comprehensive model evaluation metrics
- Feature importance analysis
- Model persistence and versioning

### Visualization and Analysis
- Interactive visualizations using Plotly and Dash
- Static visualizations using Matplotlib and Seaborn
- Performance metrics visualization
- Feature importance plots
- Confusion matrix analysis
- Fraud pattern analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MashaWaleed/Fraud_Detection.git
cd Fraud_Detection
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
Fraud_Detection/
├── data/
│   ├── fraudTrain.csv
│   └── fraudTest.csv
├── src/
│   ├── data_preprocessing.py    # Data cleaning and feature engineering
│   ├── train_models.py         # Model training and evaluation
│   ├── hyperparameter_tuning.py # Model optimization
│   ├── prediction_pipeline.py  # Production prediction pipeline
│   └── visualization.py        # Data and model visualization
├── visualizations/             # Generated plots and visualizations
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Implementation Details

### Data Preprocessing Pipeline
1. Data Loading and Validation
   - Load raw transaction data
   - Validate data integrity
   - Handle missing values

2. Feature Engineering
   - Time-based features (hour, day, month)
   - Merchant category encoding
   - Transaction amount transformations
   - Geographical feature processing
   - Custom feature creation

3. Data Transformation
   - Standardization of numerical features
   - One-hot encoding of categorical features
   - Class imbalance handling
   - Train-test split

### Model Training Pipeline
1. Model Selection
   - Logistic Regression (baseline)
   - Decision Tree (interpretable)
   - Random Forest (ensemble)
   - Gradient Boosting (advanced)

2. Model Training
   - Cross-validation
   - Hyperparameter tuning
   - Model evaluation
   - Performance metrics calculation

3. Model Selection
   - Compare performance metrics
   - Analyze feature importance
   - Select best performing model

### Prediction Pipeline
1. Data Preprocessing
   - Apply same transformations as training
   - Handle new categories
   - Validate input data

2. Model Prediction
   - Load trained model
   - Generate predictions
   - Calculate prediction probabilities

3. Results Processing
   - Format predictions
   - Generate confidence scores
   - Save results

## Usage

1. Data Preprocessing:
```bash
python src/data_preprocessing.py
```

2. Model Training:
```bash
python src/train_models.py
```

3. Hyperparameter Tuning:
```bash
python src/hyperparameter_tuning.py
```

4. Generate Visualizations:
```bash
python src/visualization.py
```

5. Make Predictions:
```bash
python src/prediction_pipeline.py
```

## Results

### Model Performance
The best performing model (Random Forest) achieved:
- Accuracy: 96.87%
- Precision: 14.98%
- Recall: 94.20%
- F1-score: 25.85%
- ROC-AUC: 99.24%

### Key Findings
1. Time-based features are crucial for fraud detection
2. Transaction amount distribution differs significantly between fraud and non-fraud cases
3. Certain merchant categories show higher fraud rates
4. Geographical location plays a significant role in fraud patterns

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset source: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
- Research paper: [Credit Card Fraud Detection Using Machine Learning](https://www.sciencedirect.com/science/article/pii/S2772662223000036) 