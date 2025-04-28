# Credit Card Fraud Detection

A machine learning project for detecting fraudulent credit card transactions using various classification models.

## Project Overview

This project aims to detect and predict fraudulent credit card transactions using supervised machine learning techniques. The system is trained on a dataset of over 1.2 million transactions, with a focus on handling class imbalance and optimizing model performance.

## Features

- Data preprocessing and feature engineering
- Class imbalance handling using RandomUnderSampler
- Multiple model implementations (Logistic Regression, Decision Tree, Random Forest)
- Hyperparameter tuning using GridSearchCV
- Comprehensive model evaluation metrics
- Feature importance analysis
- Prediction pipeline for new transactions
- Interactive visualizations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
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
credit-card-fraud-detection/
├── data/
│   ├── fraudTrain.csv
│   └── fraudTest.csv
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── hyperparameter_tuning.py
│   ├── feature_engineering.py
│   └── prediction_pipeline.py
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   └── model_comparison.ipynb
├── models/
│   └── saved_models/
├── visualizations/
├── tests/
├── requirements.txt
└── README.md
```

## Usage

1. Data Preprocessing:
```bash
python src/data_preprocessing.py
```

2. Model Training:
```bash
python src/model_training.py
```

3. Hyperparameter Tuning:
```bash
python src/hyperparameter_tuning.py
```

4. Make Predictions:
```bash
python src/prediction_pipeline.py --input new_transactions.csv
```

## Results

The best performing model (Random Forest) achieved:
- Accuracy: 96.87%
- Precision: 14.98%
- Recall: 94.20%
- F1-score: 25.85%
- ROC-AUC: 99.24%

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