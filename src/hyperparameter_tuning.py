import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import RandomUnderSampler
import os
from tqdm import tqdm

def load_processed_data():
    """Load the processed data."""
    print("Loading processed data...")
    data = joblib.load('data/processed/processed_data.joblib')
    return data['X_train'], data['X_test'], data['y_train'], data['y_test']

def handle_class_imbalance(X_train, y_train):
    """Handle class imbalance using RandomUnderSampler."""
    print("Handling class imbalance...")
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

def tune_random_forest(X_train, y_train):
    """Tune Random Forest hyperparameters."""
    print("\nTuning Random Forest...")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

def tune_logistic_regression(X_train, y_train):
    """Tune Logistic Regression hyperparameters."""
    print("\nTuning Logistic Regression...")
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    
    lr = LogisticRegression(random_state=42, max_iter=1000)
    grid_search = GridSearchCV(
        estimator=lr,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

def tune_decision_tree(X_train, y_train):
    """Tune Decision Tree hyperparameters."""
    print("\nTuning Decision Tree...")
    param_grid = {
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', None]
    }
    
    dt = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=dt,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

def save_tuned_models(models, params):
    """Save tuned models and their parameters."""
    print("\nSaving tuned models...")
    os.makedirs('models/saved_models', exist_ok=True)
    
    for name, model in models.items():
        joblib.dump(model, f'models/saved_models/{name.lower().replace(" ", "_")}_tuned.joblib')
        joblib.dump(params[name], f'models/saved_models/{name.lower().replace(" ", "_")}_params.joblib')

def main():
    """Main function to run hyperparameter tuning."""
    # Load processed data
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # Handle class imbalance
    X_resampled, y_resampled = handle_class_imbalance(X_train, y_train)
    
    # Tune models
    rf_model, rf_params = tune_random_forest(X_resampled, y_resampled)
    lr_model, lr_params = tune_logistic_regression(X_resampled, y_resampled)
    dt_model, dt_params = tune_decision_tree(X_resampled, y_resampled)
    
    # Save models and parameters
    models = {
        'Random Forest': rf_model,
        'Logistic Regression': lr_model,
        'Decision Tree': dt_model
    }
    
    params = {
        'Random Forest': rf_params,
        'Logistic Regression': lr_params,
        'Decision Tree': dt_params
    }
    
    save_tuned_models(models, params)
    
    print("\nHyperparameter tuning completed successfully!")
    print("\nBest parameters for each model:")
    for name, param in params.items():
        print(f"\n{name}:")
        for key, value in param.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main() 