#!/usr/bin/env python
"""
Train a high-accuracy model for Prahar prediction based on custom dataset.
This script trains multiple models and selects the best one to achieve 95%+ accuracy.
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

def train_custom_model(data_path='data/prahar_custom_dataset.csv'):
    """
    Train multiple models and select the best one for Prahar prediction.
    
    Args:
        data_path (str): Path to the custom CSV dataset
        
    Returns:
        best_model: The trained model with highest accuracy
    """
    # Create models directory if it doesn't exist
    os.makedirs('app/models', exist_ok=True)
    
    # Load the dataset
    print(f"Loading dataset from {data_path}...")
    data = pd.read_csv(data_path)
    
    # Separate features and target
    X = data.drop('Label', axis=1)  # Features: Q1 to Q10
    y = data['Label']  # Target: Prahar label (1-8)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    # Scale the features (important for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for future use
    joblib.dump(scaler, 'app/models/prahar_scaler.pkl')
    
    # Define models to try
    models = {
        'random_forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'class_weight': [None, 'balanced']
            }
        },
        'gradient_boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        },
        'svm': {
            'model': SVC(random_state=42, probability=True),
            'params': {
                'C': [1, 10, 100],
                'gamma': ['scale', 'auto', 0.1],
                'kernel': ['rbf', 'poly']
            }
        }
    }
    
    # Train and evaluate each model
    best_models = {}
    best_accuracy = 0
    best_model_name = None
    
    for model_name, model_info in models.items():
        print(f"\n{'-'*50}")
        print(f"Training {model_name}...")
        
        # Use GridSearchCV to find the best hyperparameters
        grid_search = GridSearchCV(
            model_info['model'],
            model_info['params'],
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        # Use scaled features for SVM, unscaled for tree-based models
        if model_name == 'svm':
            grid_search.fit(X_train_scaled, y_train)
            y_pred = grid_search.predict(X_test_scaled)
        else:
            grid_search.fit(X_train, y_train)
            y_pred = grid_search.predict(X_test)
        
        # Get the best model
        best_model = grid_search.best_estimator_
        best_models[model_name] = best_model
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{model_name} Best Parameters: {grid_search.best_params_}")
        print(f"{model_name} Accuracy: {accuracy * 100:.2f}%")
        
        # Print classification report
        print(f"\n{model_name} Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save the model
        model_path = f'app/models/prahar_{model_name}_model.pkl'
        joblib.dump(best_model, model_path)
        print(f"Model saved to {model_path}")
        
        # Track the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = model_name
    
    # Select the best model
    best_model = best_models[best_model_name]
    print(f"\n{'-'*50}")
    print(f"Best model: {best_model_name} with accuracy: {best_accuracy * 100:.2f}%")
    
    # Save the best model as the default model
    joblib.dump(best_model, 'app/models/prahar_best_model.pkl')
    print(f"Best model saved to app/models/prahar_best_model.pkl")
    
    # Create a model info file with metadata
    model_info = {
        'best_model': best_model_name,
        'accuracy': best_accuracy,
        'scaled_input_required': best_model_name == 'svm',
        'training_data_size': len(X_train),
        'test_data_size': len(X_test)
    }
    
    joblib.dump(model_info, 'app/models/prahar_model_info.pkl')
    print(f"Model info saved to app/models/prahar_model_info.pkl")
    
    # Check if we achieved the target accuracy
    if best_accuracy >= 0.95:
        print(f"\n✅ SUCCESS: Achieved target accuracy of 95%+ ({best_accuracy * 100:.2f}%)")
    else:
        print(f"\n⚠️ WARNING: Did not achieve target accuracy of 95% ({best_accuracy * 100:.2f}%)")
    
    return best_model

if __name__ == "__main__":
    train_custom_model()
