import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train_model(data_path='data/prahar_quiz_dataset.csv', model_type='random_forest'):
    """
    Train a machine learning model to predict Prahar based on quiz answers.
    
    Args:
        data_path (str): Path to the CSV dataset
        model_type (str): Type of model to train ('random_forest' or 'logistic')
        
    Returns:
        model: Trained scikit-learn model
    """
    # Load the dataset
    data = pd.read_csv(data_path)
    
    # Separate features and target
    X = data.drop('Label', axis=1)  # Features: Q1 to Q10
    y = data['Label']  # Target: Prahar label (1-8)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize the selected model
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10,
            random_state=42
        )
    else:  # logistic regression
        model = LogisticRegression(
            max_iter=1000,
            multi_class='multinomial',
            solver='lbfgs',
            random_state=42
        )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate on the test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"\nCross-Validation Accuracy: {np.mean(cv_scores) * 100:.2f}% ± {np.std(cv_scores) * 100:.2f}%")
    
    # Create directory if it doesn't exist
    os.makedirs('app/models', exist_ok=True)
    
    # Save the model
    model_path = f'app/models/prahar_{model_type}_model.pkl'
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")
    
    return model

if __name__ == "__main__":
    # Train both models and compare
    print("Training Random Forest model...")
    rf_model = train_model(model_type='random_forest')
    
    print("\n" + "="*50 + "\n")
    
    print("Training Logistic Regression model...")
    lr_model = train_model(model_type='logistic') 