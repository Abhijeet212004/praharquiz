#!/usr/bin/env python
"""
Train a simple, high-accuracy model for Prahar prediction based on the specific mapping.
This script creates a decision tree model that directly implements the mapping rules.
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Define the mapping of question answers to Prahars
# This is based on the provided mapping where:
# Q1: A→P1, B→P2, C→P3, D→P4
# Q2: A→P5, B→P6, C→P7, D→P8, etc.
QUESTION_PRAHAR_MAPPING = {
    # Question number: {answer option (0-3): prahar number (1-8)}
    1: {0: 1, 1: 2, 2: 3, 3: 4},
    2: {0: 5, 1: 6, 2: 7, 3: 8},
    3: {0: 1, 1: 2, 2: 3, 3: 4},
    4: {0: 5, 1: 6, 2: 7, 3: 8},
    5: {0: 1, 1: 2, 2: 3, 3: 4},
    6: {0: 5, 1: 6, 2: 7, 3: 8},
    7: {0: 1, 1: 2, 2: 3, 3: 4},
    8: {0: 5, 1: 6, 2: 7, 3: 8},
    9: {0: 1, 1: 2, 2: 3, 3: 4},
    10: {0: 5, 1: 6, 2: 7, 3: 8},
}

def generate_simple_dataset(num_samples=5000):
    """Generate a dataset that directly follows the mapping rules"""
    
    # Generate random answers for each question (values 0-3 representing A, B, C, D options)
    data = {}
    np.random.seed(42)
    for i in range(1, 11):
        data[f'Q{i}'] = np.random.randint(0, 4, size=num_samples)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Count occurrences of each Prahar for each sample
    prahar_counts = {i: np.zeros(num_samples) for i in range(1, 9)}
    
    for idx, row in df.iterrows():
        for q_num in range(1, 11):
            answer = row[f'Q{q_num}']
            prahar = QUESTION_PRAHAR_MAPPING[q_num][answer]
            prahar_counts[prahar][idx] += 1
    
    # Convert to DataFrame for easier analysis
    prahar_count_df = pd.DataFrame(prahar_counts)
    
    # Determine the most frequent Prahar for each sample
    df['Label'] = prahar_count_df.idxmax(axis=1)
    
    return df

def train_simple_model():
    """Train a simple decision tree model that directly implements the mapping"""
    
    # Create models directory if it doesn't exist
    os.makedirs('app/models', exist_ok=True)
    
    print("Generating dataset based on exact Prahar mappings...")
    data = generate_simple_dataset()
    
    # Separate features and target
    X = data.drop('Label', axis=1)  # Features: Q1 to Q10
    y = data['Label']  # Target: Prahar label (1-8)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    # Create a decision tree model with sufficient depth to capture all rules
    model = DecisionTreeClassifier(max_depth=10, random_state=42)
    
    # Train the model
    print("Training decision tree model...")
    model.fit(X_train, y_train)
    
    # Evaluate on the test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model
    model_path = 'app/models/prahar_best_model.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Create a model info file with metadata
    model_info = {
        'model_type': 'decision_tree',
        'accuracy': accuracy,
        'training_data_size': len(X_train),
        'test_data_size': len(X_test),
        'scaled_input_required': False
    }
    
    joblib.dump(model_info, 'app/models/prahar_model_info.pkl')
    print(f"Model info saved to app/models/prahar_model_info.pkl")
    
    # Check if we achieved the target accuracy
    if accuracy >= 0.95:
        print(f"\n✅ SUCCESS: Achieved target accuracy of 95%+ ({accuracy * 100:.2f}%)")
    else:
        print(f"\n⚠️ WARNING: Did not achieve target accuracy of 95% ({accuracy * 100:.2f}%)")
    
    return model

if __name__ == "__main__":
    train_simple_model()
