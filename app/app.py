from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os
import json

app = Flask(__name__)

# Load the trained model (trying multiple model files)
current_dir = os.path.dirname(os.path.abspath(__file__))

# List of model files to try in order of preference
model_files = [
    'models/prahar_random_forest_model.pkl',  # Original model
    'models/prahar_best_model.pkl',          # Rule-based model
]

# Try loading models in order until one succeeds
model = None
for model_file in model_files:
    try:
        model_path = os.path.join(current_dir, model_file)
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(f"Loaded model from {model_path}")
            break
    except Exception as e:
        print(f"Could not load model from {model_path}: {e}")
        continue

# Prahar descriptions with traditional names
PRAHAR_INFO = {
    1: {
        "name": "Pratham Prahar (Dawn Prahar)",
        "description": "You are aligned with the Pratham Prahar, representing new beginnings, fresh energy, and spiritual awakening."
    },
    2: {
        "name": "Dwitiya Prahar (Morning Prahar)",
        "description": "You resonate with the Dwitiya Prahar, symbolizing growth, development, and the building of life foundations."
    },
    3: {
        "name": "Tritiya Prahar (Midday Prahar)",
        "description": "The Tritiya Prahar matches your personality, representing balance, harmony, and the fullness of life's experiences."
    },
    4: {
        "name": "Chaturtha Prahar (Afternoon Prahar)",
        "description": "You align with the Chaturtha Prahar, characterized by stability, foundation, and the strong manifestation of your life purpose."
    },
    5: {
        "name": "Pancham Prahar (Evening Prahar)",
        "description": "The Pancham Prahar reflects your nature, representing transformation, change, and wisdom gained through experience."
    },
    6: {
        "name": "Shastha Prahar (Sunset Prahar)",
        "description": "You connect with the Shastha Prahar, symbolizing communication, expression, and the sharing of accumulated knowledge."
    },
    7: {
        "name": "Saptam Prahar (Twilight Prahar)",
        "description": "The Saptam Prahar matches your essence, representing spirituality, wisdom, and the transition to higher consciousness."
    },
    8: {
        "name": "Ashtam Prahar (Night Prahar)",
        "description": "You align with the Ashtam Prahar, characterized by completion, mastery, and the deep rest that prepares for new beginnings."
    }
}

# Quiz questions and options
QUIZ_QUESTIONS = [
    {
        "id": 1,
        "question": "Which activity do you enjoy most during your free time?",
        "options": ["Reading a book", "Outdoor adventures", "Creative pursuits", "Social gatherings"]
    },
    {
        "id": 2,
        "question": "What type of environment helps you feel most productive?",
        "options": ["Quiet and organized space", "Bustling and energetic atmosphere", "Natural surroundings", "Collaborative setting"]
    },
    {
        "id": 3,
        "question": "How do you typically approach challenges?",
        "options": ["Analyze methodically", "Take immediate action", "Seek advice from others", "Trust your intuition"]
    },
    {
        "id": 4,
        "question": "Which quality do you value most in relationships?",
        "options": ["Loyalty", "Honesty", "Understanding", "Growth"]
    },
    {
        "id": 5,
        "question": "What time of day do you feel most energetic?",
        "options": ["Early morning", "Mid-day", "Evening", "Late night"]
    },
    {
        "id": 6,
        "question": "How do you prefer to learn new information?",
        "options": ["Reading/visual materials", "Hands-on experience", "Listening to experts", "Group discussion"]
    },
    {
        "id": 7,
        "question": "Which element resonates with you most?",
        "options": ["Earth", "Water", "Air", "Fire"]
    },
    {
        "id": 8,
        "question": "How do you make important decisions?",
        "options": ["Logic and reason", "Emotional intuition", "Weighing pros and cons", "Seeking advice"]
    },
    {
        "id": 9,
        "question": "What do you value most in your life path?",
        "options": ["Security and stability", "Growth and challenges", "Balance and harmony", "Purpose and meaning"]
    },
    {
        "id": 10,
        "question": "How do others typically describe your personality?",
        "options": ["Calm and collected", "Energetic and passionate", "Thoughtful and analytical", "Adaptable and flexible"]
    }
]

@app.route('/')
def index():
    """Render the quiz page"""
    return render_template('index.html', questions=QUIZ_QUESTIONS)

@app.route('/quiz-data')
def quiz_data():
    """Return quiz questions and options as JSON"""
    return jsonify({
        'questions': QUIZ_QUESTIONS
    })

# Fallback rule-based model implementation
class FallbackRuleBasedModel:
    """A rule-based model that directly implements the mapping of question answers to Prahars"""
    
    def __init__(self):
        # Define the mapping of question answers to Prahars
        # Odd-numbered questions: A→P1, B→P2, C→P3, D→P4
        # Even-numbered questions: A→P5, B→P6, C→P7, D→P8
        self.question_prahar_mapping = {
            # Question number (1-indexed): {answer option (0-3): prahar number (1-8)}
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
    
    def predict(self, X):
        """Predict the Prahar based on the answers to the 10 questions"""
        # Convert to numpy array if it's a list
        if isinstance(X, list):
            X = np.array(X).reshape(1, -1)
        
        # Initialize predictions array
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples, dtype=int)
        
        # For each sample, count the occurrences of each Prahar
        for i in range(n_samples):
            # Count occurrences of each Prahar
            prahar_counts = {p: 0 for p in range(1, 9)}
            
            # Go through each question and increment the count for the corresponding Prahar
            for q in range(10):
                q_num = q + 1  # Convert to 1-indexed
                answer = X[i, q]
                
                # Handle both numeric and letter inputs
                if isinstance(answer, str) and answer in ['A', 'B', 'C', 'D']:
                    answer = ord(answer) - ord('A')  # Convert A->0, B->1, etc.
                
                # Get the Prahar for this question and answer
                prahar = self.question_prahar_mapping[q_num][answer]
                prahar_counts[prahar] += 1
            
            # Find the Prahar with the highest count
            predictions[i] = max(prahar_counts, key=prahar_counts.get)
        
        return predictions

@app.route('/predict', methods=['POST'])
def predict():
    """Predict Prahar based on quiz answers"""
    # If model not loaded, create a fallback rule-based model
    global model
    if model is None:
        try:
            # Try loading the model one more time
            for model_file in model_files:
                model_path = os.path.join(current_dir, model_file)
                if os.path.exists(model_path):
                    model = joblib.load(model_path)
                    print(f"Loaded model from {model_path}")
                    break
            
            # If still no model, use the fallback
            if model is None:
                print("Using fallback rule-based model")
                model = FallbackRuleBasedModel()
        except Exception as e:
            print(f"Error loading model, using fallback: {e}")
            model = FallbackRuleBasedModel()
    
    # Get quiz answers from request
    data = request.get_json()
    answers = data.get('answers')
    
    if not answers or len(answers) != 10:
        return jsonify({
            'error': 'Invalid input. Please provide answers to all 10 questions.'
        }), 400
    
    try:
        # Convert answers to numpy array for prediction
        input_data = np.array(answers).reshape(1, -1)
        
        # Make prediction
        predicted_prahar = int(model.predict(input_data)[0])
        
        # Get Prahar information
        prahar_info = PRAHAR_INFO.get(predicted_prahar, {
            "name": f"Prahar {predicted_prahar}",
            "description": "Your unique personality aligns with this Prahar."
        })
        
        return jsonify({
            'prahar': predicted_prahar,
            'prahar_name': prahar_info['name'],
            'description': prahar_info['description']
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Use PORT environment variable if available (for cloud deployment)
    port = int(os.environ.get('PORT', 5005))
    app.run(debug=True, port=port, host='0.0.0.0') 