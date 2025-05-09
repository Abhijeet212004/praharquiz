from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os
import json

app = Flask(__name__)

# Load the trained model (default to random forest)
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'models/prahar_random_forest_model.pkl')
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None

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

@app.route('/predict', methods=['POST'])
def predict():
    """Predict Prahar based on quiz answers"""
    # If model not loaded, try loading again
    global model
    if model is None:
        try:
            model = joblib.load(model_path)
        except Exception as e:
            return jsonify({
                'error': f'Model not found. Please train the model first. Error: {str(e)}'
            }), 500
    
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
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=True, port=port, host='0.0.0.0') 