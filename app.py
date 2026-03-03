from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)  # Allows your Chrome Extension to talk to this server

print("🧠 Loading the XGBoost Machine Learning Brain...")
try:
    pipeline = joblib.load('xgboost_pipeline.pkl')
    vectorizer = pipeline['vectorizer']
    model = pipeline['model']
    encoder = pipeline['encoder']
    print("✅ ML Brain loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

def calculate_exact_time(text, difficulty):
    """Calculates seconds based on ML difficulty and word count"""
    word_count = len(text.split())
    reading_time = word_count * 0.5  # Assume 0.5 seconds to simply read each word
    
    # Add cognitive load time based on the ML prediction
    if difficulty == 'easy':
        thinking_time = 15
    elif difficulty == 'medium':
        thinking_time = 45
    elif difficulty == 'hard':
        thinking_time = 90
    else:
        thinking_time = 30
        
    total_seconds = round(reading_time + thinking_time)
    return total_seconds

@app.route('/analyze_question', methods=['POST'])
def analyze():
    data = request.json
    question_text = data.get('question', '')
    
    if not question_text:
        return jsonify({"error": "No question provided"}), 400
        
    try:
        # 1. Have the Vectorizer read the text
        X_vec = vectorizer.transform([question_text])
        
        # 2. Have XGBoost predict the difficulty
        pred_num = model.predict(X_vec)[0]
        
        # 3. Translate the math prediction back into a word (easy/medium/hard)
        difficulty = encoder.inverse_transform([pred_num])[0]
        
        # 4. Calculate the final expected time
        expected_time = calculate_exact_time(question_text, difficulty)
        
        return jsonify({
            "status": "success",
            "question": question_text,
            "ml_difficulty": difficulty.upper(),
            "expected_seconds": expected_time
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Time Analyzer Server is running on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)