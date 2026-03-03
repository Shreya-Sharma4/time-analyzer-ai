from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

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
    """Stricter calculation tuned for Multiple Choice Questions"""
    word_count = len(text.split())
    reading_time = word_count * 0.75  # 0.75 seconds to read each word
    
    # Stricter thinking times for MCQs
    if difficulty == 'easy':
        thinking_time = 10   # Lowered from 15
    elif difficulty == 'medium':
        thinking_time = 25   # Lowered from 45
    elif difficulty == 'hard':
        thinking_time = 50   # Lowered from 90
    else:
        thinking_time = 20
        
    total_seconds = round(reading_time + thinking_time)
    return total_seconds

@app.route('/analyze_question', methods=['POST'])
def analyze():
    data = request.json
    question_text = data.get('question', '')
    
    if not question_text:
        return jsonify({"error": "No question provided"}), 400
        
    try:
        X_vec = vectorizer.transform([question_text])
        pred_num = model.predict(X_vec)[0]
        difficulty = encoder.inverse_transform([pred_num])[0].upper()
        
        expected_time = calculate_exact_time(question_text, difficulty.lower())
        
        return jsonify({
            "status": "success",
            "question": question_text,
            "ml_difficulty": difficulty,
            "expected_seconds": expected_time
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Time Analyzer Server is running...")
    app.run(debug=True, port=5000)
