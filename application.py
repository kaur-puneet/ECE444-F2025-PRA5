from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Load the ML model and vectorizer
def load_model_and_vectorizer():
    loaded_model = None 
    with open('basic_classifier.pkl', 'rb') as fid: 
        loaded_model = pickle.load(fid) 
    vectorizer = None
    with open('count_vectorizer.pkl', 'rb') as vd:
        vectorizer = pickle.load(vd)
    return loaded_model, vectorizer

loaded_model, vectorizer = load_model_and_vectorizer()
# Initialize Flask app
application = Flask(__name__)  # for Elastic Beanstalk

# Health check endpoint
@application.route('/')
def home():
    return "Fake News Detection API is running!"

# Prediction endpoint
@application.route('/predict', methods=['POST'])
def predict():
    """
    Expects JSON:{ "text": "Some news text here" }
    Returns JSON:{ "prediction": "FAKE" or "REAL" }
    """
    data = request.get_json(force=True)
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Transform text and predict
    pred_label = loaded_model.predict(vectorizer.transform([text]))[0]

    return jsonify({'prediction': pred_label})

# Run app locally 
if __name__ == '__main__':
    application.run(host='0.0.0.0', port=8000)
