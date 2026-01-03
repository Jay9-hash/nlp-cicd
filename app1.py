from flask import Flask, request, jsonify
import re
import joblib
import nltk

# NLTK setup
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load model & vectorizer
model = joblib.load('my_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Preprocessing objects
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Flask app
app = Flask(__name__)

# Text preprocessing function (same as training)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(words)

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if 'text' not in data:
        return jsonify({'error': 'Please provide text'}), 400

    cleaned_text = preprocess_text(data['text'])
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)[0]

    result = "Positive" if prediction == 1 else "Negative"

    return jsonify({
        'input_text': data['text'],
        'prediction': result
    })

# Health check
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'API is running'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)








#===========================================================

# from flask import Flask,request,jsonify
# #Flask - create server
# #request - receives data from user (post,put,....)
# #jsonify - usee to send response back (JSON)

# from flask_cors import CORS


# import joblib
# # load pkl file


# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "*"}})

# # create Flask Server
# # server --> app
# # app --> GET,POST,PUT, DELETE
# # inform to server app.py is the main file

# # load ai model
# model = joblib.load("my_model.pkl")
# # model loads only once, not loads every request, model performance will increase

# # Dummy GET Request
# @app.route("/")
# def home():
#     return "welcome to first NLP application API"

# @app.route("/predict",methods=["POST"])
# def predict():
#     data = request.get_json()
#     size = data["size"]
#     bedrooms = data["bedrooms"]

#     prediction = model.predict([[size,bedrooms]])

#     return jsonify({"price":prediction[0]})

# # start flask server
# # Auto restart when ever changes are detected
# # show errors clearly
# if __name__ == "__main__":
#     app.run(host="0.0.0.0",port=5000,debug=True)
#     # app.run(debug=True)
