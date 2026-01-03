from flask import Flask,request,jsonify
#Flask - create server
#request - receives data from user (post,put,....)
#jsonify - usee to send response back (JSON)

from flask_cors import CORS


import joblib
# load pkl file


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# create Flask Server
# server --> app
# app --> GET,POST,PUT, DELETE
# inform to server app.py is the main file

# load ai model
model = joblib.load("my_model.pkl")
# model loads only once, not loads every request, model performance will increase

# Dummy GET Request
@app.route("/")
def home():
    return "welcome to first NLP application API"

@app.route("/predict",methods=["POST"])
def predict():
    data = request.get_json()
    size = data["size"]
    bedrooms = data["bedrooms"]

    prediction = model.predict([[size,bedrooms]])

    return jsonify({"price":prediction[0]})

# start flask server
# Auto restart when ever changes are detected
# show errors clearly
if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)
    # app.run(debug=True)
