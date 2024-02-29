from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return "Welcome to the Iris Classifier API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = [data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']]
    prediction = model.predict([features])[0]
    target_names = ['setosa', 'versicolor', 'virginica']
    result = {'predicted_class': target_names[prediction]}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
