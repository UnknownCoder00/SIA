import pandas as pd
from joblib import load
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load model
model = load("decision_tree_model.joblib")

# List of expected features (must match training!)
expected_features = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]

# Flask app
api = Flask(__name__)
CORS(api)

# Mapping from snake_case to training column names
rename_map = {
    'fixed_acidity': 'fixed acidity',
    'volatile_acidity': 'volatile acidity',
    'citric_acid': 'citric acid',
    'residual_sugar': 'residual sugar',
    'chlorides': 'chlorides',
    'free_sulfur_dioxide': 'free sulfur dioxide',
    'total_sulfur_dioxide': 'total sulfur dioxide',
    'density': 'density',
    'pH': 'pH',
    'sulphates': 'sulphates',
    'alcohol': 'alcohol'
}

@api.route('/predict/hfp_prediction', methods=['POST'])
def predict_heart_failure():
    data = request.json

    # Insert dummy Id directly in input
    data['Id'] = 0   # <<< add here inside the dictionary itself

    input_df = pd.DataFrame([data])

    print("Received Data:", input_df.columns.tolist())
    
    input_df.rename(columns=rename_map, inplace=True)

    # Reorder features to match model expectation
    input_df = input_df[expected_features + ['Id']]  # <<< NOTE: Id FIRST, then features

    # Predict
    prediction = model.predict(input_df)

    return jsonify({'prediction': int(prediction[0])})


if __name__ == '__main__':
    api.run(debug=true, host='0.0.0.0')
