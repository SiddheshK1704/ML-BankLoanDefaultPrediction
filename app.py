from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

app = Flask(__name__)

# Global variables
model = None
label_encoders = {}
feature_names = []

def load_and_preprocess_data():
    """Load and preprocess the German Credit Data from local file"""
    # Column names for German Credit Data
    columns = [
        'existing_account', 'duration', 'credit_history', 'purpose', 'credit_amount',
        'savings', 'employment', 'installment_rate', 'personal_status', 'debtors',
        'residence', 'property', 'age', 'other_plans', 'housing',
        'existing_credits', 'job', 'dependents', 'telephone', 'foreign_worker', 'class'
    ]
    
    # Load data from local file
    # The file should be in the same directory as app.py or provide full path
    try:
        df = pd.read_csv('german.data', sep=' ', names=columns, header=None)
        print(f"Dataset loaded successfully! Shape: {df.shape}")
    except FileNotFoundError:
        print("Error: german.data file not found!")
        print("Please download it from: https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data")
        raise
    
    # Convert class to binary (1=good, 2=bad) -> (0=good, 1=bad)
    df['class'] = df['class'].map({1: 0, 2: 1})
    
    return df

def train_model():
    """Train Random Forest model"""
    global model, label_encoders, feature_names
    
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data()
    
    # Separate features and target
    X = df.drop('class', axis=1)
    y = df['class']
    
    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    feature_names = X.columns.tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Random Forest
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print(f"Model trained successfully!")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Good Credit', 'Default Risk']))
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"{'='*50}\n")
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 Most Important Features:")
    print(feature_importance.head(10))
    print(f"{'='*50}\n")
    
    # Save model
    with open('loan_model.pkl', 'wb') as f:
        pickle.dump((model, label_encoders, feature_names), f)
    print("Model saved to loan_model.pkl\n")
    
    return accuracy

def load_model():
    """Load trained model"""
    global model, label_encoders, feature_names
    
    if os.path.exists('loan_model.pkl'):
        with open('loan_model.pkl', 'rb') as f:
            model, label_encoders, feature_names = pickle.load(f)
        print("Model loaded from loan_model.pkl")
    else:
        print("No saved model found. Training new model...")
        train_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Create input dataframe
        input_data = pd.DataFrame([data])
        
        # Encode categorical variables
        for col in label_encoders.keys():
            if col in input_data.columns:
                le = label_encoders[col]
                # Handle unknown categories
                input_data[col] = input_data[col].apply(
                    lambda x: x if x in le.classes_ else le.classes_[0]
                )
                input_data[col] = le.transform(input_data[col].astype(str))
        
        # Ensure all features are present
        for col in feature_names:
            if col not in input_data.columns:
                input_data[col] = 0
        
        # Reorder columns to match training
        input_data = input_data[feature_names]
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        result = {
            'prediction': 'Default Risk' if prediction == 1 else 'Good Credit',
            'probability': {
                'good': float(probability[0]),
                'default': float(probability[1])
            },
            'risk_level': 'High' if probability[1] > 0.7 else 'Medium' if probability[1] > 0.4 else 'Low'
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/retrain', methods=['POST'])
def retrain():
    """Retrain the model"""
    try:
        accuracy = train_model()
        return jsonify({'message': 'Model retrained successfully', 'accuracy': f'{accuracy:.2%}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("\n" + "="*50)
    print("BANK LOAN DEFAULT PREDICTION SYSTEM")
    print("="*50 + "\n")
    load_model()
    print("Starting Flask server...")
    print("Open http://localhost:5000 in your browser\n")
    app.run(debug=True, port=5000)