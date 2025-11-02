from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from xgboost import XGBClassifier
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables
model = None
label_encoders = {}
feature_names = []
model_type = "Random Forest"  # Default model
model_comparison_results = None

def load_and_preprocess_data():
    """Load and preprocess the German Credit Data from local file"""
    columns = [
        'existing_account', 'duration', 'credit_history', 'purpose', 'credit_amount',
        'savings', 'employment', 'installment_rate', 'personal_status', 'debtors',
        'residence', 'property', 'age', 'other_plans', 'housing',
        'existing_credits', 'job', 'dependents', 'telephone', 'foreign_worker', 'class'
    ]
    
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

def compare_models():
    """Compare multiple ML models and return results"""
    global model_comparison_results
    
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data()

    X = df.drop('class', axis=1)
    y = df['class']

    temp_label_encoders = {}
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        temp_label_encoders[col] = le

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced'),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM (RBF)": SVC(kernel='rbf', probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB()
    }

    results = []

    print("\nTraining and evaluating models...\n")

    for name, model_obj in models.items():
        model_obj.fit(X_train, y_train)
        y_pred = model_obj.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')

        results.append({
            'Model': name,
            'Accuracy': f"{acc:.2%}",
            'F1 Score': f"{f1:.2%}",
            'Accuracy_Raw': acc,
            'F1_Raw': f1
        })

        print(f" {name}")
        print(f"   Accuracy: {acc:.2%}")
        print(f"   F1 Score: {f1:.2%}\n")

    results_df = pd.DataFrame(results).sort_values('Accuracy_Raw', ascending=False).reset_index(drop=True)
    model_comparison_results = results_df
    
    print("==================================================")
    print("Model Comparison Results")
    print(results_df[['Model', 'Accuracy', 'F1 Score']])
    print("==================================================")

    best_model_name = results_df.iloc[0]['Model']
    print(f"\nBest model based on accuracy: {best_model_name}")

    return results_df[['Model', 'Accuracy', 'F1 Score']].to_dict('records')

def train_random_forest_model():
    """Train Random Forest model"""
    global model, label_encoders, feature_names, model_type
    
    model_type = "Random Forest"
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data()
    
    X = df.drop('class', axis=1)
    y = df['class']
    
    label_encoders = {}
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    feature_names = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
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
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print(f"Random Forest Model trained successfully!")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Good Credit', 'Default Risk']))
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"{'='*50}\n")
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 Most Important Features:")
    print(feature_importance.head(10))
    print(f"{'='*50}\n")
    
    return accuracy

def train_random_forest_with_tuning():
    """Train Random Forest with hyperparameter tuning"""
    global model, label_encoders, feature_names, model_type
    
    model_type = "Random Forest (Tuned)"
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data()

    X = df.drop('class', axis=1)
    y = df['class']

    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nStarting hyperparameter tuning with RandomizedSearchCV...")
    param_dist = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [8, 10, 12, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        'max_features': ['sqrt', 'log2', None]
    }

    rf = RandomForestClassifier(
        class_weight='balanced', random_state=42
    )

    search = RandomizedSearchCV(
        rf, param_distributions=param_dist,
        n_iter=20, cv=3, scoring='f1_macro',
        n_jobs=-1, random_state=42, verbose=1
    )

    search.fit(X_train, y_train)
    print(f"\nBest Params Found: {search.best_params_}")

    model = search.best_estimator_

    print("\nRunning 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"Cross-validation mean accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")

    print("\nTraining final model with best parameters...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"Random Forest (Tuned) Model trained successfully!")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Good Credit', 'Default Risk']))
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"{'='*50}\n")

    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("Top 10 Most Important Features:")
    print(feature_importance.head(10))
    print(f"{'='*50}\n")

    return accuracy

def train_xgboost_model():
    """Train XGBoost model"""
    global model, label_encoders, feature_names, model_type
    
    model_type = "XGBoost"
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data()
    X = df.drop('class', axis=1)
    y = df['class']

    label_encoders = {}
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training XGBoost model...")
    model = XGBClassifier(
        eval_metric='logloss',
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"XGBoost Model trained successfully!")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Good Credit', 'Default Risk']))
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"{'='*50}\n")

    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("Top 10 Most Important Features:")
    print(feature_importance.head(10))
    print(f"{'='*50}\n")

    return accuracy

def train_xgboost_ensemble():
    """Train XGBoost Ensemble with hyperparameter tuning"""
    global model, label_encoders, feature_names, model_type
    
    model_type = "XGBoost Ensemble (Tuned)"
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data()

    X = df.drop('class', axis=1)
    y = df['class']

    label_encoders = {}
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("🔍 Hyperparameter tuning for XGBoost...")

    xgb = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)

    param_dist = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [3, 4, 5, 6, 7],
        'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5]
    }

    rand_search = RandomizedSearchCV(
        xgb,
        param_distributions=param_dist,
        n_iter=20,
        scoring='accuracy',
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    rand_search.fit(X_train, y_train)
    best_params = rand_search.best_params_
    print(f"Best Params Found: {best_params}")

    print("Training ensemble model with tuned XGBoost...")

    xgb_best = XGBClassifier(**best_params, eval_metric='logloss', use_label_encoder=False, random_state=42)
    rf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=10,
                                min_samples_leaf=5, class_weight='balanced', random_state=42)
    gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)

    model = StackingClassifier(
        estimators=[('xgb', xgb_best), ('rf', rf), ('gb', gb)],
        final_estimator=LogisticRegression(),
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"XGBoost Ensemble Model trained successfully!")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Good Credit', 'Default Risk']))
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"{'='*50}\n")

    return accuracy

def load_model():
    """Load trained model"""
    global model, label_encoders, feature_names, model_type
    
    if os.path.exists('loan_model.pkl'):
        try:
            with open('loan_model.pkl', 'rb') as f:
                loaded_data = pickle.load(f)
                
            # Handle both old (3 values) and new (4 values) format
            if len(loaded_data) == 3:
                model, label_encoders, feature_names = loaded_data
                model_type = "Random Forest"  # Default for old models
                print(f"Model loaded from loan_model.pkl (Legacy format)")
                # Re-save in new format
                save_model()
            else:
                model, label_encoders, feature_names, model_type = loaded_data
                print(f"Model loaded from loan_model.pkl ({model_type})")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Training new XGBoost model...")
            train_xgboost_model()
            save_model()
    else:
        print("No saved model found. Training XGBoost model (best performing)...")
        train_xgboost_model()
        save_model()

def save_model():
    """Save current model"""
    with open('loan_model.pkl', 'wb') as f:
        pickle.dump((model, label_encoders, feature_names, model_type), f)
    print(f"Model saved to loan_model.pkl ({model_type})")

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/predict')
def predict_page():
    return render_template('index.html')

@app.route('/admin')
def admin():
    return render_template('admin.html')

@app.route('/api/predict', methods=['POST'])
def predict_api():
    try:
        data = request.json
        
        input_data = pd.DataFrame([data])
        
        for col in label_encoders.keys():
            if col in input_data.columns:
                le = label_encoders[col]
                input_data[col] = input_data[col].apply(
                    lambda x: x if x in le.classes_ else le.classes_[0]
                )
                input_data[col] = le.transform(input_data[col].astype(str))
        
        for col in feature_names:
            if col not in input_data.columns:
                input_data[col] = 0
        
        input_data = input_data[feature_names]
        
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        result = {
            'prediction': 'Default Risk' if prediction == 1 else 'Good Credit',
            'probability': {
                'good': float(probability[0]),
                'default': float(probability[1])
            },
            'risk_level': 'High' if probability[1] > 0.7 else 'Medium' if probability[1] > 0.4 else 'Low',
            'model_used': model_type
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/retrain', methods=['POST'])
def retrain():
    """Retrain the model"""
    try:
        data = request.json
        model_choice = data.get('model', 'xgboost')
        
        if model_choice == 'random_forest':
            accuracy = train_random_forest_model()
        elif model_choice == 'random_forest_tuned':
            accuracy = train_random_forest_with_tuning()
        elif model_choice == 'xgboost':
            accuracy = train_xgboost_model()
        elif model_choice == 'xgboost_ensemble':
            accuracy = train_xgboost_ensemble()
        else:
            return jsonify({'error': 'Invalid model choice'}), 400
        
        save_model()
        return jsonify({
            'message': f'{model_type} retrained successfully',
            'accuracy': f'{accuracy:.2%}',
            'model_type': model_type
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/compare', methods=['POST'])
def compare():
    """Compare all models"""
    try:
        results = compare_models()
        return jsonify({
            'message': 'Model comparison completed',
            'results': results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get current model information"""
    try:
        return jsonify({
            'model_type': model_type,
            'features_count': len(feature_names),
            'is_loaded': model is not None
        })
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