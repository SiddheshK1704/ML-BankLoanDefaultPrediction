# 🏦 Bank Loan Default Prediction  
### Powered by Random Forest Machine Learning  

This project is a **Flask-based web application** that predicts the likelihood of a customer defaulting on a loan using **Machine Learning (Random Forest Classifier)** trained on the **German Credit Dataset** from the **UCI Machine Learning Repository**.  

It provides an interactive interface where users can input loan applicant details and receive an instant prediction of **Default Risk**, **Default Probability**, and **Risk Level (Low/Medium/High)**.

---

## 🚀 Features  
- 🌐 **Web-based interface** built with Flask and TailwindCSS  
- 🧠 **Random Forest Classifier** trained on real credit data  
- 🧾 **Instant prediction** with detailed risk probability  
- 📊 **Model retraining endpoint** for updating predictions with new data  
- 💾 **Model persistence** using Pickle (`loan_model.pkl`)  

---

## 🧩 Tech Stack  

| Component | Technology |
|------------|-------------|
| **Frontend** | HTML, TailwindCSS, JavaScript |
| **Backend** | Flask (Python) |
| **Machine Learning** | scikit-learn, pandas, numpy |
| **Model** | RandomForestClassifier |
| **Dataset** | German Credit Data (`german.data`) |

---

## 📂 Project Structure  

```
📁 bank-loan-default-prediction/
│
├── app.py                  # Main Flask application
├── loan_model.pkl          # Trained ML model (auto-generated)
├── german.data             # Dataset file (UCI German Credit Data)
├── templates/
│   └── index.html          # Frontend HTML file
├── static/                 # (Optional) For CSS/JS if needed
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## ⚙️ Installation and Setup  

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/yourusername/bank-loan-default-prediction.git
cd bank-loan-default-prediction
```

### 2️⃣ Create and activate a virtual environment  
```bash
python -m venv venv
venv\Scripts\activate     # On Windows
# OR
source venv/bin/activate  # On macOS/Linux
```

### 3️⃣ Install dependencies  
Create a `requirements.txt` (if not already present):  
```text
Flask
pandas
numpy
scikit-learn
```

Then install using:  
```bash
pip install -r requirements.txt
```

### 4️⃣ Download the dataset  
Download **German Credit Data** from the [UCI Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/)  
and place it in your project directory as `german.data`.

---

## 🧠 Running the App  

### Train the Model (Automatically on first run)
```bash
python app.py
```

You’ll see logs showing:
- Dataset loading  
- Model training & accuracy  
- Top 10 important features  

Then open your browser at:  
👉 **http://localhost:5000**

---

## 💻 Web Interface  

The user fills out applicant details such as:
- Existing Account Type  
- Credit Amount  
- Employment Status  
- Purpose of Loan  
- Age, Dependents, Property, etc.

When the user clicks **"Predict Default Risk"**, the model returns:  
- **Prediction:** Good Credit / Default Risk  
- **Default Probability:** e.g., 75.6%  
- **Risk Level:** Low / Medium / High  

---

## 🔁 Retraining the Model  

You can retrain the model anytime using:
```bash
curl -X POST http://localhost:5000/retrain
```
This retrains the Random Forest model and updates `loan_model.pkl`.

---

## 🧾 Example Prediction Output  

| Field | Example |
|--------|----------|
| Prediction | Default Risk |
| Default Probability | 75.6% |
| Risk Level | High |

---

## 🧠 Model Info  

- **Algorithm:** RandomForestClassifier  
- **n_estimators:** 100  
- **max_depth:** 10  
- **min_samples_split:** 10  
- **min_samples_leaf:** 5  
- **class_weight:** balanced  

The model learns from historical credit data to classify loan applicants as:
- `0 → Good Credit`  
- `1 → Default Risk`

---

## 📊 Example Screenshot  

![App Screenshot](0f4fb198-4a12-4778-85fd-41720810cbc5.png)

---

## 👨‍💻 Author  

**Siddhesh Khankhoje**  
💡 Passionate about AI/ML and Web Development  

---

## 📜 License  
This project is open-source and available under the **MIT License**.
