# 🩺 Medical Premium Price Predictor  

## 📘 Overview  
The **Medical Premium Price Predictor** is a machine learning web app that predicts a user's expected **medical insurance premium** based on various health and lifestyle factors such as **age**, **height**, **weight**, **chronic diseases**, and more.

It uses a **Random Forest Classifier** trained on a healthcare dataset and is deployed using **Flask** with a clean modern UI.

---

## 📂 Project Structure  

```
MEDICALML/
│
├── templates/
│   └── index.html               # Stylish frontend for user input
│
├── app.py                       # Flask backend for handling requests
├── premium_insurance.py         # Model training and saving script
├── Medicalpremium.csv           # Dataset
├── Premium_Insurance.ipynb      # Data exploration notebook
├── premium_model.pkl            # Trained Random Forest model
├── scaler.pkl                   # Scaler for preprocessing inputs
└── requirements.txt             # Project dependencies
```

---

## ⚙️ Installation & Setup  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/habibsheikhh/MedicalML.git
cd MedicalML
```

### 2️⃣ Create Virtual Environment  
```bash
python -m venv venv
```

Activate it:  
- **Windows:** `venv\Scripts\activate`  
- **Mac/Linux:** `source venv/bin/activate`

### 3️⃣ Install Dependencies  
```bash
pip install -r requirements.txt
```

### 4️⃣ Train the Model  
This script will train the Random Forest model and save `premium_model.pkl` and `scaler.pkl`.
```bash
python premium_insurance.py
```

### 5️⃣ Run the Flask App  
```bash
python app.py
```

Then open your browser and go to:  
👉 http://127.0.0.1:5000/

---

## 💡 Features  
✅ Predicts medical premium instantly  
✅ Calculates BMI automatically  
✅ Handles categorical and numeric inputs  
✅ Uses Random Forest Classifier for high accuracy  
✅ Clean, responsive black & white modern UI  
✅ Ready for deployment on Render, Railway, or Heroku  

---

## 🧠 Model Workflow  

1. Load and clean dataset (`Medicalpremium.csv`)  
2. Feature engineering — add **BMI** and **BMI Status**  
3. Encode categorical columns  
4. Train Random Forest Classifier  
5. Evaluate accuracy  
6. Save model (`premium_model.pkl`) and scaler (`scaler.pkl`)  

---

## 🧾 Example Input  

| Field | Example |
|-------|----------|
| Age | 35 |
| Diabetes | 1 |
| BloodPressureProblems | 0 |
| AnyTransplants | 0 |
| AnyChronicDiseases | 1 |
| Height (cm) | 172 |
| Weight (kg) | 75 |
| KnownAllergies | 0 |
| HistoryOfCancerInFamily | 1 |
| NumberOfMajorSurgeries | 2 |

**Predicted Output:**  
💰 *Estimated Premium: ₹18,450*

---

## 🧰 Requirements  

`requirements.txt` contains:  
```
numpy
pandas
matplotlib
seaborn
scikit-learn
flask
```

Install using:
```bash
pip install -r requirements.txt
```

---

## 📊 Model Accuracy  
The trained Random Forest Classifier achieved around:  
```
✅ Accuracy: 85–90%
```

---

## 🖥️ UI Preview  

- Clean black & white modern layout  
- Glass-effect card with soft shadows  
- Stylish rounded input boxes  
- Gradient predict button with hover animation  
- Responsive design for all screen sizes  

---


## 👨‍💻 Author  

**Mohammad Habib , Mohammad Ujer, Snehanand**  

---
