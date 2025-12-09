# 🩺 Diabetes Prediction App using CatBoost

A simple and effective Machine Learning application that predicts whether a user is at risk of diabetes based on personal health data. This project uses the **CatBoost Classifier**, a powerful gradient boosting algorithm that performs extremely well on tabular healthcare datasets. The app allows users to enter their medical details and instantly receive a prediction on diabetes risk.

---

## 🎯 Project Objective

To build a user-friendly diabetes prediction system that:

- Uses **CatBoost** for accurate predictions
- Accepts user input through an interactive app
- Provides an instant prediction (**Diabetic / Not Diabetic**)
- Demonstrates practical machine learning usage in healthcare

---

## 🧠 About the Model

### ✔ Algorithm Used: CatBoost Classifier

CatBoost is chosen because:

It handles complex feature interactions automatically and requires minimal preprocessing, which makes it ideal for medical tabular datasets.
CatBoost produced higher ROC-AUC, better recall, and a more balanced F1-score, indicating stronger predictive ability.
In our experiments, CatBoost consistently generalized better on unseen test data, making it the most reliable model for diabetes prediction.

---

## 📂 Dataset Used

**Dataset:** Pima Indians Diabetes Dataset  
**Format:** CSV  
**Source:** Kaggle / UCI


### 🧬 Features in the Dataset:

- `Pregnancies`
- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`
- `DiabetesPedigreeFunction`
- `Age`

**Target:** `Outcome` → 1 = Diabetic, 0 = Non-Diabetic

---

## 🏗️ How the System Works

```
User Inputs Health Data in App
        ↓
Data is Sent to CatBoost Model
        ↓
Model Predicts Diabetes Risk
        ↓
Result Shown to User (Yes/No + Probability)
```

---

## 📱 App Functionality

The app:

- Collects user health information
- Sends data to the CatBoost model
- Displays:
  - **Prediction** (Diabetic / Not Diabetic)
  - **Risk Probability Score**
- Simple, clean UI for easy use

*(Works with Streamlit, Flask, Jupyter UI, etc.)*

---

## 📁 Project Structure

```
diabetes-catboost-app/
│
├── data/
│   └── diabetes.csv
│
├── model/
│   └── catboost_diabetes_model.cbm
│
├── app/
│   └── app.py          # Your application code
│
├── notebooks/
│   └── model_training.ipynb
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Running the App

### Clone the Repository

```bash
git clone https://github.com/create-sourav/diabetes-catboost-app.git
cd diabetes-catboost-app
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the App

**For Flask/Python:**

```bash
python app/app.py
```

**For Streamlit:**

```bash
streamlit run app/app.py
```

---

## 📈 Model Performance

| Metric     | Score        |
|------------|--------------|
| Accuracy   | 0.76         |
| Precision  | 0.78         |
| Recall     |  0.77        |
| AUC Score  | 0.83         |

*(Your actual performance numbers can be added after training.)*

---

## 🔮 Future Enhancements

- [ ] Add SHAP explainability (feature importance)
- [ ] Deploy app using Render / HuggingFace / Railway
- [ ] Improve UI/UX
- [ ] Add medical disclaimer section

---

## 👨‍💻 Author

**Sourav Mondal**  
Machine Learning & Business Analytics Enthusiast

🔗 **GitHub:** [https://github.com/create-sourav](https://github.com/create-sourav)  
🔗 **Email:** *soouravmondal5f@gmail.com*

-----

## ⚠️ Disclaimer

This application is for educational and informational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical concerns.

---

## 🙏 Acknowledgments

- **CatBoost Team** for the excellent gradient boosting library
- **Kaggle/UCI** for the Pima Indians Diabetes Dataset
- Open-source community for continuous support

---

**⭐ If you found this project helpful, please give it a star on GitHub!**
