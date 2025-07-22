\# 🧠 Employee Salary Prediction

This project predicts whether an individual's annual income exceeds $50K based on demographic and employment-related attributes. It uses machine learning models trained on the UCI Adult dataset and provides a user-friendly prediction interface built with Streamlit.

## 📌 Problem Statement

The goal is to build a predictive model that can classify whether an individual's income is **<=50K or >50K** using various features such as age, education, work class, occupation, and more. This helps in making informed decisions in HR and recruitment processes.

---

## 🛠️ System Development Approach

### ✅ System Requirements

- **Python:** 3.8+
- **IDE:** Jupyter Notebook / VS Code
- **Platform:** Windows, Linux, or macOS
- **Deployment:** Hugging Face Spaces (Streamlit SDK)

### ✅ Libraries Used

| Library         | Purpose                                      |
|----------------|----------------------------------------------|
| pandas          | Data manipulation and preprocessing         |
| numpy           | Numerical computations                      |
| scikit-learn    | ML models, preprocessing, evaluation        |
| streamlit       | UI for user interaction and predictions     |
| joblib / pickle | Save and load trained models and encoders   |

---

## 📂 Project Structure

```bash
employee-salary-prediction/
├── app.py                      # Streamlit web app
├── predictor_employee_salary.ipynb  # Jupyter Notebook with full ML workflow
├── best_model.pkl             # Trained ML model
├── pipeline.pkl               # Preprocessing pipeline
├── encoders.pkl               # Label encoders for categorical features
├── requirements.txt           # Required Python libraries
├── README.md                  # Project documentation

**How to Run the Project**
Clone the repository or upload files to Hugging Face Spaces.

Install requirements locally:
               pip install -r requirements.txt
Run the app locally:
                    streamlit run app.py
**Model Training Summary**
Dataset: UCI Adult Income Dataset

Preprocessing:

Categorical encoding using LabelEncoder

Scaling using MinMaxScaler

Algorithms evaluated:

Logistic Regression

Random Forest Classifier ✅ (Best Accuracy)

K-Nearest Neighbors

Final model: Random Forest Classifier

Accuracy: ~85% on test set
**Streamlit App**
User inputs demographic data

Behind the scenes, the input is transformed using the saved preprocessing pipeline

Model predicts income category

Output displayed as >50K or <=50K

**Future Scope**

Add model explainability (e.g., SHAP, LIME)

REST API integration

Deploy with Docker for scalability

Add more detailed educational and geographic features


