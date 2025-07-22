import streamlit as st
import pandas as pd
import joblib

# Load the trained pipeline (scaler + model)
model = joblib.load("pipeline.pkl")

# LabelEncoder mappings (must match training encodings)
workclass_map = {'Private': 4, 'Self-emp-not-inc': 5, 'Self-emp-inc': 3, 'Federal-gov': 1,
                 'Local-gov': 2, 'State-gov': 6, 'Notlisted': 0}
marital_map = {'Never-married': 1, 'Married-civ-spouse': 2, 'Divorced': 0,
               'Separated': 3, 'Widowed': 4, 'Married-spouse-absent': 5}
occupation_map = {'Tech-support': 12, 'Craft-repair': 0, 'Other-service': 9, 'Sales': 11,
                  'Exec-managerial': 4, 'Prof-specialty': 10, 'Handlers-cleaners': 5,
                  'Machine-op-inspct': 6, 'Adm-clerical': 1, 'Farming-fishing': 2,
                  'Transport-moving': 13, 'Priv-house-serv': 8, 'Protective-serv': 7,
                  'Armed-Forces': 3, 'Others': 14}
relationship_map = {'Wife': 5, 'Own-child': 1, 'Husband': 2,
                    'Not-in-family': 3, 'Other-relative': 4, 'Unmarried': 0}
race_map = {'White': 4, 'Black': 0, 'Asian-Pac-Islander': 1,
            'Amer-Indian-Eskimo': 2, 'Other': 3}
gender_map = {'Male': 1, 'Female': 0}
native_country_map = {'United-States': 38, 'India': 15, 'Mexico': 26,
                      'Philippines': 30, 'Germany': 10, 'Canada': 4, 'Others': 0}

# App UI
st.set_page_config(page_title="Employee Salary Predictor", page_icon="💼", layout="centered")
st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

st.sidebar.header("Input Employee Details")

# Collecting user inputs
age = st.sidebar.slider("Age", 17, 75, 30)
workclass = st.sidebar.selectbox("Workclass", list(workclass_map.keys()))
fnlwgt = st.sidebar.number_input("Fnlwgt", 10000, 1000000, 150000)
marital_status = st.sidebar.selectbox("Marital Status", list(marital_map.keys()))
occupation = st.sidebar.selectbox("Occupation", list(occupation_map.keys()))
relationship = st.sidebar.selectbox("Relationship", list(relationship_map.keys()))
race = st.sidebar.selectbox("Race", list(race_map.keys()))
gender = st.sidebar.radio("Gender", list(gender_map.keys()))
capital_gain = st.sidebar.number_input("Capital Gain", 0, 100000, 0)
capital_loss = st.sidebar.number_input("Capital Loss", 0, 10000, 0)
hours_per_week = st.sidebar.slider("Hours per Week", 1, 100, 40)
native_country = st.sidebar.selectbox("Native Country", list(native_country_map.keys()))
educational_num = st.sidebar.slider("Education Level (numeric)", 1, 16, 10)

# Display-friendly input dataframe (before encoding)
display_df = pd.DataFrame([{
    'age': age,
    'workclass': workclass,
    'fnlwgt': fnlwgt,
    'marital-status': marital_status,
    'occupation': occupation,
    'relationship': relationship,
    'race': race,
    'gender': gender,
    'capital-gain': capital_gain,
    'capital-loss': capital_loss,
    'hours-per-week': hours_per_week,
    'native-country': native_country,
    'educational-num': educational_num
}])

# Encoded input for prediction
input_df = pd.DataFrame([{
    'age': age,
    'workclass': workclass_map[workclass],
    'fnlwgt': fnlwgt,
    'marital-status': marital_map[marital_status],
    'occupation': occupation_map[occupation],
    'relationship': relationship_map[relationship],
    'race': race_map[race],
    'gender': gender_map[gender],
    'capital-gain': capital_gain,
    'capital-loss': capital_loss,
    'hours-per-week': hours_per_week,
    'native-country': native_country_map[native_country],
    'educational-num': educational_num
}])

# Display inputs
st.write("### 🔎 Input Data")
st.dataframe(display_df)
st.write("Input shape:", input_df.shape)
st.write("Input columns:", input_df.columns.tolist())

# Predict button
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.success("✅ Prediction: This employee likely earns >50K")
    else:
        st.info("ℹ️ Prediction: This employee likely earns ≤50K")
