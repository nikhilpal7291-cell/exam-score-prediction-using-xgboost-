import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the tuned XGBoost model
with open('xgb_tuned_model.pkl', 'rb') as f:
    xgb_tuned = pickle.load(f)



# Define the mappings for the categorical features used in the model (X)
categorical_mappings = {
    'sleep_quality': {'average': 0, 'good': 1, 'poor': 2},
    'study_method': {'coaching': 0, 'no study': 1, 'group study': 2, 'online videos': 3, 'self-study': 4},
    'facility_rating': {'high': 0, 'low': 1, 'medium': 2}
}

# Streamlit App Title
st.title('Exam Score Prediction Model')
st.write('Enter the student details to predict their exam score.')

# Input features
st.sidebar.header('Student Features')

def user_input_features():
    student_id = st.sidebar.number_input('Student ID', min_value=1, value=1)
    study_hours = st.sidebar.slider('Study Hours', min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    class_attendance = st.sidebar.slider('Class Attendance (%)', min_value=0.0, max_value=100.0, value=75.0, step=0.1)
    sleep_hours = st.sidebar.slider('Sleep Hours', min_value=0.0, max_value=12.0, value=7.0, step=0.1)
    
    # Categorical features with their mappings
    sleep_quality_str = st.sidebar.selectbox('Sleep Quality', options=list(categorical_mappings['sleep_quality'].keys()))
    study_method_str = st.sidebar.selectbox('Study Method', options=list(categorical_mappings['study_method'].keys()))
    facility_rating_str = st.sidebar.selectbox('Facility Rating', options=list(categorical_mappings['facility_rating'].keys()))
    
    # Encode categorical features
    sleep_quality = categorical_mappings['sleep_quality'][sleep_quality_str]
    study_method = categorical_mappings['study_method'][study_method_str]
    facility_rating = categorical_mappings['facility_rating'][facility_rating_str]

    data = {
        'student_id': student_id,
        'study_hours': study_hours,
        'class_attendance': class_attendance,
        'sleep_hours': sleep_hours,
        'sleep_quality': sleep_quality,
        'study_method': study_method,
        'facility_rating': facility_rating
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('User Input Features')
st.write(input_df)

# Predict button
if st.button('Predict Exam Score'):
    prediction = xgb_tuned.predict(input_df)
    st.subheader('Predicted Exam Score')
    st.write(f'The predicted exam score is: **{prediction[0]:.2f}**')

