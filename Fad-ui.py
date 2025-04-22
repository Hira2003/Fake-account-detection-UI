import streamlit as st
import pandas as pd
import joblib
import numpy as np
import gender_guesser.detector as gender

# Load the saved model
model = joblib.load('fake_account_model_RF95.pkl')

# Initialize gender detector
sex_predictor = gender.Detector(case_sensitive=False)

# Language mapping (should match training)
lang_dict = {
    'en': 0, 'fr': 1, 'es': 2, 'ja': 3, 'de': 4, 'it': 5, 'ko': 6, 'pt': 7, 
    'ru': 8, 'tr': 9, 'ar': 10, 'pl': 11, 'zh': 12, 'nl': 13, 'fa': 14, 
    'sv': 15, 'no': 16, 'da': 17, 'fi': 18
}  # Update this based on your actual training data

sex_map = {'female': -2, 'mostly_female': -1, 'unknown': 0, 'mostly_male': 1, 'male': 2}

def predict_sex(name):
    first_name = name.split(" ")[0]
    gender_label = sex_predictor.get_gender(first_name)
    return sex_map.get(gender_label, 0)

st.title("🕵️‍♂️ Fake Account Detector")
st.image("fakenot.png")
st.write("Enter user profile details below to predict whether the account is **Fake or Genuine**.")

# User Inputs
name = st.text_input("Full Name", "Alice Johnson")
lang = st.selectbox("Language Code", list(lang_dict.keys()))
statuses_count = st.number_input("Statuses Count", min_value=0, value=120)
followers_count = st.number_input("Followers Count", min_value=0, value=250)
friends_count = st.number_input("Friends Count", min_value=0, value=300)
favourites_count = st.number_input("Favourites Count", min_value=0, value=90)
listed_count = st.number_input("Listed Count", min_value=0, value=2)

# Predict button
if st.button("Predict"):
    sex_code = predict_sex(name)
    lang_code = lang_dict.get(lang, -1)

    features = pd.DataFrame([[
        statuses_count, followers_count, friends_count,
        favourites_count, listed_count, sex_code, lang_code
    ]], columns=[
        'statuses_count', 'followers_count', 'friends_count',
        'favourites_count', 'listed_count', 'sex_code', 'lang_code'
    ])

    prediction = model.predict(features)[0]
    label = "🟢 Genuine" if prediction == 1 else "🔴 Fake"

    st.subheader("🔍 Prediction:")
    st.markdown(f"**This account is likely: {label}**")



custom_css = """
<style>
/* Banner area */
.app-banner {
    background-image: url('https://raw.githubusercontent.com/<user>/<repo>/main/assets/banner.png');
    background-size: cover;
    height: 160px;
    border-radius: 12px;
    margin-bottom: 1rem;
}

/* Card‑like containers */
.stContainer, .stTextInput, .stNumberInput, .stSelectbox {
    background: #ffffff !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05) !important;
}

/* Make primary button a little thicker */
div.stButton > button:first-child {
    padding: 0.6rem 2rem;
    font-weight: 600;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Optional banner render
st.markdown('<div class="app-banner"></div>', unsafe_allow_html=True)
