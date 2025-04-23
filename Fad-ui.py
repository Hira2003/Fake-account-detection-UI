import streamlit as st
import pandas as pd
import joblib
import numpy as np
import gender_guesser.detector as gender

# ---------- CONFIG -----------------------------------------------------------
MODEL_FILES = {
    "Random Forest": "fake_account_model_RF99.pkl",
    "SVM":           "svm_model.pkl",
    "XGBoost":       "xgb_model.pkl",
    "ANN (MLP)":     "nn_model.pkl"
}

@st.cache_resource(show_spinner=False)
def load_model(model_key: str):
    """Load a model only once per session."""
    return joblib.load(MODEL_FILES[model_key])

sex_predictor = gender.Detector(case_sensitive=False)

lang_dict = {
    'en': 0, 'fr': 1, 'es': 2, 'ja': 3, 'de': 4, 'it': 5, 'ko': 6, 'pt': 7, 
    'ru': 8, 'tr': 9, 'ar': 10, 'pl': 11, 'zh': 12, 'nl': 13, 'fa': 14, 
    'sv': 15, 'no': 16, 'da': 17, 'fi': 18
}
sex_map = {'female': -2, 'mostly_female': -1, 'unknown': 0,
           'mostly_male': 1, 'male': 2}

def predict_sex(name: str) -> int:
    first_name = name.split(" ")[0]
    return sex_map.get(sex_predictor.get_gender(first_name), 0)

# ---------- UI ---------------------------------------------------------------
st.title("🕵️‍♂️ Fake Account Detector")
st.image("fakenot.png")
st.sidebar.header("🔧 Settings")
model_choice = st.sidebar.selectbox("Choose model", list(MODEL_FILES.keys()),
                                    index=0)
model = load_model(model_choice)

st.write("Enter profile details to predict whether the account is "
         "**Fake or Genuine** using **{}**.".format(model_choice))

# --- Inputs ------------------------------------------------------------------
name              = st.text_input("Full Name", "Alice Johnson")
lang              = st.selectbox("Language Code", list(lang_dict.keys()))
statuses_count    = st.number_input("Statuses Count",  min_value=0, value=120)
followers_count   = st.number_input("Followers Count", min_value=0, value=250)
friends_count     = st.number_input("Friends Count",   min_value=0, value=300)
favourites_count  = st.number_input("Favourites Count",min_value=0, value=90)
listed_count      = st.number_input("Listed Count",    min_value=0, value=2)

# Add extra features only for Random Forest model
if model_choice == "Random Forest":
    verified = st.checkbox("Is Verified", value=False)  # New input
    default_profile_image = st.checkbox("Has Default Profile Image", value=False)  # New input

    # Update the feature set to include the extra features
    features = pd.DataFrame([[
        statuses_count, followers_count, friends_count,
        favourites_count, listed_count,
        predict_sex(name),
        lang_dict.get(lang, -1),
        verified,  # Only for Random Forest
        default_profile_image  # Only for Random Forest
    ]], columns=[
        'statuses_count', 'followers_count', 'friends_count',
        'favourites_count', 'listed_count', 'sex_code', 'lang_code',
        'verified', 'default_profile_image'
    ])
else:
    features = pd.DataFrame([[
        statuses_count, followers_count, friends_count,
        favourites_count, listed_count,
        predict_sex(name),
        lang_dict.get(lang, -1)
    ]], columns=[
        'statuses_count', 'followers_count', 'friends_count',
        'favourites_count', 'listed_count', 'sex_code', 'lang_code'
    ])

# Prediction button
if st.button("Predict"):
    pred = int(model.predict(features)[0])
    label = "🟢 Genuine" if pred == 1 else "🔴 Fake"

    st.subheader("🔍 Prediction:")
    st.markdown(f"**This account is likely: {label}**")
