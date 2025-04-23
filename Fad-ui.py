import streamlit as st
import pandas as pd
import joblib
import numpy as np
import gender_guesser.detector as gender

# ---- Load Models ----
@st.cache_resource
def load_model(model_key):
    return joblib.load(MODEL_FILES[model_key])

MODEL_FILES = {
    "Random Forest": "fake_account_model_RF99.pkl",
    "SVM":           "svm_model.pkl",
    "XGBoost":       "xgb_model.pkl",
    "ANN (MLP)":     "nn_model.pkl"
}

# ---- Load Dataset ----
@st.cache_data
def load_user_dataset():
    return pd.read_csv("users.csv")

user_df = load_user_dataset()

# ---- Gender Detector ----
sex_predictor = gender.Detector(case_sensitive=False)
sex_map = {'female': -2, 'mostly_female': -1, 'unknown': 0, 'mostly_male': 1, 'male': 2}

def predict_sex(name):
    first_name = name.split(" ")[0]
    gender_label = sex_predictor.get_gender(first_name)
    return sex_map.get(gender_label, 0)

lang_dict = {
    'en': 0, 'fr': 1, 'es': 2, 'ja': 3, 'de': 4, 'it': 5, 'ko': 6, 'pt': 7, 
    'ru': 8, 'tr': 9, 'ar': 10, 'pl': 11, 'zh': 12, 'nl': 13, 'fa': 14, 
    'sv': 15, 'no': 16, 'da': 17, 'fi': 18
}

# ---- Streamlit UI ----
st.set_page_config(page_title="Fake Account Detector", layout="centered")
st.title("🕵️‍♀️ Fake Account Detector")
st.image("fakenot.png")
st.markdown("Enter user details manually, or choose a user from the dataset to predict if it's **Fake or Genuine**.")

# ---- Model Selection ----
selected_model = st.selectbox("Select Model", list(MODEL_FILES.keys()))
model = load_model(selected_model)

# ---- User Inputs ----
name = st.text_input("Full Name", "Alice Johnson")
lang = st.selectbox("Language Code", list(lang_dict.keys()))
statuses_count = st.number_input("Statuses Count", min_value=0, value=120)
followers_count = st.number_input("Followers Count", min_value=0, value=250)
friends_count = st.number_input("Friends Count", min_value=0, value=300)
favourites_count = st.number_input("Favourites Count", min_value=0, value=90)
listed_count = st.number_input("Listed Count", min_value=0, value=2)

# Extra fields (only for Random Forest)
verified = False
default_profile_image = False

# ---- Load From Dataset Option ----
st.markdown("---")
st.subheader("📂 Or choose a user from dataset")

row_index = st.number_input("Select Row (Index)", min_value=0, max_value=len(user_df)-1, value=0)

if st.button("Load User From Dataset"):
    selected = user_df.iloc[row_index]
    name = selected.get("name", name)
    lang = selected.get("lang", lang)
    statuses_count = int(selected.get("statuses_count", statuses_count))
    followers_count = int(selected.get("followers_count", followers_count))
    friends_count = int(selected.get("friends_count", friends_count))
    favourites_count = int(selected.get("favourites_count", favourites_count))
    listed_count = int(selected.get("listed_count", listed_count))
    if "verified" in selected:
        verified = bool(selected.get("verified", False))
    if "default_profile_image" in selected:
        default_profile_image = bool(selected.get("default_profile_image", False))

# ---- Prediction ----
if st.button("Predict"):
    sex_code = predict_sex(name)
    lang_code = lang_dict.get(lang, -1)

    if selected_model == "Random Forest":
        features = pd.DataFrame([[
            statuses_count, followers_count, friends_count,
            favourites_count, listed_count, sex_code, lang_code,
            int(verified), int(default_profile_image)
        ]], columns=[
            'statuses_count', 'followers_count', 'friends_count',
            'favourites_count', 'listed_count', 'sex_code', 'lang_code',
            'verified', 'default_profile_image'
        ])
    else:
        features = pd.DataFrame([[
            statuses_count, followers_count, friends_count,
            favourites_count, listed_count, sex_code, lang_code
        ]], columns=[
            'statuses_count', 'followers_count', 'friends_count',
            'favourites_count', 'listed_count', 'sex_code', 'lang_code'
        ])

    prediction = model.predict(features)[0]
    label = "🟢 Genuine" if prediction == 1 else "🔴 Fake"
    
    st.subheader("🔍 Prediction Result:")
    st.markdown(f"**This account is likely: {label}**")
