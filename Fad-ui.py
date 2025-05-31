import streamlit as st
import pandas as pd
import joblib
import numpy as np
import gender_guesser.detector as gender
import matplotlib.pyplot as plt

# ---- Load Models ----
@st.cache_resource
def load_model(model_key):
    return joblib.load(MODEL_FILES[model_key])

# Separate model files for Twitter and Instagram
MODEL_FILES = {
    # Twitter Models
    "Random Forest": "fake_account_model_new99.pkl",
    "SVM": "svm_model.pkl",
    "XGBoost": "xgb_model.pkl",
    "ANN (MLP)": "nn_model.pkl",
    
    # Instagram Models
    "IG Random Forest": "fake_account_model_new_insta.pkl",
    "IG SVM":  "svm_model_insta.pkl",
    "IG XGBoost": "xgb_model-clone99_insta.pkl",
    "IG ANN (MLP)": "fake_account_nn_model_insta.pkl"
}

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

# ---- Sidebar Navigation ----
page = st.sidebar.radio("Select Page", ["Main","Twitter Account Detection", "Instagram Account Detection", "About the Study"])

if page == "Main":
    st.title("Fake Account Detector 🕵🏻‍♀️")
    st.image("fakenot.png")
    st.text("Made by: Bouziza Hadjer and Abbassi Khawla")
# ---- Twitter Detection Page ----
if page == "Twitter Account Detection":
    st.title("🐦 Twitter Fake Account Detector")

    selected_model = st.selectbox("Select Model", ["Random Forest", "SVM", "XGBoost", "ANN (MLP)"])
    model = load_model(selected_model)

    st.subheader("✍️ Manual Entry")

    name = st.text_input("Full Name", "Alice Johnson")
    lang = st.selectbox("Language Code", list(lang_dict.keys()))
    statuses_count = st.number_input("Statuses Count", min_value=0, value=120)
    followers_count = st.number_input("Followers Count", min_value=0, value=250)
    friends_count = st.number_input("Friends Count", min_value=0, value=300)
    favourites_count = st.number_input("Favourites Count", min_value=0, value=90)
    listed_count = st.number_input("Listed Count", min_value=0, value=2)

    st.subheader("📂 Load Dataset")
    uploaded_file = st.file_uploader("Upload Twitter CSV", type=["csv"])

    if uploaded_file:
        user_df = pd.read_csv(uploaded_file)
        st.success(f"Loaded dataset with {len(user_df)} rows.")
        row_index = st.number_input("Select Row Index", min_value=0, max_value=len(user_df)-1, value=0)

        if st.button("Load Row"):
            selected = user_df.iloc[row_index]
            name = selected.get("name", name)
            lang = selected.get("lang", lang)
            statuses_count = int(selected.get("statuses_count", statuses_count))
            followers_count = int(selected.get("followers_count", followers_count))
            friends_count = int(selected.get("friends_count", friends_count))
            favourites_count = int(selected.get("favourites_count", favourites_count))
            listed_count = int(selected.get("listed_count", listed_count))

    if st.button("Predict Twitter Account"):
        sex_code = predict_sex(name)
        lang_code = lang_dict.get(lang, -1)

        features = pd.DataFrame([[
            statuses_count, followers_count, friends_count,
            favourites_count, listed_count, sex_code, lang_code
        ]], columns=[
            'statuses_count', 'followers_count', 'friends_count',
            'favourites_count', 'listed_count', 'sex_code', 'lang_code'])

        prediction = model.predict(features)[0]
        label = "🟢 Genuine" if prediction == 1 else "🔴 Fake"
        st.success(f"This account is likely: {label}")

# ---- Instagram Detection Page ----
elif page == "Instagram Account Detection":
    st.title("📸 Instagram Fake Account Detector")

    st.markdown("Fill in the fields below or load from dataset")

    selected_ig_model = st.selectbox("Select Instagram Model", [
        "IG Random Forest", "IG SVM", "IG XGBoost", "IG ANN (MLP)"
    ])
    insta_model = load_model(selected_ig_model)

    num_followers = st.number_input("Followers", min_value=0, value=1000)
    num_following = st.number_input("Following", min_value=0, value=500)
    num_posts = st.number_input("Posts", min_value=0, value=50)
    extern_url = st.checkbox("External URL Present", True)
    len_desc = st.number_input("Description Length", min_value=0, value=150)
    profile_pic = st.checkbox("Profile Picture Present", True)
    Private = st.checkbox("Private Account", False)
    sim_name_username = st.number_input("Similarity Name/Username", min_value=0.0, max_value=1.0, value=0.6)
    len_fullname = st.number_input("Full Name Length", min_value=0, value=10)
    ratio_numlen_username = st.number_input("Number Length / Username Length", min_value=0.0, value=0.3)
    ratio_numlen_fullname = st.number_input("Number Length / Fullname Length", min_value=0.0, value=0.2)

    st.subheader("📂 Load Instagram CSV")
    insta_file = st.file_uploader("Upload Instagram CSV", type=["csv"])

    if insta_file:
        insta_df = pd.read_csv(insta_file)
        st.success(f"Loaded {len(insta_df)} Instagram entries.")
        insta_idx = st.number_input("Select Row Index", min_value=0, max_value=len(insta_df)-1, value=0)

        if st.button("Load Instagram Row"):
            row = insta_df.iloc[insta_idx]
            num_followers = row.get("num_followers", num_followers)
            num_following = row.get("num_following", num_following)
            num_posts = row.get("num_posts", num_posts)
            extern_url = bool(row.get("extern_url", extern_url))
            len_desc = row.get("len_desc", len_desc)
            profile_pic = bool(row.get("profile_pic", profile_pic))
            Private = bool(row.get("Private", Private))
            sim_name_username = row.get("sim_name_username", sim_name_username)
            len_fullname = row.get("len_fullname", len_fullname)
            ratio_numlen_username = row.get("ratio_numlen_username", ratio_numlen_username)
            ratio_numlen_fullname = row.get("ratio_numlen_fullname", ratio_numlen_fullname)

    if st.button("Predict Instagram Account"):
        features = pd.DataFrame([[
            num_followers, num_following, num_posts, int(extern_url), len_desc,
            int(profile_pic), int(Private), sim_name_username, len_fullname,
            ratio_numlen_username, ratio_numlen_fullname
        ]], columns=[
            'num_followers', 'num_following', 'num_posts', 'extern_url', 'len_desc',
            'profile_pic', 'Private', 'sim_name_username', 'len_fullname',
            'ratio_numlen_username', 'ratio_numlen_fullname'])

        prediction = insta_model.predict(features)[0]
        label = "🟢 Genuine" if prediction == 1 else "🔴 Fake"
        st.success(f"This account is likely: {label}")

# ---- About the Study Page ----
elif page == "About the Study":
   st.title("📖 About the Study")

   if st.button("Show Datasets"):
        st.info("Datasets used in this study include Twitter profile metadata and Instagram public features extracted for classification.")
       # File uploader
       uploaded_file = "users.csv"
       uploaded_file1 = "fusers.csv"
       uploaded_file2 = "social_media_merged_numeric.csv"



       # If a file is uploaded
       if uploaded_file is not None:
       # Read CSV into a DataFrame
          df = pd.read_csv(uploaded_file)
          df1 = pd.read_csv(uploaded_file1)
          df2 = pd.read_csv(uploaded_file2)

    
          # Display the DataFrame
          st.subheader("Datasets")
          st.dataframe(df)
          st.dataframe(df1)
          st.dataframe(df2)

   if st.button("Show Figures"):
    import matplotlib.pyplot as plt

    # --- Overall Accuracy Before vs After Optimization ---
    models = ['SVM', 'ANN', 'Random Forest', 'XGBoost']
    before = [91.00, 93.20, 95.20, 92.10]
    after = [91.84, 95.27, 99.00, 97.75]

    x = range(len(models))
    bar_width = 0.35

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(x, before, width=bar_width, label='Before Optimization', color='salmon')
    ax1.bar([i + bar_width for i in x], after, width=bar_width, label='After Optimization', color='seagreen')
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy before and after negative and clonal selection dataset1')
    ax1.set_xticks([i + bar_width / 2 for i in x])
    ax1.set_xticklabels(models)
    ax1.set_ylim(85, 100)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    st.pyplot(fig1)

    # --- Twitter vs Instagram Accuracy per Model ---
    models = ['SVM', 'ANN', 'Random Forest', 'XGBoost']
    
    before = [89.60, 91.08, 93.06, 94.05]
    after = [91.09, 95.05, 96.53, 99.01]

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.bar(x, before, width=bar_width, label='before', color='skyblue')
    ax2.bar([i + bar_width for i in x], after, width=bar_width, label='after', color='violet')
    ax2.set_xlabel('Models')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy before and after negative and clonal selection dataset2')
    ax2.set_xticks([i + bar_width / 2 for i in x])
    ax2.set_xticklabels(models)
    ax2.set_ylim(85, 100)
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    st.pyplot(fig2)

   if st.button("Show Results"):
        st.info("Accuracy Dataset1:")
        st.markdown("- Random Forest: **99.00%**\n- ANN: **95.27%**\n- XGBoost: **97.75%**\n- SVM: **91.84%**")
        st.info("ROC Curves:")
        st.image("fig(2).jpg")
        st.image("fig(3).jpg")

        st.info("Accuracy Dataset2:")
        st.markdown("- Random Forest: **96.53%**\n- ANN: **95.05%**\n- XGBoost: **99.01%**\n- SVM: **91.09%**")
        st.info("ROC Curves:")
        st.image("fig(1).jpg")
        st.image("fig(4).jpg")

       
    
