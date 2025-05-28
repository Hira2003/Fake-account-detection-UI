# streamlit_app.py
import streamlit as st

st.set_page_config(page_title="Fake Account Detector", layout="wide")
st.title("🔍 Fake Account Detection System")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Twitter Detection", "Instagram Detection", "About the Study"])

if page == "Twitter Detection":
    from pages import twitter_detection
    twitter_detection.run()
elif page == "Instagram Detection":
    from pages import instagram_detection
    instagram_detection.run()
elif page == "About the Study":
    from pages import about_the_study
    about_the_study.run()

# pages/twitter_detection.py
import streamlit as st
import pandas as pd
import joblib
import gender_guesser.detector as gender

MODEL_FILES = {
    "Random Forest": "fake_account_model_new99.pkl",
    "SVM": "svm_model(1).pkl",
    "XGBoost": "xgb_model-clone.pkl",
    "ANN (MLP)": "fake_account_nn_model.pkl"
}

@st.cache_resource
def load_model(model_key):
    return joblib.load(MODEL_FILES[model_key])

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

def run():
    st.header("🐦 Twitter Account Detection")
    selected_model = st.selectbox("Select Twitter Model", list(MODEL_FILES.keys()))
    model = load_model(selected_model)

    name = st.text_input("Full Name", "Alice Johnson")
    lang = st.selectbox("Language Code", list(lang_dict.keys()))
    statuses_count = st.number_input("Statuses Count", min_value=0, value=120)
    followers_count = st.number_input("Followers Count", min_value=0, value=250)
    friends_count = st.number_input("Friends Count", min_value=0, value=300)
    favourites_count = st.number_input("Favourites Count", min_value=0, value=90)
    listed_count = st.number_input("Listed Count", min_value=0, value=2)
    verified = st.checkbox("Verified", value=False)
    default_profile_image = st.checkbox("Default Profile Image", value=False)

    st.markdown("---")
    uploaded_file = st.file_uploader("Upload Twitter CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(df)} rows.")
        idx = st.number_input("Pick Row", 0, len(df)-1)
        if st.button("Fill from Dataset"):
            selected = df.iloc[int(idx)]
            name = selected.get("name", name)
            lang = selected.get("lang", lang)
            statuses_count = int(selected.get("statuses_count", statuses_count))
            followers_count = int(selected.get("followers_count", followers_count))
            friends_count = int(selected.get("friends_count", friends_count))
            favourites_count = int(selected.get("favourites_count", favourites_count))
            listed_count = int(selected.get("listed_count", listed_count))
            verified = bool(selected.get("verified", False))
            default_profile_image = bool(selected.get("default_profile_image", False))

    if st.button("Predict Twitter Account"):
        sex_code = predict_sex(name)
        lang_code = lang_dict.get(lang, -1)
        features = pd.DataFrame([[
            statuses_count, followers_count, friends_count,
            favourites_count, listed_count, sex_code, lang_code,
            int(verified), int(default_profile_image)
        ]], columns=[
            'statuses_count', 'followers_count', 'friends_count',
            'favourites_count', 'listed_count', 'sex_code', 'lang_code',
            'verified', 'default_profile_image'])
        pred = model.predict(features)[0]
        st.success("Prediction: 🟢 Genuine" if pred == 1 else "🔴 Fake")

# pages/instagram_detection.py
import streamlit as st
import pandas as pd
import joblib

INSTA_MODEL_FILES = {
    "Random Forest": "fake_account_model_new_insta.pkl",
    "SVM": "svm_model_insta.pkl",
    "XGBoost": "xgb_model-clone99_insta.pkl",
    "ANN (MLP)": "fake_account_nn_model_insta.pkl"
}

@st.cache_resource
def load_insta_model(model_key):
    return joblib.load(INSTA_MODEL_FILES[model_key])

def run():
    st.header("📸 Instagram Account Detection")
    selected_model = st.selectbox("Select Instagram Model", list(INSTA_MODEL_FILES.keys()))
    model = load_insta_model(selected_model)

    num_followers = st.number_input("Followers", 0)
    num_following = st.number_input("Following", 0)
    num_posts = st.number_input("Posts", 0)
    extern_url = st.checkbox("External URL", value=False)
    len_desc = st.number_input("Description Length", 0)
    profile_pic = st.checkbox("Has Profile Pic", value=True)
    is_private = st.checkbox("Private", value=False)
    sim_name_username = st.checkbox("Similar Username and Name", value=False)
    len_fullname = st.number_input("Fullname Length", 0)
    ratio_numlen_username = st.number_input("Username Num Ratio", 0.0, 1.0, 0.1)
    ratio_numlen_fullname = st.number_input("Fullname Num Ratio", 0.0, 1.0, 0.1)

    st.markdown("---")
    uploaded_file = st.file_uploader("Upload Instagram CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(df)} rows.")
        idx = st.number_input("Pick Row", 0, len(df)-1)
        if st.button("Fill from Dataset"):
            row = df.iloc[int(idx)]
            num_followers = row.get("num_followers", num_followers)
            num_following = row.get("num_following", num_following)
            num_posts = row.get("num_posts", num_posts)
            extern_url = bool(row.get("extern_url", False))
            len_desc = row.get("len_desc", len_desc)
            profile_pic = bool(row.get("profile_pic", profile_pic))
            is_private = bool(row.get("Private", is_private))
            sim_name_username = bool(row.get("sim_name_username", sim_name_username))
            len_fullname = row.get("len_fullname", len_fullname)
            ratio_numlen_username = row.get("ratio_numlen_username", ratio_numlen_username)
            ratio_numlen_fullname = row.get("ratio_numlen_fullname", ratio_numlen_fullname)

    if st.button("Predict Instagram Account"):
        features = pd.DataFrame([[
            num_followers, num_following, num_posts, int(extern_url), len_desc,
            int(profile_pic), int(is_private), int(sim_name_username),
            len_fullname, ratio_numlen_username, ratio_numlen_fullname
        ]], columns=[
            'num_followers', 'num_following', 'num_posts', 'extern_url', 'len_desc',
            'profile_pic', 'Private', 'sim_name_username',
            'len_fullname', 'ratio_numlen_username', 'ratio_numlen_fullname'])
        pred = model.predict(features)[0]
        st.success("Prediction: 🟢 Genuine" if pred == 1 else "🔴 Fake")

# pages/about_the_study.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def run():
    st.header("📖 About the Study")
    if st.button("Show Datasets"):
        st.markdown("Twitter Sample:")
        st.dataframe(pd.read_csv("fusers.csv").head())
        st.dataframe(pd.read_csv("users.csv").head())
        st.markdown("Instagram Sample:")
        st.dataframe(pd.read_csv("social_media_merged_numeric.csv").head())

    if st.button("Show Figures"):
        st.title("Twitter Accuracies Before and After Negative and Clonal Selection")
        models = ['SVM', 'ANN', 'RF', 'XGB']
        before = [91, 93.2, 95.2, 92.1]
        after = [91.8, 95.3, 99, 97.8]
        x = range(len(models))
        fig, ax = plt.subplots()
        ax.bar(x, before, width=0.4, label='Before', align='center')
        ax.bar([i + 0.4 for i in x], after, width=0.4, label='After', align='center')
        ax.set_xticks([i + 0.2 for i in x])
        ax.set_xticklabels(models)
        ax.set_ylabel("Accuracy %")
        ax.legend()
        st.pyplot(fig)

        st.title("Instagram Accuracies Before and After Negative and Clonal Selection")
        models = ['SVM', 'ANN', 'RF', 'XGB']
        before = [89.60, 91.08, 93.06, 94.05]
        after = [91.09, 95.05, 96.53, 99.01]
        x = range(len(models))
        fig, ax = plt.subplots()
        ax.bar(x, before, width=0.4, label='Before', align='center')
        ax.bar([i + 0.4 for i in x], after, width=0.4, label='After', align='center')
        ax.set_xticks([i + 0.2 for i in x])
        ax.set_xticklabels(models)
        ax.set_ylabel("Accuracy %")
        ax.legend()
        st.pyplot(fig)


    if st.button("Show Results"):
        st.info("Random Forest (Twitter): 99% accuracy\nSVM (Insta): 91.8%\nConfusion Matrices available upon request")

