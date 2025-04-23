# 🕵️‍♀️ Fake Account Detector

A stylish and smart web app to **detect fake social media accounts** using multiple machine learning models. Built with ❤️ using **Streamlit** and optimized with features like gender prediction and language handling.

## 🌟 Features

- ✅ **Supports multiple models**: Random Forest, SVM, XGBoost, and ANN (MLP)
- 🧠 **Gender inference** using the user's name
- 🌍 **Language detection** with 19+ language options
- 🧾 Special features for Random Forest: `verified`, `default_profile_image`
- 🎨 Beautiful, responsive UI with background art
- 🔒 No personal data stored, all predictions happen locally

---

## 🧠 Models Used

| Model         | Extra Features         |
|---------------|------------------------|
| Random Forest | `verified`, `default_profile_image` |
| SVM           | —                      |
| XGBoost       | —                      |
| ANN (MLP)     | —                      |

Each model was trained on profile-based features like:
- Statuses count
- Followers & Friends count
- Favourites & Listed count
- Name-based gender prediction
- Language code

---

## 🚀 How to Run

1. Clone this repo:
   ```bash
   git clone https://github.com/yourusername/fake-account-detector.git
   cd fake-account-detector
   pip install -r requirements.txt
   streamlit run Fad-ui.py
   .
   ├── Fad-ui.py                 # Main Streamlit app
   ├── models/                   # Contains saved .pkl model files
   ├── assets/                   # Background illustrations & UI images
   ├── requirements.txt          # Python dependencies
   └── README.md                 # This file

🐱‍💻 Author

Hira Fuyu
🎓 Master’s student | 💻 Programmer | 🎶 Music lover
📺 YouTube - Hirafuyu: Programming Garden
🌸 “May this little garden grow into something beautiful...”
![image](https://github.com/user-attachments/assets/0e5cb6d1-761e-4361-a230-a9f9fdb0a50a)

