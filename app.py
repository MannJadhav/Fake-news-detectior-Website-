import joblib
import streamlit as st
from scipy.sparse import csr_matrix
import time  # For progress bar simulation

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Load the trained model and vectorizer
try:
    model = joblib.load("model.pkl")  # Ensure your model.pkl exists
    vectorizer = joblib.load("vectorizer.pkl")  # Ensure vectorizer.pkl matches your model training
except FileNotFoundError as e:
    st.error(f"❌ File not found: {e.filename}. Please ensure the file exists.")
    st.stop()
except Exception as e:
    st.error(f"⚠️ An error occurred while loading the model or vectorizer: {e}")
    st.stop()

# Sidebar Information
st.sidebar.title("📝 About the App")
st.sidebar.info("""
🔹 This AI-based Fake News Detector helps you identify whether a news article or headline is real or fake.  
🔹 Uses **Machine Learning** to analyze text patterns and provide predictions.  
🔹 Supports both short and long articles.  
""")

# Custom Styling
st.markdown(
    """
    <style>
    body {
        background-color: #f9f9f9;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 10px;
        padding: 8px 16px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .result-box {
        background-color: #e6f7ff;
        padding: 10px;
        border-radius: 8px;
        font-size: 18px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit App Title
st.title("📰 AI-Based Fake News Detector")
st.write("Enter a news article or headline below to check if it's **Real** or **Fake**.")

# User Input Box
user_input = st.text_area("✍️ Enter News Text Here", "")

# Handle the "Check News" button click
if st.button("🔍 Check News"):
    if user_input.strip():  
        with st.spinner("🔄 Analyzing..."):
            time.sleep(1.5)  # Simulate processing time

        try:
            # Predict using the trained model
            prediction = model.predict([user_input.strip()])[0]

            # Display Results
            result = "✅ **Real News** 🟢" if prediction == 1 else "❌ **Fake News** 🔴"
            explanation = "This article appears **trustworthy**." if prediction == 1 else "This article might be **misleading or false**."

            st.subheader("🧠 Prediction Result:")
            st.markdown(f"<div class='result-box'>{result}</div>", unsafe_allow_html=True)
            st.info(explanation)

        except Exception as e:
            st.error(f"⚠️ An error occurred during prediction: {e}")
    else:
        st.warning("⚠️ Please enter some text to analyze!")

# Footer
st.markdown("---")
st.markdown("📌 Developed with ❤️ using **Machine Learning & Streamlit**")

