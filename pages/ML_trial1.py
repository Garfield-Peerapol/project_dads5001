import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus.common import thai_stopwords
import matplotlib as mpl
import matplotlib.pyplot as plt
from keras.models import load_model
import plotly.express as px

# Set page configuration
st.set_page_config(layout="wide", page_title="Comment Classification")

# --- Global Variables ---
thai_stopwords_set = set(thai_stopwords())

# --- Helper Functions ---
def remove_emojis_and_symbols(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002700-\U000027BF"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    # Corrected regex for general symbols to avoid SyntaxWarning
    text = re.sub(r"[^\u0E00-\u0E7Fa-zA-Z0-9\s]", "", text)
    return text

def clean_text_combined(text):
    if isinstance(text, str):
        text = remove_emojis_and_symbols(text)
        tokens = word_tokenize(text, engine='newmm')
        seen = set()
        unique_tokens = [token for token in tokens if not (token in seen or seen.add(token))]
        filtered_tokens = [token for token in unique_tokens if token not in thai_stopwords_set and token.strip()]
        return " ".join(filtered_tokens)

def thai_tokenizer(text):
    return word_tokenize(text, engine="newmm")

@st.cache_resource
def load_all_models(model_type):
    if model_type == "Random Forest":
        model = joblib.load('pages/final_model_rf.pkl')
        vectorizer = joblib.load('pages/vectorizer_rf.pkl')
        encoder = joblib.load('pages/label_encoder_rf.pkl')
        selector = joblib.load('pages/selector_rf.pkl')
    elif model_type == "Neural Network":
        model = load_model('pages/final_model_NN.keras')
        vectorizer = joblib.load('pages/vectorizer_NN.pkl')
        encoder = joblib.load('pages/label_encoder_NN.pkl')
        selector = joblib.load('pages/selector_NN.pkl')
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    # Assign tokenizer directly, if not already set by joblib load
    if hasattr(vectorizer, 'tokenizer'):
        vectorizer.tokenizer = thai_tokenizer
    return model, vectorizer, encoder, selector

@st.cache_data
def process_and_predict(df, _model, _vectorizer, _selector, model_type, _encoder): # Added underscore to encoder too
    cleaned_texts = [clean_text_combined(text) for text in df.iloc[:, 0]]
    vect_texts = _vectorizer.transform(cleaned_texts)
    selected_features = _selector.transform(vect_texts)

    if model_type == "Neural Network":
        predictions_probs = _model.predict(selected_features)
        predictions_classes = np.argmax(predictions_probs, axis=1)
        decoded_predictions = _encoder.inverse_transform(predictions_classes)
        probabilities = np.max(predictions_probs, axis=1)  # Get max probability for each prediction
    else:  # Random Forest
        predictions = _model.predict(selected_features)
        decoded_predictions = _encoder.inverse_transform(predictions)
        probabilities = _model.predict_proba(selected_features).max(axis=1) # Get max probability

    return decoded_predictions, probabilities

# --- Main Application ---
st.title("🧠 Comment Classification")

# --- Sub-topic Navigation ---
option = st.radio("เลือกหัวข้อย่อย", ["🔍 Preview Comments", "🧪 ML Modeling"])

# --- Section 1: Preview Comments ---
if option == "🔍 Preview Comments":
    st.subheader("🔍 ดูตัวอย่างคอมเมนต์")
    # In a real app, load and show top 5 comments related to each category.  Placeholder for now.
    st.write("แสดงคอมเมนต์ top 5 ที่เกี่ยวกับแต่ละหมวด (Currently a placeholder)")

# --- Section 2: ML Modeling ---
elif option == "🧪 ML Modeling":
    st.subheader("🧪 สร้างโมเดล Machine Learning")

    # Load data
    try:
        df = pd.read_csv('pages/predict_data.csv')
        st.success("โหลดไฟล์สำเร็จ!")
        st.dataframe(df.head(10))
    except FileNotFoundError:
        st.error("Error: predict_data.csv not found in pages/ directory. Please ensure it's there.")
        st.stop()

    # Model Selection
    model_type = st.selectbox("เลือกโมเดล", ["Random Forest", "Neural Network"])
    st.write(f"คุณเลือกโมเดล: **{model_type}**")

    if st.button("Run Prediction"):
        with st.spinner(f"Running prediction with {model_type}..."):
            try:
                model, vectorizer, encoder, selector = load_all_models(model_type)
                # Pass underscore prefixed arguments when calling the cached function
                decoded_predictions, probabilities = process_and_predict(df, model, vectorizer, selector, model_type, encoder)

                # Store results in session state
                st.session_state['results_df'] = pd.DataFrame({
                    'ข้อความต้นฉบับ': df.iloc[:, 0],
                    'หมวดหมู่ที่ทำนายได้': decoded_predictions,
                    'ความน่าจะเป็น': probabilities
                })
                st.session_state['model_type'] = model_type # Store model type for later use

                st.write("✅ ผลการทำนาย:")
                st.dataframe(st.session_state['results_df'])

                # Analysis Options - Moved here, after successful prediction
                st.subheader("📊 Analysis Options")
                analysis_option = st.selectbox(
                    "Select an analysis option:",
                    ["", "1. Explore Prediction Results", "2. Visualize Prediction Distribution"]
                )

                if analysis_option == "1. Explore Prediction Results":
                    st.subheader("1. Explore Prediction Results")
                    results_df = st.session_state['results_df']  # Access from session state
                    
                    if not results_df.empty:
                        all_labels = results_df['หมวดหมู่ที่ทำนายได้'].unique().tolist()
                        label_choice = st.selectbox("Select Label Type:", ["Show All Labels"] + all_labels)
                        num_samples = st.slider("Number of samples to view:", 1, min(1000, len(results_df)), 50) # Adjusted max for sample size

                        filtered_df = results_df.copy()
                        if label_choice != "Show All Labels":
                            filtered_df = filtered_df[filtered_df['หมวดหมู่ที่ทำนายได้'] == label_choice]
                        
                        if not filtered_df.empty:
                            st.dataframe(filtered_df.sample(min(num_samples, len(filtered_df)), random_state=42).reset_index(drop=True))
                        else:
                            st.info("No results match the selected criteria.")
                    else:
                        st.info("No prediction results available to explore.")


                elif analysis_option == "2. Visualize Prediction Distribution":
                    st.subheader("2. Visualize Prediction Distribution")
                    results_df = st.session_state['results_df'] # Access from session state

                    vis_type = st.selectbox("Select Visualization Type", ["Pie Chart", "Bar
