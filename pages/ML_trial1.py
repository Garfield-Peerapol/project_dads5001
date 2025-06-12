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
    text = re.sub(r"[^\u0E00-\u0E7Fa-zA-Z0-9\\s]", "", text)
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

    vectorizer.tokenizer = thai_tokenizer
    return model, vectorizer, encoder, selector

@st.cache_data
def process_and_predict(df, model, vectorizer, selector, model_type):
    cleaned_texts = [clean_text_combined(text) for text in df.iloc[:, 0]]
    vect_texts = vectorizer.transform(cleaned_texts)
    selected_features = selector.transform(vect_texts)

    if model_type == "Neural Network":
        predictions_probs = model.predict(selected_features)
        predictions_classes = np.argmax(predictions_probs, axis=1)
        decoded_predictions = encoder.inverse_transform(predictions_classes)
        probabilities = np.max(predictions_probs, axis=1)  # Get max probability for each prediction
    else:  # Random Forest
        predictions = model.predict(selected_features)
        decoded_predictions = encoder.inverse_transform(predictions)
        probabilities = model.predict_proba(selected_features).max(axis=1) # Get max probability

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
        st.error("Error: predict_data.csv not found in pages/ directory.")
        st.stop()

    # Model Selection
    model_type = st.selectbox("เลือกโมเดล", ["Random Forest", "Neural Network"])
    st.write(f"คุณเลือกโมเดล: {model_type}")

    if st.button("Run Prediction"):
        with st.spinner(f"Running prediction with {model_type}..."):
            try:
                model, vectorizer, encoder, selector = load_all_models(model_type)
                decoded_predictions, probabilities = process_and_predict(df, model, vectorizer, selector, model_type)

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
                    ["", "Explore Results", "Visualize Prediction Distribution"] # Removed "Show Consistent Predictions"
                )

                if analysis_option == "Explore Results":
                    st.subheader("Explore Prediction Results")
                    results_df = st.session_state['results_df']  # Access from session state
                    label_choice = st.selectbox("Select Label:", ["All Labels"] + results_df['หมวดหมู่ที่ทำนายได้'].unique().tolist())
                    num_samples = st.slider("Number of samples to display:", 5, min(100, len(results_df)), 10)

                    if label_choice == "All Labels":
                        st.dataframe(results_df.head(num_samples))
                    else:
                        st.dataframe(results_df[results_df['หมวดหมู่ที่ทำนายได้'] == label_choice].head(num_samples))

                #elif analysis_option == "Show Consistent Predictions":
                #    st.subheader("Show Consistent Predictions")
                #    results_df = st.session_state['results_df'] # Access from session state

                #    # For single model, there's no consistency to check.
                #    st.info("Consistent predictions are only relevant when using multiple models.  Since you're using a single model, this option isn't applicable.")

                elif analysis_option == "Visualize Prediction Distribution":
                    st.subheader("Visualize Prediction Distribution")
                    results_df = st.session_state['results_df'] # Access from session state

                    vis_type = st.selectbox("Select Visualization Type", ["Pie Chart", "Bar Chart (Probability)"])

                    if vis_type == "Pie Chart":
                        st.subheader("Pie Chart of Predicted Labels")
                        label_counts = results_df['หมวดหมู่ที่ทำนายได้'].value_counts().reset_index()
                        label_counts.columns = ['หมวดหมู่', 'จำนวน']
                        fig_pie = px.pie(label_counts, values='จำนวน', names='หมวดหมู่', title='สัดส่วนของแต่ละหมวดหมู่')
                        st.plotly_chart(fig_pie)

                    elif vis_type == "Bar Chart (Probability)":
                        st.subheader("Bar Chart of Predicted Labels with Probability")
                        # Group by predicted label and calculate mean probability
                        grouped_df = results_df.groupby('หมวดหมู่ที่ทำนายได้')['ความน่าจะเป็น'].mean().reset_index()
                        fig_bar = px.bar(grouped_df, x='หมวดหมู่ที่ทำนายได้', y='ความน่าจะเป็น',
                                        title='Mean Probability per Predicted Label',
                                        labels={'หมวดหมู่ที่ทำนายได้': 'Predicted Label', 'ความน่าจะเป็น': 'Mean Probability'})
                        st.plotly_chart(fig_bar)


            except Exception as e:
                st.error(f"An error occurred: {e}")
