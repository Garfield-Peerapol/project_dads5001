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
import time

#1 load original data
df = pd.read_csv('pages/predict_data.csv')

#2 def text preprocessing function
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


#3 processing for random forest
@st.cache_resource
model_rf = joblib.load('pages/final_model_rf.pkl')
vectorizer_rf = joblib.load('pages/vectorizer_rf.pkl')
encoder_rf = joblib.load('pages/label_encoder_rf.pkl')
selector_rf = joblib.load('pages/selector_rf.pkl')
vectorizer.tokenizer = thai_tokenizer

cleaned_texts = [clean_text_combined(text) for text in df.iloc[:,0]]
vect_texts = vectorizer.transform(cleaned_texts)
selected_features = selector.transform(vect_texts)
predictions = model.predict(selected_features)
decoded_predictions_rf = encoder.inverse_transform(predictions)

df_rf=pd.DataFrame({
        'Comment': df.iloc[:,0],
        'Predicted Label': decoded_predictions_rf
    })


# Neural Network
@st.cache_resource
model_nn = load_model('pages/final_model_NN.keras') 
vectorizer_nn = joblib.load('pages/vectorizer_NN.pkl')
encoder_nn = joblib.load('pages/label_encoder_NN.pkl')
selector_nn = joblib.load('pages/selector_NN.pkl')
vectorizer.tokenizer = thai_tokenizer

cleaned_new_texts = [clean_text_combined(text) for text in df.iloc[:,0]]
vect_new_texts = vectorizer.transform(cleaned_new_texts)
selected_features_new_texts = selector.transform(vect_new_texts)  # Apply selector
new_predictions_probs = model.predict(selected_features_new_texts)
new_predictions_classes = np.argmax(new_predictions_probs, axis=1)
decoded_predictions = encoder.inverse_transform(new_predictions_classes)

df_nn=pd.DataFrame({
        'Comment': df.iloc[:,0],
        'Predicted Label': decoded_predictions_rf
    })

st.dataframe(df_rf)
st.dataframe(df_nn)





#3
"""



st.title("🧠 Comment Classification")
st.write("กดปุ่มเพิ่อเริ่มการทำงาน")



# --- Sub-topic Navigation ---
option = st.radio("เลือกหัวข้อย่อย", ["🔍 Preview Comments", "🧪 ML Modeling"])

# --- Section 1 ---
if option == "🔍 Preview Comments":
    st.subheader("🔍 ดูตัวอย่างคอมเมนต์")
    # ดึงคอมเมนต์จาก MongoDB หรือ CSV แล้วโชว์
    st.write("แสดงคอมเมนต์ top 5 ที่เกี่ยวกับแต่ละหมวด")

# --- Section 2 ---
elif option == "🧪 ML Modeling":
    st.subheader("🧪 สร้างโมเดล Machine Learning")
    # เลือก model (Random Forest, Neural Network, etc.)
    model_type = st.selectbox("เลือกโมเดล", ["Random Forest", "Neural Network"])
    st.write(f"คุณเลือกโมเดล: {model_type}")
    # โหลด stopwords ภาษาไทย

    #st.success("โหลดไฟล์สำเร็จ!")
    #st.write("📌 ตัวอย่างข้อมูล:")
    st.dataframe(df.head(10))
    
    
    # ฟังก์ชันทำความสะอาดข้อความ
    thai_stopwords_set = set(thai_stopwords())

    


    #select random forest
    if model_type=="Random Forest":
        

        #processing and predict data

    
    
    #select Neural Network
    if model_type=="Neural Network":

            
        from pythainlp.tokenize import word_tokenize
        model, vectorizer, encoder, selector = load_all_models_NN()
        

        #processing and predict data

    
    
    # แสดงผลลัพธ์
    results_df = pd.DataFrame({
        'ข้อความต้นฉบับ': df.iloc[:,0],
        'หมวดหมู่ที่ทำนายได้': decoded_predictions
    })

    st.write("✅ ผลการทำนาย:")
    st.dataframe(results_df)
    """    
