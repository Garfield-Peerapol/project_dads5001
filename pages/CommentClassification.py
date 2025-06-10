import streamlit as st

st.title("🧠 Comment Classification")

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

    df = pd.read_csv(uploaded_file)
    #st.success("โหลดไฟล์สำเร็จ!")
    #st.write("📌 ตัวอย่างข้อมูล:")
    st.dataframe(df.head(10))
    
    
    # ฟังก์ชันทำความสะอาดข้อความ
    thai_stopwords_set = set(thai_stopwords())
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

    #select random forest
    if model_type=="Random Forest":
        def load_all_models():
            model = joblib.load('final_model_rf.pkl')
            vectorizer = joblib.load('vectorizer_rf.pkl')
            encoder = joblib.load('label_encoder_rf.pkl')
            selector = joblib.load('selector_rf.pkl')
            return model, vectorizer, encoder, selector
            
        model, vectorizer, encoder, selector = load_all_models()

        #processing and predict data
        cleaned_texts = [clean_text_combined(text) for text in df.iloc[:,0]]
        vect_texts = vectorizer.transform(cleaned_texts)
        selected_features = selector.transform(vect_texts)
        predictions = model.predict(selected_features)
        decoded_predictions = encoder.inverse_transform(predictions)

    #select Neural Network
    if model_type=="Neural Network":
        def load_all_models():
            model = joblib.load('final_model_NN.pkl')
            vectorizer = joblib.load('vectorizer_NN.pkl')
            encoder = joblib.load('label_encoder_NN.pkl')
            selector = joblib.load('selector_NN.pkl')
            return model, vectorizer, encoder, selector
            
        model, vectorizer, encoder, selector = load_all_models()

        #processing and predict data
        cleaned_new_texts = [clean_text_combined(text) for text in df.iloc[:,0]]
        vect_new_texts = vectorizer.transform(cleaned_new_texts)
        selected_features_new_texts = selector.transform(vect_new_texts)  # Apply selector
        new_predictions_probs = model.predict(selected_features_new_texts)
        new_predictions_classes = np.argmax(new_predictions_probs, axis=1)
        decoded_predictions = encoder.inverse_transform(new_predictions_classes)

    # แสดงผลลัพธ์
    results_df = pd.DataFrame({
        'ข้อความต้นฉบับ': df.iloc[:,0],
        'หมวดหมู่ที่ทำนายได้': decoded_predictions
    })

    st.write("✅ ผลการทำนาย:")
    st.dataframe(results_df)
        




