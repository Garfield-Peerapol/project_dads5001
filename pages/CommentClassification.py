import streamlit as st

st.title("🧠 Comment Classification")

# --- Sub-topic Navigation ---
option = st.radio("เลือกหัวข้อย่อย", ["🔍 Preview Comments", "🧪 ML Modeling", "📈 Result Visualization"])

# --- Section 1 ---
if option == "🔍 Preview Comments":
    st.subheader("🔍 ดูตัวอย่างคอมเมนต์")
    # ดึงคอมเมนต์จาก MongoDB หรือ CSV แล้วโชว์
    st.write("แสดงคอมเมนต์ top 5 ที่เกี่ยวกับแต่ละหมวด")

# --- Section 2 ---
elif option == "🧪 ML Modeling":
    st.subheader("🧪 สร้างโมเดล Machine Learning")
    # เลือก model (Naive Bayes, SVM, etc.)
    model_type = st.selectbox("เลือกโมเดล", ["Naive Bayes", "SVM", "Random Forest"])
    st.write(f"คุณเลือกโมเดล: {model_type}")

# --- Section 3 ---
elif option == "📈 Result Visualization":
    st.subheader("📈 ผลการจำแนกคอมเมนต์")
    # แสดง confusion matrix หรือ accuracy score
    st.write("แสดงกราฟ performance ของโมเดล")

