import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus.common import thai_stopwords
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from keras.models import load_model
import random

# Configure page
st.set_page_config(page_title="Comment Classification", page_icon="🧠", layout="wide")
st.title("🧠 Comment Classification System")

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'results_data' not in st.session_state:
    st.session_state.results_data = None

# Thai stopwords and cleaning functions
@st.cache_resource
def get_thai_stopwords():
    return set(thai_stopwords())

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

def clean_text_combined(text, thai_stopwords_set):
    if isinstance(text, str):
        text = remove_emojis_and_symbols(text)
        tokens = word_tokenize(text, engine='newmm')
        seen = set()
        unique_tokens = [token for token in tokens if not (token in seen or seen.add(token))]
        filtered_tokens = [token for token in unique_tokens if token not in thai_stopwords_set and token.strip()]
        return " ".join(filtered_tokens)
    return ""

def thai_tokenizer(text):
    return word_tokenize(text, engine="newmm")

@st.cache_resource
def load_all_models():
    """Load all models and return them"""
    try:
        # Load Random Forest models
        rf_model = joblib.load('pages/final_model_rf.pkl')
        rf_vectorizer = joblib.load('pages/vectorizer_rf.pkl')
        rf_encoder = joblib.load('pages/label_encoder_rf.pkl')
        rf_selector = joblib.load('pages/selector_rf.pkl')
        rf_vectorizer.tokenizer = thai_tokenizer
        
        # Load Neural Network models
        nn_model = load_model('pages/final_model_NN.keras')
        nn_vectorizer = joblib.load('pages/vectorizer_NN.pkl')
        nn_encoder = joblib.load('pages/label_encoder_NN.pkl')
        nn_selector = joblib.load('pages/selector_NN.pkl')
        nn_vectorizer.tokenizer = thai_tokenizer
        
        return {
            'rf': {'model': rf_model, 'vectorizer': rf_vectorizer, 'encoder': rf_encoder, 'selector': rf_selector},
            'nn': {'model': nn_model, 'vectorizer': nn_vectorizer, 'encoder': nn_encoder, 'selector': nn_selector}
        }
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

def predict_with_models(texts, models, thai_stopwords_set):
    """Predict with all models and return results with probabilities"""
    results = {}
    
    # Clean texts
    cleaned_texts = [clean_text_combined(text, thai_stopwords_set) for text in texts]
    
    # Random Forest predictions
    try:
        rf_models = models['rf']
        vect_texts = rf_models['vectorizer'].transform(cleaned_texts)
        selected_features = rf_models['selector'].transform(vect_texts)
        rf_predictions = rf_models['model'].predict(selected_features)
        rf_probabilities = rf_models['model'].predict_proba(selected_features)
        rf_decoded = rf_models['encoder'].inverse_transform(rf_predictions)
        
        results['Random Forest'] = {
            'predictions': rf_decoded,
            'probabilities': rf_probabilities,
            'classes': rf_models['encoder'].classes_
        }
    except Exception as e:
        st.error(f"Error with Random Forest: {str(e)}")
    
    # Neural Network predictions
    try:
        nn_models = models['nn']
        vect_texts = nn_models['vectorizer'].transform(cleaned_texts)
        selected_features = nn_models['selector'].transform(vect_texts)
        nn_probabilities = nn_models['model'].predict(selected_features)
        nn_predictions = np.argmax(nn_probabilities, axis=1)
        nn_decoded = nn_models['encoder'].inverse_transform(nn_predictions)
        
        results['Neural Network'] = {
            'predictions': nn_decoded,
            'probabilities': nn_probabilities,
            'classes': nn_models['encoder'].classes_
        }
    except Exception as e:
        st.error(f"Error with Neural Network: {str(e)}")
    
    return results

# Main interface
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("🚀 เริ่มการทำงาน")
    
    if st.button("🔄 โหลดโมเดลและเริ่มทำงาน", type="primary"):
        with st.spinner("กำลังโหลดโมเดลและประมวลผล..."):
            try:
                # Load data
                df = pd.read_csv('pages/predict_data.csv')
                
                # Load models
                models = load_all_models()
                if models is None:
                    st.error("ไม่สามารถโหลดโมเดลได้")
                    st.stop()
                
                # Get Thai stopwords
                thai_stopwords_set = get_thai_stopwords()
                
                # Predict with all models
                results = predict_with_models(df.iloc[:, 0].tolist(), models, thai_stopwords_set)
                
                # Store results in session state
                st.session_state.results_data = {
                    'original_data': df,
                    'predictions': results,
                    'texts': df.iloc[:, 0].tolist()
                }
                st.session_state.models_loaded = True
                
                st.success("โหลดโมเดลและประมวลผลเสร็จสิ้น!")
                
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาด: {str(e)}")

with col2:
    if st.session_state.models_loaded and st.session_state.results_data:
        st.subheader("📊 เลือกการแสดงผล")
        
        display_option = st.selectbox(
            "เลือกออปชันการแสดงผล",
            ["ออปชัน 1: แสดงผลตามโมเดลและลาเบล", 
             "ออปชัน 2: แสดงผลที่ทำนายเหมือนกันทุกโมเดล", 
             "ออปชัน 3: แสดงกราฟสถิติ"]
        )
        
        results_data = st.session_state.results_data
        
        # Option 1: Show results by model and label
        if display_option == "ออปชัน 1: แสดงผลตามโมเดลและลาเบล":
            st.subheader("🎯 ผลการทำนายตามโมเดลและลาเบล")
            
            col1_opt1, col2_opt1, col3_opt1 = st.columns(3)
            
            with col1_opt1:
                selected_model = st.selectbox("เลือกโมเดล", list(results_data['predictions'].keys()))
            
            with col2_opt1:
                num_samples = st.number_input("จำนวนตัวอย่าง", min_value=1, max_value=len(results_data['texts']), value=10)
            
            with col3_opt1:
                # Get unique labels from selected model
                unique_labels = list(set(results_data['predictions'][selected_model]['predictions']))
                label_options = ["ทั้งหมด"] + unique_labels
                selected_label = st.selectbox("เลือกลาเบล", label_options)
            
            # Filter and display results
            model_predictions = results_data['predictions'][selected_model]['predictions']
            model_probabilities = results_data['predictions'][selected_model]['probabilities']
            
            # Create results dataframe
            results_df = pd.DataFrame({
                'ข้อความ': results_data['texts'],
                'ลาเบลที่ทำนาย': model_predictions,
                'ความมั่นใจสูงสุด': [np.max(prob) for prob in model_probabilities]
            })
            
            # Filter by label if not "ทั้งหมด"
            if selected_label != "ทั้งหมด":
                results_df = results_df[results_df['ลาเบลที่ทำนาย'] == selected_label]
            
            # Sample results
            if len(results_df) > num_samples:
                display_df = results_df.sample(n=num_samples).reset_index(drop=True)
            else:
                display_df = results_df
            
            st.dataframe(display_df, use_container_width=True)
            st.write(f"แสดง {len(display_df)} จาก {len(results_df)} รายการ")
        
        # Option 2: Show consensus predictions
        elif display_option == "ออปชัน 2: แสดงผลที่ทำนายเหมือนกันทุกโมเดล":
            st.subheader("🤝 ผลการทำนายที่ทุกโมเดลเห็นตรงกัน")
            
            # Find consensus predictions
            model_names = list(results_data['predictions'].keys())
            if len(model_names) >= 2:
                consensus_mask = []
                for i in range(len(results_data['texts'])):
                    predictions = [results_data['predictions'][model]['predictions'][i] for model in model_names]
                    consensus_mask.append(len(set(predictions)) == 1)  # All predictions are the same
                
                consensus_df = pd.DataFrame({
                    'ข้อความ': [results_data['texts'][i] for i in range(len(results_data['texts'])) if consensus_mask[i]],
                    'ลาเบลที่ทุกโมเดลเห็นตรงกัน': [results_data['predictions'][model_names[0]]['predictions'][i] for i in range(len(results_data['texts'])) if consensus_mask[i]]
                })
                
                if len(consensus_df) > 0:
                    st.dataframe(consensus_df, use_container_width=True)
                    st.write(f"พบ {len(consensus_df)} รายการที่ทุกโมเดลทำนายเหมือนกัน")
                else:
                    st.warning("ไม่พบรายการที่ทุกโมเดลทำนายเหมือนกัน")
            else:
                st.warning("ต้องมีอย่างน้อย 2 โมเดลเพื่อเปรียบเทียบ")
        
        # Option 3: Show statistics charts
        elif display_option == "ออปชัน 3: แสดงกราฟสถิติ":
            st.subheader("📈 กราฟสถิติการทำนาย")
            
            # 3.1 Pie charts for each model
            st.subheader("3.1 สัดส่วนลาเบลที่แต่ละโมเดลทำนาย")
            
            cols = st.columns(len(results_data['predictions']))
            for i, (model_name, model_data) in enumerate(results_data['predictions'].items()):
                with cols[i]:
                    predictions = model_data['predictions']
                    label_counts = pd.Series(predictions).value_counts()
                    
                    fig_pie = px.pie(
                        values=label_counts.values,
                        names=label_counts.index,
                        title=f"{model_name}"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            # 3.2 Bar charts with probability breakdown
            st.subheader("3.2 จำนวนลาเบลแยกตามระดับความมั่นใจ")
            
            for model_name, model_data in results_data['predictions'].items():
                st.write(f"**{model_name}**")
                
                predictions = model_data['predictions']
                probabilities = model_data['probabilities']
                
                # Create probability bins
                prob_bins = ['0.0-0.5', '0.5-0.7', '0.7-0.9', '0.9-1.0']
                max_probs = [np.max(prob) for prob in probabilities]
                
                # Create dataframe for plotting
                plot_data = []
                for i, (pred, max_prob) in enumerate(zip(predictions, max_probs)):
                    if max_prob < 0.5:
                        prob_bin = '0.0-0.5'
                    elif max_prob < 0.7:
                        prob_bin = '0.5-0.7'
                    elif max_prob < 0.9:
                        prob_bin = '0.7-0.9'
                    else:
                        prob_bin = '0.9-1.0'
                    
                    plot_data.append({
                        'label': pred,
                        'prob_bin': prob_bin,
                        'count': 1
                    })
                
                plot_df = pd.DataFrame(plot_data)
                pivot_df = plot_df.groupby(['label', 'prob_bin']).count().reset_index()
                pivot_df = pivot_df.pivot(index='label', columns='prob_bin', values='count').fillna(0)
                
                # Create stacked bar chart
                fig_bar = go.Figure()
                
                colors = ['#ff9999', '#ffcc99', '#99ccff', '#99ff99']
                for i, prob_bin in enumerate(prob_bins):
                    if prob_bin in pivot_df.columns:
                        fig_bar.add_trace(go.Bar(
                            name=f'ความมั่นใจ {prob_bin}',
                            x=pivot_df.index,
                            y=pivot_df[prob_bin],
                            marker_color=colors[i]
                        ))
                
                fig_bar.update_layout(
                    barmode='stack',
                    title=f'การกระจายความมั่นใจของ {model_name}',
                    xaxis_title='ลาเบล',
                    yaxis_title='จำนวน'
                )
                
                st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("👆 กดปุ่ม 'โหลดโมเดลและเริ่มทำงาน' เพื่อเริ่มต้น")

# Sidebar information
with st.sidebar:
    st.header("ℹ️ ข้อมูลระบบ")
    st.write("**โมเดลที่รองรับ:**")
    st.write("- Random Forest")
    st.write("- Neural Network")
    
    st.write("**ไฟล์ที่ต้องการ:**")
    st.write("- predict_data.csv")
    st.write("- final_model_rf.pkl")
    st.write("- vectorizer_rf.pkl")
    st.write("- label_encoder_rf.pkl")
    st.write("- selector_rf.pkl")
    st.write("- final_model_NN.keras")
    st.write("- vectorizer_NN.pkl")
    st.write("- label_encoder_NN.pkl")
    st.write("- selector_NN.pkl")
    
    if st.session_state.models_loaded:
        st.success("✅ ระบบพร้อมใช้งาน")
        if st.session_state.results_data:
            st.write(f"**ข้อมูลที่ประมวลผล:** {len(st.session_state.results_data['texts'])} รายการ")
    else:
        st.warning("⏳ รอการโหลดระบบ")
