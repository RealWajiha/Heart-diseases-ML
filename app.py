import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

# Page Config
st.set_page_config(
    page_title="Heart Guard AI - Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Neon Aesthetic)
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .main-title {
        color: #ff2a7e;
        text-align: center;
        font-size: 2.8rem;
        font-weight: 800;
        text-shadow: 0 0 10px rgba(255, 42, 126, 0.5);
        margin-bottom: 5px;
    }
    .sub-title {
        text-align: center;
        color: #a0aab2;
        margin-bottom: 25px;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 42, 126, 0.2);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    .stButton>button {
        background: linear-gradient(135deg, #ff2a7e 0%, #a100ff 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255, 42, 126, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-title'>❤️ Heart Guard AI Platform</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Machine Learning Powered Heart Disease Risk Analysis & Prediction</p>", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/isometric/100/null/heart-with-pulse.png", width=80)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to Page:", ["📊 Model Training & Dataset", "🩺 Patient Prediction"])

MODEL_PATH = 'heart_disease_model.pkl'
SCALER_PATH = 'scaler.pkl'

# ---------------------------------------------------------
# PAGE 1: MODEL TRAINING & DATA ANALYSIS
# ---------------------------------------------------------
if page == "📊 Model Training & Dataset":
    st.header("1. Upload Dataset & Train Model")
    
    uploaded_file = st.file_uploader("Heart Disease CSV Dataset Upload Karein (`heart.csv`)", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        tab1, tab2, tab3 = st.tabs(["📋 Data Preview", "📈 Visualizations", "⚙️ Model Training"])
        
        with tab1:
            st.subheader("Dataset Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Rows", df.shape[0])
            col2.metric("Total Columns", df.shape[1])
            col3.metric("Missing Values", df.isnull().sum().sum())
            
            st.write("### Data Head:")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.write("### Descriptive Statistics:")
            st.dataframe(df.describe().T, use_container_width=True)

        with tab2:
            st.subheader("Feature Correlations & Graphs")
            numeric_df = df.select_dtypes(include=[np.number])
            
            fig_corr = px.imshow(
                numeric_df.corr(),
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="Viridis",
                title="Feature Correlation Heatmap"
            )
            st.plotly_chart(fig_corr, use_container_width=True)

        with tab3:
            st.subheader("Random Forest Classifier Train Karein")
            
            cols = df.columns.tolist()
            target_col = st.selectbox("Target Column Select Karein (Presence of Disease):", cols, index=len(cols)-1)
            
            test_size = st.slider("Test Data Percentage", 0.1, 0.4, 0.2, 0.05)
            n_estimators = st.slider("Random Forest Trees (n_estimators)", 50, 300, 100, 10)
            
            if st.button("🚀 Train Model Now"):
                with st.spinner("Model Train Ho Raha Hai... Please wait!"):
                    X = df.drop(columns=[target_col])
                    y = df[target_col]
                    
                    # Convert Categorical features if any
                    X = pd.get_dummies(X, drop_first=True)
                    
                    # Split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42, stratify=y
                    )
                    
                    # Scale
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Model
                    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
                    model.fit(X_train_scaled, y_train)
                    
                    # Evaluate
                    y_pred = model.predict(X_test_scaled)
                    acc = accuracy_score(y_test, y_pred)
                    
                    # Save
                    joblib.dump(model, MODEL_PATH)
                    joblib.dump(scaler, SCALER_PATH)
                    joblib.dump(X.columns.tolist(), 'feature_columns.pkl')
                    
                    st.success(f"✅ Model Successfully Train Ho Gaya! Accuracy: **{acc * 100:.2f}%**")
                    
                    # Metrics Display
                    col_m1, col_m2 = st.columns(2)
                    col_m1.metric("Model Accuracy", f"{acc * 100:.2f}%")
                    
                    # Confusion Matrix Plot
                    cm = confusion_matrix(y_test, y_pred)
                    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                       labels=dict(x="Predicted", y="Actual"),
                                       x=['No Disease', 'Heart Disease'],
                                       y=['No Disease', 'Heart Disease'],
                                       title="Confusion Matrix")
                    col_m2.plotly_chart(fig_cm, use_container_width=True)

    else:
        st.info("💡 Please upload `heart.csv` file to proceed with model training and exploration.")

# ---------------------------------------------------------
# PAGE 2: PATIENT PREDICTION FORM
# ---------------------------------------------------------
elif page == "🩺 Patient Prediction":
    st.header("2. Patient Health Metrics Input")
    
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        st.warning("⚠️ Trained model nahi mila! Pehle 'Model Training & Dataset' tab par ja kar model train karein ya model file upload karein.")
    else:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        
        st.markdown("Enter patient medical parameters below:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age (Years)", min_value=1, max_value=120, value=52)
            sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
            cp = st.selectbox("Chest Pain Type", options=[
                ("Typical Angina (0)", 0),
                ("Atypical Angina (1)", 1),
                ("Non-anginal Pain (2)", 2),
                ("Asymptomatic (3)", 3)
            ], format_func=lambda x: x[0])[1]
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=125)

        with col2:
            chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=212)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[("No (0)", 0), ("Yes (1)", 1)], format_func=lambda x: x[0])[1]
            restecg = st.selectbox("Resting ECG Results", options=[
                ("Normal (0)", 0),
                ("ST-T Wave Abnormality (1)", 1),
                ("Left Ventricular Hypertrophy (2)", 2)
            ], format_func=lambda x: x[0])[1]
            thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=168)

        with col3:
            exang = st.selectbox("Exercise Induced Angina", options=[("No (0)", 0), ("Yes (1)", 1)], format_func=lambda x: x[0])[1]
            oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            slope = st.selectbox("Slope of Peak Exercise ST", options=[("Upsloping (0)", 0), ("Flat (1)", 1), ("Downsloping (2)", 2)], format_func=lambda x: x[0])[1]
            ca = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3], index=0)
            thal = st.selectbox("Thalassemia (thal)", options=[("Normal (1)", 1), ("Fixed Defect (2)", 2), ("Reversible Defect (3)", 3)], format_func=lambda x: x[0])[1]

        st.markdown("---")
        
        if st.button("🔮 Predict Heart Disease Risk", use_container_width=True):
            input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
            
            # Scale input
            scaled_input = scaler.transform(input_data)
            
            # Predict
            prediction = model.predict(scaled_input)[0]
            probability = model.predict_proba(scaled_input)[0][1] * 100
            
            st.subheader("Prediction Result:")
            
            res_col1, res_col2 = st.columns([1, 1])
            
            with res_col1:
                if prediction == 1:
                    st.error(f"⚠️ **High Risk of Heart Disease Detected!**\n\nProbability: **{probability:.1f}%**")
                    st.warning("⚠️ Recommendation: Consult a Cardiologist for detailed evaluation and clinical examination.")
                else:
                    st.success(f"✅ **Low Risk / No Heart Disease Detected.**\n\nProbability: **{probability:.1f}%**")
                    st.info("👍 Recommendation: Maintain a healthy lifestyle, balanced diet, and regular exercise.")
            
            with res_col2:
                # Gauge Chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Heart Disease Probability (%)", 'font': {'size': 18}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#ff2a7e" if probability > 50 else "#00e676"},
                        'steps': [
                            {'range': [0, 50], 'color': "rgba(0, 230, 118, 0.1)"},
                            {'range': [50, 100], 'color': "rgba(255, 42, 126, 0.1)"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_gauge, use_container_width=True)