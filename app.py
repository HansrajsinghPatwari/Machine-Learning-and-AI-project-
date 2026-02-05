import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import joblib
import pickle

import google.generativeai as genai



genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
genmodel = genai.GenerativeModel("gemini-2.5-flash")


def generate_answer(prompt):
    

    response = genmodel.generate_content(
        prompt,
        generation_config={
            "temperature": 0.3
            
        }
    )

    return response.text

def rag_answer(query):
    

    prompt = f"""
You are a medical AI assistant.
Use the patient's health data and prediction result to give personalized advice.
                 'Age': {age},
                'Gender': {gender}, 
                 'Family_History': {family_hist},
                 'Diabetes': {diabetes},
                 'Hypertension': {hypertension}, # Make sure you have this checkbox
                 'Cholestrol_Level': {chol},
                  'Triglyceride_Level': {trig},
                 'Systolic_BP': {sys_bp},
                 'Diastolic_BP': {dia_bp},
                 'Alcohol_Consumption': {alcohol},
                'Physical_Activity': {physical_activity}, # Make sure you have this checkbox
                'Stress_Level': {stress},
                 'Smoking': {smoking}


Question:
{query}
Guidelines:
- Be simple
- Give lifestyle & preventive advice
- Mention risk factors if relevant
Answer:
"""

    return generate_answer(prompt)



with open('heart_disease_model.pkl', 'rb') as file:
    model = pickle.load(file)





# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Heart Risk Predictor", layout="wide")

# --- LOAD DATA & MODEL ---
@st.cache_data
def load_data():
    # Make sure this CSV file is in the same folder as app.py
    df = pd.read_csv("HeartDisease_dataset.csv") 
    return df



df = load_data()


# --- SIDEBAR: NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard (EDA)", "Risk Prediction"])

# --- PAGE 1: DASHBOARD (Data Visualization) ---
if page == "Dashboard (EDA)":
    st.title("📊 Health Data Analytics Dashboard")
    st.markdown("Explore the dataset to understand key risk factors.")

    # KPI Row
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Patients", len(df))
    col1.metric("Average Age", f"{df['Age'].mean():.1f} years")
    col3.metric("High Risk Patients", f"{len(df[df['Heart_Attack_Risk'] == 1])}")

    st.markdown("---")

    # Row 1: Distribution & Correlation
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Risk Distribution")
        # Pie Chart
        fig_pie = px.pie(df, names='Heart_Attack_Risk', 
                         title='Patients at Risk (1) vs Safe (0)',
                         color_discrete_sequence=['#66b3ff', '#ff9999'])
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.subheader("Correlation Heatmap")
        # Heatmap
        fig_corr, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df.corr(), annot=False, cmap='coolwarm', ax=ax)
        st.pyplot(fig_corr)

    # Row 2: Deep Dive Factors
    st.subheader("⚠️ Critical Risk Factors")
    
    tab1, tab2 = st.tabs(["Cholesterol vs BP", "Age Analysis"])
    
    with tab1:
        # Scatter Plot
        fig_scatter = px.scatter(df, x="Cholestrol_Level", y="Systolic_BP", 
                                 color="Heart_Attack_Risk",
                                 title="Cholesterol vs. Systolic BP (Colored by Risk)",
                                 opacity=0.6)
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption("Notice how high BP and High Cholesterol often appear together in risk patients.")

    with tab2:
        # Box Plot
        fig_box = px.box(df, x="Heart_Attack_Risk", y="Age", 
                         color="Heart_Attack_Risk", 
                         title="Age Distribution by Risk Group")
        st.plotly_chart(fig_box, use_container_width=True)

# --- PAGE 2: PREDICTION (User Interface) ---
elif page == "Risk Prediction":
    st.title(" Heart Attack Risk Predictor")
    
    if model is None:
        st.error("⚠️ Model file not found. Please save your trained model as 'heart_disease_model.pkl' first.")
    else:
        st.write("Enter patient details below to estimate risk.")
        
        # Input Form
        with st.form("prediction_form"):
            c1, c2 = st.columns(2)
            
            # Numeric Inputs (Adjust min/max based on your data)
            age = c1.number_input("Age", min_value=18, max_value=120, value=30)
            chol = c2.number_input("Cholesterol Level", min_value=100, max_value=400, value=200)
            sys_bp = c2.number_input("Systolic BP", min_value=90, max_value=200, value=120)
            
            dia_bp = c1.number_input("Diastolic BP", min_value=60, max_value=130, value=80)
            trig = c1.number_input("Triglycerides", min_value=50, max_value=500, value=150)


            # Categorical Inputs (Drop-downs)
            st.markdown("---")
            col_cat1, col_cat2 = st.columns(2)
            gender = col_cat1.selectbox("Gender?", ["Male", "Female"])
            diabetes = col_cat2.selectbox("Diabetes History?", ["No", "Yes"])
            smoking = col_cat2.selectbox("Smoking Status?", ["No", "Yes"])
            family_hist = col_cat1.selectbox("Family History?", ["No", "Yes"])
            physical_activity = col_cat1.selectbox("Physical Activity?", ["No", "Yes"])
            hypertension = col_cat2.selectbox("Hypertension?", ["No", "Yes"])
            alcohol = col_cat2.selectbox("Do you comsume Alcohol?", ["No", "Yes"])
            stress= col_cat1.selectbox("Stress Level Range?", ["Low",  "High"])
            
            # Submit Button
            submitted = st.form_submit_button("Predict Risk")
        
        if submitted:
            # 1. Create the DataFrame with the EXACT columns and order
            input_data = pd.DataFrame({
                 'Age': [age],
                'Gender': [1 if gender == "Male" else 0], 
                 'Family_History': [1 if family_hist =="Yes" else 0],
                 'Diabetes': [1 if diabetes =="No" else 0],
                 'Hypertension': [1 if hypertension =="Yes" else 0], # Make sure you have this checkbox
                 'Cholestrol_Level': [chol],
                  'Triglyceride_Level': [trig],
                 'Systolic_BP': [sys_bp],
                 'Diastolic_BP': [dia_bp],
                 'Alcohol_Consumption': [1 if alcohol=="Yes" else 0],
                'Physical_Activity': [0 if physical_activity=="No" else 1], # Make sure you have this checkbox
                'Stress_Level': [0 if stress == "Low" else 1],
                 'Smoking': [0 if smoking =="No" else 1]
            })
            print(input_data)
            
            # 2. Make Prediction
            try:
                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data)[0][1] # Probability of Class 1
                
                # 3. Show Result
                st.markdown("---")
                if prediction == 1:
                    st.error(f"High Risk Detected! (Probability: {probability:.2%})")
                    st.write("High Risk of Heart Disease.Please consult a cardiologist immediately.")
                else:
                    st.success(f"Low Risk (Probability: {probability:.2%})")
                    st.write("Keep maintaining a healthy lifestyle!")

                    
            except Exception as e:
                st.warning("Error during prediction. Please check if input features match the model.")
                st.write(e)
        
                st.markdown("---")
        st.subheader("🧠 AI Health Assistant (Chatbot)")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # display previous chats
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat["user"])
            with st.chat_message("assistant"):
                st.write(chat["bot"])

        # input box
        user_input = st.chat_input("Ask about your health, risk, or lifestyle advice...")

        if user_input:
            # show user message
            with st.chat_message("user"):
                st.write(user_input)

            # generate RAG answer
            with st.spinner("Thinking..."):
                bot_reply = rag_answer(user_input)

            # show bot message
            with st.chat_message("assistant"):
                st.write(bot_reply)

            # store chat
            st.session_state.chat_history.append({
                "user": user_input,
                "bot": bot_reply
            })
