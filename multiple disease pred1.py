
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
#import seaborn as sns
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import requests
import joblib


# loading the saved models

diabetes_model2 = pickle.load(open('trained models sav files/diabetes_model.sav', 'rb'))

#diabetes_model = pickle.load(open('C:/Users/arnav/Desktop/Major Project/Major Project/Trained Model/diabetes_model1.sav', 'rb'))

heart_disease_model = pickle.load(open('trained models sav files/heart_disease_model.sav', 'rb'))

#heart_disease_model = pickle.load(open('C:/Users/arnav/Desktop/Major Project/Major Project/Trained Model/heart_disease_model.sav','rb'))

parkinsons_model = pickle.load(open('trained models sav files/parkinsons_model.sav', 'rb'))

diabetes_model1 = pickle.load(open('trained models sav files/diabetes_model.sav', 'rb'))

breastcancer_model = pickle.load(open('trained models sav files/breast_cancer_model (2).sav', 'rb'))

lungcancer_model = pickle.load(open('trained models sav files/lungcancer_model.sav', 'rb'))



st.set_page_config(page_title="Multiple Disease Prediction System webapp by Arnav", layout="centered")





# Sidebar for navigation
with st.sidebar:
    selected = option_menu(
        'MULTI-DISEASE PREDICTION SYSTEM',
        ['CHOOSE AN OPTION',  # placeholder item
         'Diabetes',
         'Heart Disease',
         "Parkinson's Disease",
         'Breast Cancer',
         'Lung Cancer',
         ],
        icons=['question-circle', '1-circle-fill', '2-circle-fill', '3-circle-fill', '4-circle-fill', '5-circle-fill', '6-circle-fill'],
        menu_icon='hospital',
        default_index=0
    )
    

if selected == 'CHOOSE AN OPTION':
    
    st.success("👨🏻‍💻 WELCOME TO THIS ML-BASED MENU-DRIVEN HEALTH PREDICTION WEB APP.")
    st.info("🖥 Designed with a user-friendly interface and backed by real medical datasets and models, this system uses advanced ML algorithms to predict various health conditions based on user input including: Diabetes, Heart Disease, Parkinson's Disease, Breast Cancer, Lung Cancer.")
    st.error("🚀 Choose a prediction model from the sidebar to get started.")
    st.warning("⚠️ This tool is for educational use and not a substitute for professional medical advice!")
    col1, col2, col3 = st.columns([1, 0.5, 1])
    with col2:
        st.image('images/IMG-20241119-WA0030.jpg', width=150)

    




                          
# Helper to check empty fields
def check_empty(inputs):
    return any(str(val).strip() == "" for val in inputs)



# Diabetes num Prediction Page
if selected == 'Diabetes':
    
    # Sidebar information
    st.sidebar.header("About the Input Terms")
    
    st.sidebar.markdown("""
    - **Age of the Person**  
      The age (in years) of the individual. Age can influence disease risk.
    
    - **Glucose Level**  
      The concentration of glucose (sugar) in the blood. High glucose levels are a key indicator of diabetes.
    
    - **Blood Pressure Value**  
      The pressure of circulating blood on the walls of blood vessels. High blood pressure is associated with diabetes.
    
    - **Skin Thickness Value**  
      Measurement of the thickness of a fold of skin (in millimeters), usually taken at the tricep, to estimate body fat.
    
    - **Insulin Level**  
      The amount of insulin hormone present in the blood (measured in mu U/ml), important for blood sugar regulation.
    
    - **BMI Value (Body Mass Index)**  
      A value calculated from height and weight to categorize underweight, normal, overweight, or obesity. Higher BMI is linked to greater diabetes risk.
    
    - **Number of Pregnancies**  
      The number of times the person has been pregnant. Pregnancy history can influence diabetes risk in women.
    
    - **Diabetes Pedigree Function Value**  
      A score that summarizes genetic risk for diabetes based on family history. Higher values suggest a stronger hereditary risk.
    """)


    
    st.title('Diabetes Prediction using ML 🩸')
    st.markdown("ENTER HEALTH PARAMETERS TO CHECK FOR DIABETES:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Age = st.slider('Age of the Person', min_value=1, max_value=100, value=None, help='The age (in years) of the individual. Age can influence disease risk.')
    
    with col2:
        Glucose = st.number_input('Glucose Level', min_value=0, max_value=200, value=None, help='The concentration of glucose (sugar) in the blood.', placeholder="Enter glucose level (mg/dL)")
    
    with col3:
        BloodPressure = st.number_input('Blood Pressure value', min_value=0, max_value=125, value=None, help='The pressure of circulating blood on the walls of blood vessels.', placeholder="Enter blood pressure (mmHg)")
    
    with col1:
        SkinThickness = st.number_input('Skin Thickness value', min_value=0, max_value=50, value=None, help='Measurement of skin fold thickness (in mm) to estimate body fat.', placeholder="Enter skin thickness (mm)")
    
    with col2:
        Insulin = st.number_input('Insulin Level', min_value=0, max_value=1000, value=None, help='The amount of insulin hormone in the blood (mu U/ml).', placeholder="Enter insulin level (mu U/ml)")
    
    with col3:
        BMI = st.number_input('BMI value', min_value=0.0, max_value=60.0, value=None, help='Body Mass Index calculated from height and weight.', placeholder="Enter BMI value")
    
    with col1:
        Pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=None, help='The number of times the person has been pregnant.', placeholder="Enter number of pregnancies")
    
    with col2:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value', format="%.3f", value=None, help='Genetic risk score based on family history of diabetes.', placeholder="Enter pedigree function value")


    diab_diagnosis2 = ''
    
    if st.button('Diabetes Test Result'):
        inputs = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]

        if any(v is None for v in inputs):
            st.warning("⚠️ Please enter all the values before prediction.")
        else:
            diab_prediction2 = diabetes_model2.predict([inputs])
            
            if diab_prediction2[0] == 1:
                    st.markdown("""
                        <div style='padding: 1rem; background-color: #880000; border-left: 5px solid red; font-weight: bold;'>
                            🔴 Positive.
                        </div>
                    """, unsafe_allow_html=True)
            else:
                    st.markdown("""
                        <div style='padding: 1rem; background-color: #006600; border-left: 5px solid green; font-weight: bold;'>
                            🟢 Negative.
                        </div>
                    """, unsafe_allow_html=True)

            # Visualization
            #df = pd.DataFrame({
               # 'Feature': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
                #'Value': [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
            #})
            #fig, ax = plt.subplots(figsize=(10, 6))
           # sns.barplot(x='Feature', y='Value', data=df, palette='viridis', ax=ax)
            #ax.set_title('Entered Diabetes Health Parameters')
           # ax.tick_params(axis='x', rotation=45)
            #st.pyplot(fig)

            #st.success(diab_diagnosis2)
    
    
    
    
# Diabetes Prediction Page
if (selected == 'Diabetes (Y/N)'):
    
    # page title
    st.title('Diabetes Prediction (with symptoms)')
    
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age', min_value=1, max_value=120, value=None)

    with col2:
        gender = st.selectbox('Gender', options=[0, 1], index=None,)

    with col3:
        polyuria = st.selectbox('Excess Urination (Polyuria)', options=[0, 1], index=None)

    with col1:
        polydipsia = st.selectbox('Excess Thirst (Polydipsia)', options=[0, 1], index=None)

    with col2:
        sudden_weight_loss = st.selectbox('Sudden Weight Loss', options=[0, 1], index=None)

    with col3:
        weakness = st.selectbox('Weakness', options=[0, 1], index=None)

    with col1:
        polyphagia = st.selectbox('Excess Hunger (Polyphagia)', options=[0, 1], index=None)

    with col2:
        genital_thrush = st.selectbox('Yeast Infection (Genital Thrush)', options=[0, 1], index=None)

    with col3:
        visual_blurring = st.selectbox('Blurred Vision', options=[0, 1], index=None)

    with col1:
        itching = st.selectbox('Itching', options=[0, 1], index=None)

    with col2:
        irritability = st.selectbox('Irritability', options=[0, 1], index=None)

    with col3:
        delayed_healing = st.selectbox('Delayed Healing of Wound', options=[0, 1], index=None)

    with col1:
        partial_paresis = st.selectbox('Muscle Weakening (Partial Paresis)', options=[0, 1], index=None)

    with col2:
        muscle_stiffness = st.selectbox('Episode of Muscle Stiffness', options=[0, 1], index=None)

    with col3:
        alopecia = st.selectbox('Hair Loss (Alopecia)', options=[0, 1], index=None)

    with col1:
        obesity = st.selectbox('Obesity', options=[0, 1], index=None)

         
    # code for Prediction
    diab_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        input_data = [
            age, gender, polyuria, polydipsia, sudden_weight_loss, weakness,
            polyphagia, genital_thrush, visual_blurring, itching,
            irritability, delayed_healing, partial_paresis,
            muscle_stiffness, alopecia, obesity
        ]
        
        diab_prediction = diabetes_model1.predict([input_data])

        if diab_prediction[0] == 1:
                    st.markdown("""
                        <div style='padding: 1rem; background-color: #880000; border-left: 5px solid red; font-weight: bold;'>
                            🔴 Positive.
                        </div>
                    """, unsafe_allow_html=True)
        else:
                    st.markdown("""
                        <div style='padding: 1rem; background-color: #006600; border-left: 5px solid green; font-weight: bold;'>
                            🟢 Negative.
                        </div>
                    """, unsafe_allow_html=True)
        
        st.success(diab_diagnosis)






# Heart Disease Prediction Page

default_values = {
    "trestbps": 0,
    "chol": 0,
    "thalach": 0,
    "oldpeak": 0.0,
}

if (selected == 'Heart Disease'):

    st.sidebar.header("About the Input Terms")
    st.sidebar.markdown("""
    - **Age:** Age of the person in years. Heart disease risk typically increases with age.

    - **Gender:** Biological sex — Male or Female. Men generally have a higher risk at younger ages.

    - **Chest Pain Type:**
      - 0 = Typical Angina (chest pain due to reduced blood flow)
      - 1 = Atypical Angina (chest pain not related to heart)
      - 2 = Non-anginal Pain (non-heart chest pain)
      - 3 = Asymptomatic (no chest pain but heart disease present)

    - **Blood Pressure:** Resting blood pressure (in mm Hg). High blood pressure can damage arteries and the heart.

    - **Serum Cholesterol:** Total cholesterol in blood (mg/dl). Higher levels can build plaque in arteries.

    - **Fasting Blood Sugar:** Blood sugar after fasting.
      - More than 120 mg/dl suggests a higher risk of diabetes and heart disease.

    - **Resting ECG:**
      - 0 = Normal
      - 1 = ST-T wave abnormality (signs of heart strain)
      - 2 = Probable or definite left ventricular hypertrophy (enlarged heart muscle)

    - **Max Heart Rate Achieved:** Maximum heart rate reached during exercise. Lower than expected rates can indicate heart problems.

    - **Exercise Induced Angina:**
      - 0 = No chest pain during exercise
      - 1 = Chest pain occurs during exercise (possible restricted blood flow)

    - **ST Depression:** Decrease in ST segment of ECG after exercise. Indicates possible heart stress or lack of blood flow.

    - **Slope of ST Segment:**
      - 0 = Upsloping (normal, less likely heart disease)
      - 1 = Flat (possible heart disease)
      - 2 = Downsloping (highly suggestive of heart disease)

    - **Major Vessels Colored:** Number of major coronary arteries showing blockage via fluoroscopy (imaging technique).
      - 0–3 = Number of blocked vessels seen
      - 4 = Special case (all major vessels involved)

    - **Thalassemia:** Thalassemia refers to a type of inherited blood disorder affecting oxygen delivery in blood, which can stress the heart.
      - 1 = Normal blood flow
      - 2 = Fixed defect (permanent heart damage)
      - 3 = Reversible defect (temporary blockage that improves)
            
    """)
    
    # page title
    st.title('Heart Disease Prediction using ML 🫀')
    st.markdown("ENTER HEALTH PARAMETERS TO CHECK FOR HEART DISEASE:")
    

    
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", min_value=1, max_value=100, value=None, help="The age (in years) of the individual.")
    
    with col2:
        sex_input = st.selectbox('Gender', options=['Male', 'Female'], index=None, help="Select the biological gender of the person.")
        sex = 0 if sex_input == 'Male' else 1
    
    with col3:
        cp = st.selectbox('Chest Pain Types (0 to 3)', options=[0, 1, 2, 3], index=None, 
                          help="\n0 = Typical Angina,\n1 = Atypical Angina,\n2 = Non-anginal Pain,\n3 = Asymptomatic")
    
    with col1:
        trestbps = st.number_input('Blood Pressure', min_value=0, max_value=200, value=None, 
                                   help="Resting blood pressure in (mm Hg) on admission.", placeholder="Enter resting BP (mm Hg)")
    
    with col2:
        chol = st.number_input('Serum Cholestoral in mg/dl', min_value=0, max_value=600, value=None, 
                                help="Serum cholesterol measurement in (mg/dl).", placeholder="Enter cholesterol level")
    
    with col3:
        fbs_input = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=['No', 'Yes'], index=None, 
                                 help="Whether fasting blood sugar is more than 120 mg/dl.")
        fbs = 1 if fbs_input == 'Yes' else 0
    
    with col1:
        restecg = st.selectbox('Resting Electrocardiographic Results (0 to 2)', options=[0, 1, 2], index=None, 
                               help="ECG results:\n0 = Normal,\n1 = ST-T wave abnormality,\n2 = Left ventricular hypertrophy")
    
    with col2:
        thalach = st.number_input('Maximum Heart Rate Achieved', min_value=0, max_value=220, value=None, 
                                   help="Maximum heart rate achieved during exercise.", placeholder="Enter max heart rate")
    
    with col3:
        exang_input = st.selectbox('Exercise Induced Angina', options=['No', 'Yes'], index=None, 
                                   help="Exercise-induced chest pain.")
        exang = 1 if exang_input == 'Yes' else 0
    
    with col1:
        oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, value=None, 
                                  help="ST depression induced by exercise relative to rest.", placeholder="Enter ST depression value")
    
    with col2:
        slope = st.selectbox('Slope of the Peak Exercise ST Segment (0 to 2)', options=[0, 1, 2], index=None, 
                             help="\n0 = Upsloping,\n1 = Flat,\n2 = Downsloping")
    
    with col3:
        ca = st.selectbox('Major Vessels Colored by Fluoroscopy (0 to 4)', options=[0, 1, 2, 3, 4], index=None, 
                          help="Number of major vessels observed")
    
    with col1:
        thal = st.selectbox('Thalassemia (1 to 3)', options=[1, 2, 3], index=None, 
                            help="Type of blood disorder:\n1 = Normal,\n2 = Fixed Defect,\n3 = Reversible Defect")

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction
    if st.button('Heart Disease Test Result'):
        if (trestbps == default_values["trestbps"] or
            chol == default_values["chol"] or
            thalach == default_values["thalach"] or
            oldpeak == default_values["oldpeak"] or
            age is None or sex_input is None or cp is None or
            fbs_input is None or restecg is None or
            exang_input is None or slope is None or
            ca is None or thal is None):

            st.warning("⚠️ Please fill in all the values before making a prediction.")
        else:
            heart_prediction = heart_disease_model.predict([[int(age),int(sex),int(cp),int(trestbps),int(chol), int(fbs), 
                                                         int(restecg), int(thalach),int(exang),float(oldpeak),int(slope),
                                                         int(ca),int(thal)]])                          

            if (heart_prediction[0] == 1):
                    st.markdown("""
                        <div style='padding: 1rem; background-color: #880000; border-left: 5px solid red; font-weight: bold;'>
                            🔴 Positive.
                        </div>
                    """, unsafe_allow_html=True)
            else:
                    st.markdown("""
                        <div style='padding: 1rem; background-color: #006600; border-left: 5px solid green; font-weight: bold;'>
                            🟢 Negative.
                        </div>
                    """, unsafe_allow_html=True)

            st.success(heart_diagnosis)

            
        
  # Parkinson's Prediction Page
if (selected == "Parkinson's Disease"):
    
    # page title
    st.title("Parkinson's Disease Prediction using ML 🧠")
    st.markdown("ENTER HEALTH PARAMETERS TO CHECK FOR PARKINSONS:")
    
    st.sidebar.header("About the Input Terms")
    st.sidebar.markdown("""
                        **Note:**  
                        *MDVP* **(Multi-Dimensional Voice Program)** *features are acoustic measurements extracted from voice signals to detect subtle speech impairments, often used in Parkinson's disease analysis.*
                        """)
    st.sidebar.markdown("""
    - **MDVP:Fo (Hz):**  
      The average fundamental frequency (pitch) of the voice. Parkinson’s patients often have a lower or unstable pitch.

    - **MDVP:Fhi (Hz):**  
      The highest pitch frequency recorded. Abnormally high values could indicate uncontrolled vocal tremors.

    - **MDVP:Flo (Hz):**  
      The lowest pitch frequency recorded. Helps detect monotonic (flat) voice tone, a common Parkinson’s symptom.

    - **MDVP:Jitter (%):**  
      Measures cycle-to-cycle variations in voice frequency. Higher jitter means more instability or roughness in voice.

    - **MDVP:Jitter (Abs):**  
      Absolute measure of small period-to-period pitch fluctuations, giving another view of vocal irregularity.

    - **MDVP:RAP:**  
      Relative Average Perturbation — captures short-term, three-period frequency changes to spot subtle voice shakiness.

    - **MDVP:PPQ:**  
      Five-point Period Perturbation Quotient — a smoothed version of RAP that reduces measurement noise.

    - **Jitter:DDP:**  
      Difference of Differences of Periods — highlights micro-variations in vocal pitch cycles.

    - **MDVP:Shimmer:**  
      Measures changes in loudness across voice cycles. Larger shimmer values indicate breathy or weak voice.

    - **MDVP:Shimmer (dB):**  
      Shimmer measured in decibels (loudness units). Sensitive to subtle loudness instability in the voice.

    - **Shimmer:APQ3:**  
      Average Perturbation Quotient over 3 voice periods — detects very short-term loudness fluctuations.

    - **Shimmer:APQ5:**  
      Similar to APQ3 but averaged over 5 periods — giving a slightly more stable shimmer measure.

    - **MDVP:APQ:**  
      General amplitude perturbation quotient — summarizes overall voice amplitude instability.

    - **Shimmer:DDA:**  
      Measures differences of the amplitude between three consecutive cycles — useful for detecting quivering voices.

    - **NHR (Noise-to-Harmonics Ratio):**  
      Ratio of noise compared to the tonal part of voice. Higher NHR indicates a noisier, less clear voice.

    - **HNR (Harmonics-to-Noise Ratio):**  
      Ratio of clean vocal sound (harmonics) to noise. Lower HNR points to possible vocal cord dysfunction.

    - **RPDE (Recurrence Period Density Entropy):**  
      Measures irregularities and unpredictability in the voice signal. Higher RPDE indicates disordered phonation.

    - **D2:**  
      Correlation dimension that captures the complexity of the voice waveform. Parkinson's often increases signal complexity.

    - **DFA (Detrended Fluctuation Analysis):**  
      Analyzes the fractal-like structure of voice signals. Irregular scaling behaviors are common in Parkinson’s speech.

    - **PPE (Pitch Period Entropy):**  
      Quantifies how unpredictable the pitch period is. A more erratic (higher PPE) pitch is typical in Parkinson’s.

    - **spread1:**  
      Measures the variation between the fundamental frequency and the average voice spectrum. More negative values mean instability.

    - **spread2:**  
      Another spread measure showing the second type of voice frequency variation. High absolute values imply greater vocal disorder.
    """)

    
    col1, col2, col3, col4 = st.columns(4)  

    with col1:
        fo = st.number_input('MDVP:Fo (Hz)', format="%.6f", value=None, placeholder="Enter average vocal pitch", help="Average fundamental frequency of the voice. Lower or unstable values can signal Parkinson’s.")
    
    with col2:
        fhi = st.number_input('MDVP:Fhi (Hz)', format="%.6f", value=None, placeholder="Enter highest vocal frequency", help="Highest frequency in the voice recording, detects uncontrolled tremors.")
    
    with col3:
        flo = st.number_input('MDVP:Flo (Hz)', format="%.6f", value=None, placeholder="Enter lowest vocal frequency", help="Lowest vocal frequency, often reduced in Parkinson’s.")
    
    with col4:
        Jitter_percent = st.number_input('MDVP:Jitter (%)', format="%.6f", value=None, placeholder="Enter jitter percentage", help="Measures frequency instability. Higher values show vocal roughness.")
    
    with col1:
        Jitter_Abs = st.number_input('MDVP:Jitter (Abs)', format="%.6f", value=None, placeholder="Enter absolute jitter", help="Absolute cycle-to-cycle pitch variations, another indicator of instability.")
    
    with col2:
        RAP = st.number_input('MDVP:RAP', format="%.6f", value=None, placeholder="Enter RAP value", help="Relative Average Perturbation, measures short-term pitch changes.")
    
    with col3:
        PPQ = st.number_input('MDVP:PPQ', format="%.6f", value=None, placeholder="Enter PPQ value", help="Period Perturbation Quotient over 5 periods, smoother version of RAP.")
    
    with col4:
        DDP = st.number_input('Jitter:DDP', format="%.6f", value=None, placeholder="Enter DDP value", help="Highlights very small variations between pitch cycles.")
    
    with col1:
        Shimmer = st.number_input('MDVP:Shimmer', format="%.6f", value=None, placeholder="Enter shimmer value", help="Measures loudness instability between vocal cycles.")
    
    with col2:
        Shimmer_dB = st.number_input('MDVP:Shimmer (dB)', format="%.6f", value=None, placeholder="Enter shimmer (dB)", help="Amplitude instability measured in decibels.")
    
    with col3:
        APQ3 = st.number_input('Shimmer:APQ3', format="%.6f", value=None, placeholder="Enter APQ3 value", help="Short-term amplitude perturbations over 3 periods.")
    
    with col4:
        APQ5 = st.number_input('Shimmer:APQ5', format="%.6f", value=None, placeholder="Enter APQ5 value", help="Amplitude perturbation over 5 periods for stability.")
    
    with col1:
        APQ = st.number_input('MDVP:APQ', format="%.6f", value=None, placeholder="Enter APQ value", help="General average of amplitude variations across the signal.")
    
    with col2:
        DDA = st.number_input('Shimmer:DDA', format="%.6f", value=None, placeholder="Enter DDA value", help="Average difference between amplitudes of consecutive cycles.")
    
    with col3:
        NHR = st.number_input('NHR', format="%.6f", value=None, placeholder="Enter NHR value", help="Noise-to-Harmonics Ratio. Higher values mean more noise in the voice.")
    
    with col4:
        HNR = st.number_input('HNR', format="%.6f", value=None, placeholder="Enter HNR value", help="Harmonics-to-Noise Ratio. Lower values suggest vocal cord issues.")
    
    with col1:
        RPDE = st.number_input('RPDE', format="%.6f", value=None, placeholder="Enter RPDE value", help="Measures irregularity and unpredictability of the voice pattern.")
    
    with col2:
        DFA = st.number_input('DFA', format="%.6f", value=None, placeholder="Enter DFA value", help="Detects fractal scaling behavior in voice signals.")
    
    with col3:
        spread1 = st.number_input('spread1', format="%.6f", value=None, placeholder="Enter spread1 value", help="Difference between average spectrum and fundamental frequency.")
    
    with col4:
        spread2 = st.number_input('spread2', format="%.6f", value=None, placeholder="Enter spread2 value", help="Another metric showing spread and variation of frequencies.")
    
    with col1:
        D2 = st.number_input('D2', format="%.6f", value=None, placeholder="Enter D2 value", help="Correlation dimension indicating complexity of the voice signal.")
    
    with col2:
        PPE = st.number_input('PPE', format="%.6f", value=None, placeholder="Enter PPE value", help="Pitch Period Entropy — measures how unpredictable the voice pitch is.")



    # code for Prediction
    parkinsons_diagnosis = ''

    # creating a button for Prediction    
    if st.button("Parkinson's Disease Test Result"):
        inputs = [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB,
                  APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

        if any(val is None or np.isnan(val) for val in inputs):
            st.warning("⚠️ Please fill in all the values before making a prediction.")
        else:
            parkinsons_prediction = parkinsons_model.predict([inputs])                           

            if parkinsons_prediction[0] == 1:
                st.markdown('<div style="padding: 1rem; background-color: #ffe6e6; color: #b30000; border-radius: 0.5rem; font-weight: bold;">🔴 The person <b>has</b> Parkinson\'s disease.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="padding: 1rem; background-color: #e6ffe6; color: #006600; border-radius: 0.5rem; font-weight: bold;">🟢 The person <b>does not have</b> Parkinson\'s disease.</div>', unsafe_allow_html=True)
                
                st.success(parkinsons_diagnosis)



#BACKUP ([[float(fo), float(fhi), float(flo), float(Jitter_percent), float(Jitter_Abs),
#                                                   float(RAP), float(PPQ), float(DDP), float(Shimmer), float(Shimmer_dB),
 #                                                  float(APQ3), float(APQ5), float(APQ), float(DDA), float(NHR), float(HNR),
  #                                                 float(RPDE), float(DFA), float(spread1), float(spread2), float(D2), float(PPE)]]) 



#Breast Cancer Prediction Page:

if(selected == "Breast Cancer"):
    
    #page title
    st.title("Breast Cancer Prediction using ML 🎗")
    st.markdown("ENTER HEALTH PARAMETERS TO CHECK FOR BREAST CANCER:")
    
    st.sidebar.header("About the Input Terms")
    st.sidebar.markdown("""
    - **Mean Radius**  
      The average distance from the center of the tumor to its boundary. Larger values often suggest larger tumors.
    
    - **Mean Texture**  
      The standard deviation of gray-scale pixel values in the image, reflecting surface variation.
    
    - **Mean Perimeter**  
      The average length around the tumor boundary. Longer perimeters can indicate irregular tumors.
    
    - **Mean Area**  
      The average size of the tumor (in pixel count). Larger areas are often associated with malignancy.
    
    - **Mean Smoothness**  
      Measurement of how smooth the tumor boundary is. Less smooth (more jagged) edges may suggest malignancy.
    
    - **Mean Compactness**  
      A combination of perimeter and area to describe how tightly packed the tumor shape is.
    
    - **Mean Concavity**  
      Measures the severity of indentations or concave portions in the tumor boundary.
    
    - **Mean Concave Points**  
      The number of concave sections (small indentations) in the tumor's perimeter.
    
    - **Mean Symmetry**  
      Describes how symmetrical the tumor is. Malignant tumors tend to be less symmetrical.
    
    - **Mean Fractal Dimension**  
      Reflects the complexity of the tumor boundary — higher values mean more irregular edges.
    
    - **Radius Error**  
      Variation in radius measurements across multiple observations.
    
    - **Texture Error**  
      Variation in the surface texture of the tumor.
    
    - **Perimeter Error**  
      Variation in the measured perimeter lengths.
    
    - **Area Error**  
      Variation in the measured areas of the tumor.
    
    - **Smoothness Error**  
      Variation in smoothness across the tumor boundary.
    
    - **Compactness Error**  
      Variation in the compactness across different measurements.
    
    - **Concavity Error**  
      Variation in the depth of indentations across the tumor boundary.
    
    - **Concave Points Error**  
      Variation in the number of concave points.
    
    - **Symmetry Error**  
      Variation in symmetry across different parts of the tumor.
    
    - **Fractal Dimension Error**  
      Variation in the complexity of the tumor boundary.
    
    - **Worst Radius**  
      The largest observed radius among different tumor regions.
    
    - **Worst Texture**  
      The most extreme texture variation observed.
    
    - **Worst Perimeter**  
      The largest observed perimeter.
    
    - **Worst Area**  
      The largest observed area.
    
    - **Worst Smoothness**  
      The least smooth boundary observed.
    
    - **Worst Compactness**  
      The highest compactness observed (more irregular shape).
    
    - **Worst Concavity**  
      The most severe concave portions observed.
    
    - **Worst Concave Points**  
      The highest number of concave points observed.
    
    - **Worst Symmetry**  
      The lowest symmetry observed.
    
    - **Worst Fractal Dimension**  
      The most complex (roughest) tumor boundary observed.
    """)

    
# getting the input data from the user

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        mean_radius = st.number_input('Mean Radius', format='%.6f', value=None, placeholder="Enter Mean Radius value", help="Average distance from center to boundary of tumor cells.")
        mean_smoothness = st.number_input('Mean Smoothness', format='%.6f', value=None, placeholder="Enter Mean Smoothness value", help="Variation in radius lengths. Higher smoothness suggests more uniform shapes.")
        mean_symmetry = st.number_input('Mean Symmetry', format='%.6f', value=None, placeholder="Enter Mean Symmetry value", help="Symmetry of the cell nucleus. Less symmetry may indicate malignancy.")
        perimeter_error = st.number_input('Perimeter Error', format='%.6f', value=None, placeholder="Enter Perimeter Error value", help="Variation in perimeter measurements across cells.")
    
    with col2:
        mean_texture = st.number_input('Mean Texture', format='%.6f', value=None, placeholder="Enter Mean Texture value", help="Standard deviation of gray-scale pixel values. Texture variation indicates structural irregularities.")
        mean_compactness = st.number_input('Mean Compactness', format='%.6f', value=None, placeholder="Enter Mean Compactness value", help="How closely packed the cells are.")
        mean_fractal_dimension = st.number_input('Mean Fractal Dimension', format='%.6f', value=None, placeholder="Enter Mean Fractal Dimension value", help="A measure of complexity in the tumor shape.")
        area_error = st.number_input('Area Error', format='%.6f', value=None, placeholder="Enter Area Error value", help="Variability in the area across cells.")
    
    with col3:
        mean_perimeter = st.number_input('Mean Perimeter', format='%.6f', value=None, placeholder="Enter Mean Perimeter value", help="Mean perimeter of the cell nuclei.")
        mean_concavity = st.number_input('Mean Concavity', format='%.6f', value=None, placeholder="Enter Mean Concavity value", help="Severity of concave (hollowed) portions of the tumor contour.")
        radius_error = st.number_input('Radius Error', format='%.6f', value=None, placeholder="Enter Radius Error value", help="Variation in radius measurement across cells.")
        smoothness_error = st.number_input('Smoothness Error', format='%.6f', value=None, placeholder="Enter Smoothness Error value", help="Variation in smoothness across the tumor surface.")
    
    with col4:
        mean_area = st.number_input('Mean Area', format='%.6f', value=None, placeholder="Enter Mean Area value", help="Average size of the tumor nuclei.")
        mean_concave_points = st.number_input('Mean Concave Points', format='%.6f', value=None, placeholder="Enter Mean Concave Points value", help="Average number of concave points on tumor surface.")
        texture_error = st.number_input('Texture Error', format='%.6f', value=None, placeholder="Enter Texture Error value", help="Variation in texture measurements across cells.")
        compactness_error = st.number_input('Compactness Error', format='%.6f', value=None, placeholder="Enter Compactness Error value", help="Variation in compactness across the tumor boundary.")
    
    with col1:
        concavity_error = st.number_input('Concavity Error', format='%.6f', value=None, placeholder="Enter Concavity Error value", help="Variation in concavity measurements across cells.")
        worst_radius = st.number_input('Worst Radius', format='%.6f', value=None, placeholder="Enter Worst Radius value", help="Largest radius measurement observed.")
        worst_smoothness = st.number_input('Worst Smoothness', format='%.6f', value=None, placeholder="Enter Worst Smoothness value", help="Highest smoothness value observed.")
        worst_symmetry = st.number_input('Worst Symmetry', format='%.6f', value=None, placeholder="Enter Worst Symmetry value", help="Worst symmetry observed among all nuclei.")
    
    with col2:
        concave_points_error = st.number_input('Concave Points Error', format='%.6f', value=None, placeholder="Enter Concave Points Error value", help="Variation in the number of concave points across tumors.")
        worst_texture = st.number_input('Worst Texture', format='%.6f', value=None, placeholder="Enter Worst Texture value", help="Maximum texture irregularity observed.")
        worst_compactness = st.number_input('Worst Compactness', format='%.6f', value=None, placeholder="Enter Worst Compactness value", help="Maximum compactness observed.")
        worst_fractal_dimension = st.number_input('Worst Fractal Dimension', format='%.6f', value=None, placeholder="Enter Worst Fractal Dimension value", help="Highest observed complexity of tumor shape.")
    
    with col3:
        symmetry_error = st.number_input('Symmetry Error', format='%.6f', value=None, placeholder="Enter Symmetry Error value", help="Variation in symmetry across cells.")
        worst_perimeter = st.number_input('Worst Perimeter', format='%.6f', value=None, placeholder="Enter Worst Perimeter value", help="Maximum perimeter measurement observed.")
        worst_concavity = st.number_input('Worst Concavity', format='%.6f', value=None, placeholder="Enter Worst Concavity value", help="Largest concavity observed across tumor boundaries.")
    
    with col4:
        fractal_dimension_error = st.number_input('Fractal Dimension Error', format='%.6f', value=None, placeholder="Enter Fractal Dimension Error value", help="Variation in fractal dimension measurements.")
        worst_area = st.number_input('Worst Area', format='%.6f', value=None, placeholder="Enter Worst Area value", help="Largest area measurement observed.")
        worst_concave_points = st.number_input('Worst Concave Points', format='%.6f', value=None, placeholder="Enter Worst Concave Points value", help="Maximum number of concave points observed.")

    
    #code for Prediction
    breast_cancer_check = " "

    if st.button("Breast Cancer Test Result"):
        try:
            inputs = [
                mean_radius, mean_texture, mean_perimeter, mean_area,
                mean_smoothness, mean_compactness, mean_concavity, mean_concave_points,
                mean_symmetry, mean_fractal_dimension, radius_error, texture_error,
                perimeter_error, area_error, smoothness_error, compactness_error,
                concavity_error, concave_points_error, symmetry_error, fractal_dimension_error,
                worst_radius, worst_texture, worst_perimeter, worst_area,
                worst_smoothness, worst_compactness, worst_concavity, worst_concave_points,
                worst_symmetry, worst_fractal_dimension
            ]

            if any(v is None for v in inputs):
                st.warning("⚠️ Please enter all the values before making a prediction.")
            else:
                breast_cancer_prediction = breastcancer_model.predict([inputs])

                if breast_cancer_prediction[0] == 0:
                    st.markdown("""
                        <div style='padding: 1rem; background-color: #006600; border-left: 5px solid green; font-weight: bold;'>
                            🟢 Negative.
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div style='padding: 1rem; background-color: #880000; border-left: 5px solid red; font-weight: bold;'>
                            🔴 Positive.
                        </div>
                    """, unsafe_allow_html=True)

        except ValueError:
            st.warning("⚠️ Please fill in all the fields before making a prediction.")
                      
            st.success(breast_cancer_check) 




#Lung Cancer Prediction Page:

if(selected == "Lung Cancer"):
    
    #page title
    st.title("Lung Cancer Prediction using ML 🫁")
    st.markdown("ENTER HEALTH PARAMETERS TO CHECK FOR LUNG CANCER:")
    st.sidebar.header("About the Input Terms")
    st.sidebar.markdown("""       
    - **Age**  
      The person's age in years. Increasing age is a significant risk factor for lung cancer, with most cases diagnosed in individuals over 50.
    
    - **Gender**  
      Biological sex of the individual. Men have historically had a higher risk of lung cancer, although rates among women are rising.
    
    - **Smoking**  
      Indicates whether the person has a history of smoking tobacco. Smoking is the number one risk factor, responsible for about 85% of lung cancer cases.
    
    - **Yellow Fingers**  
      Yellow discoloration of fingers, often caused by long-term smoking, especially heavy smokers. It can hint toward prolonged exposure to tobacco tar and nicotine.
    
    - **Anxiety**  
      Frequent or chronic anxiety symptoms. Anxiety can lead to health-damaging habits like smoking and poor lifestyle choices, indirectly raising lung cancer risk.
    
    - **Peer Pressure**  
      Refers to smoking or alcohol consumption initiated due to influence from friends, colleagues, or social groups — especially common among teenagers and young adults.
    
    - **Chronic Disease**  
      Presence of existing chronic health conditions (such as COPD, diabetes, or heart disease). Chronic diseases weaken the body’s resilience and may compound lung cancer risks.
    
    - **Fatigue**  
      Unusual or persistent tiredness that does not improve with rest. Fatigue can be an early symptom of lung cancer or an effect of the body's immune response.
    
    - **Allergy**  
      Having allergies (e.g., to dust, pollen). Although not directly linked to cancer, frequent respiratory allergies may signal chronic irritation or inflammation of airways.
    
    - **Wheezing**  
      A whistling or squeaky sound while breathing, usually due to narrowed or obstructed airways. It can occur if a tumor is blocking air passages.
    
    - **Alcohol Consuming**  
      Regular intake of alcohol. While not the strongest factor alone, combined alcohol and smoking usage significantly multiplies the risk for lung cancer.
    
    - **Coughing**  
      Persistent coughing that doesn't go away. A chronic cough is a hallmark symptom in many lung cancer patients, especially if accompanied by blood (hemoptysis).
    
    - **Shortness of Breath**  
      Experiencing difficulty or discomfort when breathing. Lung cancer can block major airways, reduce lung capacity, or cause fluid buildup around the lungs.
    
    - **Swallowing Difficulty**  
      Medical term: Dysphagia. Difficulty in swallowing can occur if a lung tumor presses against the esophagus or nerves controlling swallowing muscles.
    
    - **Chest Pain**  
      Pain, tightness, or pressure in the chest, especially when taking a deep breath, coughing, or laughing. Tumors pressing against the chest wall or spreading to ribs can cause this.
    """)

    
# getting the input data from the user
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        Age = st.slider('Age', min_value=1, max_value=100, value=None, help="Age of the patient. Higher age increases lung cancer risk.")
        Anxiety = st.selectbox('Anxiety', options=['No', 'Yes'], index=None, placeholder="Select Anxiety", help="Whether the patient frequently experiences anxiety.")
        Allergy = st.selectbox('Allergy', options=['No', 'Yes'], index=None, placeholder="Select Allergy", help="Presence of any allergies which may impact respiratory health.")
        ShortnessOfBreath = st.selectbox('Shortness of Breath', options=['No', 'Yes'], index=None, placeholder="Select Shortness of Breath", help="Difficulty in breathing, common symptom of lung issues.")
    
    with col2:
        Gender = st.selectbox('Gender', options=['Male', 'Female'], index=None, placeholder="Select Gender", help="Biological sex of the patient.")
        PeerPressure = st.selectbox('Peer Pressure', options=['No', 'Yes'], index=None, placeholder="Select Peer Pressure", help="Whether the patient was pressured by peers (e.g., into smoking).")
        Wheezing = st.selectbox('Wheezing', options=['No', 'Yes'], index=None, placeholder="Select Wheezing", help="High-pitched whistling sound while breathing.")
        SwallowingDifficulty = st.selectbox('Swallowing Difficulty', options=['No', 'Yes'], index=None, placeholder="Select Swallowing Difficulty", help="Difficulty in swallowing food or liquids.")
    
    with col3:
        Smoking = st.selectbox('Smoking', options=['No', 'Yes'], index=None, placeholder="Select Smoking", help="Whether the patient smokes tobacco.")
        ChronicDisease = st.selectbox('Chronic Disease', options=['No', 'Yes'], index=None, placeholder="Select Chronic Disease", help="Presence of any long-term diseases like asthma or COPD.")
        AlcoholConsuming = st.selectbox('Alcohol Consumption', options=['No', 'Yes'], index=None, placeholder="Select Alcohol Consuming", help="Whether the patient consumes alcohol regularly.")
        ChestPain = st.selectbox('Chest Pain', options=['No', 'Yes'], index=None, placeholder="Select Chest Pain", help="Experiencing pain or discomfort in the chest area.")
    
    with col4:
        YellowFingers = st.selectbox('Yellow Fingers', options=['No', 'Yes'], index=None, placeholder="Select Yellow Fingers", help="Yellow discoloration of fingers, often associated with smoking.")
        Fatigue = st.selectbox('Fatigue', options=['No', 'Yes'], index=None, placeholder="Select Fatigue", help="Constant tiredness or lack of energy.")
        Coughing = st.selectbox('Coughing', options=['No', 'Yes'], index=None, placeholder="Select Coughing", help="Persistent coughing, a key symptom of lung cancer.")



# code for Prediction
    LungCancer_result = ""

    if st.button("Lung Cancer Test Result"):
        binary_map = lambda x: 1 if str(x).strip().lower() == 'yes' else 0
        gender_map = lambda x: 1 if x == 'Female' else 0
        try:
            if None in [Gender, Age, Smoking, YellowFingers, Anxiety, PeerPressure, ChronicDisease, Fatigue, Allergy, Wheezing, AlcoholConsuming, Coughing, ShortnessOfBreath, SwallowingDifficulty, ChestPain]:
                st.warning("⚠️ Please enter all values before making a prediction.")
            else:
                inputs = [
                    gender_map(Gender), Age,
                    binary_map(Smoking), binary_map(YellowFingers), binary_map(Anxiety),
                    binary_map(PeerPressure), binary_map(ChronicDisease), binary_map(Fatigue),
                    binary_map(Allergy), binary_map(Wheezing), binary_map(AlcoholConsuming),
                    binary_map(Coughing), binary_map(ShortnessOfBreath),
                    binary_map(SwallowingDifficulty), binary_map(ChestPain)
                ]

                lung_prediction = lungcancer_model.predict([inputs])

                if lung_prediction[0] == 1:
                    st.markdown("""
                        <div style='padding: 1rem; background-color: #880000; border-left: 5px solid red; font-weight: bold;'>
                            🔴 Positive.
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div style='padding: 1rem; background-color: #006600; border-left: 5px solid green; font-weight: bold;'>
                            🟢 Negative.
                        </div>
                    """, unsafe_allow_html=True)
                    
        except Exception as e:
            st.error(f"Error in prediction: {e}")
 
            st.success(LungCancer_result)


#Chatbot
if (selected == 'Healthcare Chatbot'):

    # Define the GPT API endpoint
    API_ENDPOINT = "https://api.pawan.krd/v1/completions"

    # Define your OpenAI API key
    API_KEY = "pk-eAWvHQfEkRiWCiCNMDLnOGdfpqgxCQzbPtPrBvtdbmHmFktW"

    # Function to interact with the GPT API
    def generate_response(prompt):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
        data = {
            "prompt": prompt,
            "max_tokens": 1000  # Adjust the max tokens as needed
        }
        response = requests.post(API_ENDPOINT, headers=headers, json=data)
        
        if response.ok:
            response_json = response.json()
            
            if "choices" in response_json and len(response_json["choices"]) > 0:
                return response_json["choices"][0]["text"].strip()
            else:
                return "No response received from the chatbot."
        else:
            print("Error accessing the chatbot API. Status code:", response.status_code)
            return "An error occurred while accessing the chatbot. Please try again later."


    # Function to simulate bot typing effect
    def simulate_typing():
        st.text("Bot is typing...")

    # Main code
    st.title("Healthcare Chatbot")
    st.markdown("Welcome to the Healthcare Chatbot! How can I assist you today?")

    # User input
    user_input = st.text_input("User:")

    # Generate bot response
    if user_input:
        bot_response = generate_response(user_input)
        bot_response_html = f'<div style="overflow-wrap: break-word; height: auto; padding: 10px;">{bot_response}</div>'
        st.markdown(bot_response_html, unsafe_allow_html=True)



#Liver Prediction Page:

if(selected == "Liver"):
    
    #page title
    st.title("Liver Disease Prediction using ML")
    st.markdown("ENTER HEALTH PARAMETERS TO CHECK FOR LIVER DISEASE:")

# Set 3 columns layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider("Age", min_value=1, max_value=100)
        direct_bilirubin = st.number_input("Direct Bilirubin (mg/dL)", min_value=0.0, max_value=20.0, value=None, placeholder="Enter Direct Bilirubin")
        sgot = st.number_input("Aspartate Aminotransferase (SGOT) (IU/L)", min_value=0.0, max_value=5000.0, value=None, placeholder="Enter SGOT")
        ag_ratio = st.number_input("Albumin and Globulin Ratio", min_value=0.0, max_value=3.0, value=None, placeholder="Enter A/G Ratio")

    
    with col2:
        gender = st.selectbox("Gender", ['Male', 'Female'], index=None, placeholder="Select Gender")
        alk_phosphate = st.number_input("Alkaline Phosphotase (IU/L)", min_value=0.0, max_value=2500.0, value=None, placeholder="Enter Alkaline Phosphotase")  
        total_proteins = st.number_input("Total Proteins (g/dL)", min_value=0.0, max_value=10.0, value=None, placeholder="Enter Total Proteins")
        

    with col3:
        total_bilirubin = st.number_input("Total Bilirubin (mg/dL)", min_value=0.0, max_value=80.0, value=None, placeholder="Enter Total Bilirubin")        
        sgpt = st.number_input("Alamine Aminotransferase (SGPT) (IU/L)", min_value=0.0, max_value=3000.0, value=None, placeholder="Enter SGPT")        
        albumin = st.number_input("Albumin (g/dL)", min_value=0.0, max_value=6.0, value=None, placeholder="Enter Albumin")


   
    # Encode gender
    gender_encoded = 1 if gender == "Female" else 0
    
    Liver_result = ""
    
    # Prediction
    if st.button("Liver Disease Test Result"):
        if None in [age, total_bilirubin, direct_bilirubin, alk_phosphate, sgpt, sgot, total_proteins, albumin, ag_ratio]:
            st.warning("⚠️ Please fill in all the fields before prediction.")
        else:
            input_data = np.array([[age, gender_encoded, total_bilirubin, direct_bilirubin,
                                    alk_phosphate, sgpt, sgot, total_proteins, albumin, ag_ratio]])
            
            # Separate Age and Gender
            age_gender = input_data[:, :2]  # First 2 columns
            
    
            # Predict
            prediction = liver_model.predict(input_data)
    
            if prediction[0] == 1:
                st.error("🔴 Positive.")
            elif prediction[2] == 2:            
                st.success("🟢 Negative.")
            else:
                st.warning("⚠️ Unexpected prediction output.")
                
                st.success(Liver_result)        



# Background
def set_bg_from_url(url, opacity=1):
    footer = """
    <footer>
        <div style='visibility: visible;margin-top:2rem;justify-content:center;display:flex;'>
            <p style="font-size:1.5rem;">
                "Made by <b>ARNAV TANWAR</b>"                            
            </p>
        </div>
    </footer>
"""
    st.markdown(footer, unsafe_allow_html=True)
    
    
    # background image using HTML
    st.markdown(
        f"""
        <style>
            body {{
                background: url('{url}') no-repeat center center fixed;
                background-size: cover;
                opacity: {opacity};
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

# background image from URL
set_bg_from_url("https://news.harvard.edu/wp-content/uploads/2023/04/AI-cardiologist-heart-diagnostics_1200x800.jpg", opacity=0.85)

