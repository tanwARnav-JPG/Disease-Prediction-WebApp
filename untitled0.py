

import pickle
import streamlit as st
from streamlit_option_menu import option_menu


# loading the saved models

diabetes_model = pickle.load(open('C:/Users/arnav/Desktop/Major Project/Major Project/Trained Model/diabetes_model1.sav', 'rb'))





# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Menu Driven Multiple Disease Prediction',
                          
                          ['Diabetes Prediction'],
                          icons=['activity','heart','person'],
                          default_index=0)
    
    
# Diabetes Prediction Page
if (selected == 'Diabetes Prediction'):
    
    # page title
    st.title('Diabetes Prediction using ML')
    
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
        
    with col2:
        gender = st.text_input('Gender(Male-0/Female-1)')
    
    with col3:
        polyuria = st.text_input('Excess Urination(Y-1/N-0)')
    
    with col1:
        polydipsia = st.text_input('Excess thirst or not(Y-1/N-0)')
    
    with col2:
        sudden_weight_loss = st.text_input('Sudden Weight Loss(Y-1/N-0)')
    
    with col3:
        weakness = st.text_input('Weakness(Y-1/N-0)')
    
    with col1:
        polyphagia = st.text_input('Episode of Excess Hunger(Y-1/N-0)')
        
    with col2:
        genital_thrush = st.text_input('Yeast Infection(Y-1/N-0)')
    
    with col3:
        visual_blurring = st.text_input('Blurred Vision(Y-1/N-0)')
    
    with col1:
        itching = st.text_input('Episode of Itching(Y-1/N-0)')
        
    with col2:
        irritability = st.text_input('Episode of Irritability(Y-1/N-0)')
    
    with col3:
        delayed_healing = st.text_input('Delayed Healing of Wound(Y-1/N-0)')
    
    with col1:
        partial_paresis = st.text_input('Episode of Muscle Weakening(Y-1/N-0)')
        
    with col2:
        muscle_stiffness = st.text_input('Episode of muscle stiffness(Y-1/N-0)')
    
    with col3:
        alopecia = st.text_input('Hair Loss(Y-1/N-0)')
    
    with col1:
        obesity = st.text_input('Obesity(Y-1/N-0)')
        
    
    
    
    # code for Prediction
    diab_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[age, gender, polyuria, polydipsia, sudden_weight_loss, weakness, polyphagia, genital_thrush,visual_blurring,
                                                   itching,irritability,delayed_healing,partial_paresis,muscle_stiffness,alopecia,obesity]])
        
        if (diab_prediction[0] == 1):
          diab_diagnosis = 'The person is diabetic'
        else:
          diab_diagnosis = 'The person is not diabetic'
        
    st.success(diab_diagnosis)




