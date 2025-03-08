
import pickle
import streamlit as st
from streamlit_option_menu import option_menu


# loading the saved models

diabetes_model = pickle.load(open('Trained Model/diabetes_model1.sav', 'rb'))

#diabetes_model = pickle.load(open('C:/Users/arnav/Desktop/Major Project/Major Project/Trained Model/diabetes_model1.sav', 'rb'))

heart_disease_model = pickle.load(open('Trained Model/heart_disease_model.sav', 'rb'))

#heart_disease_model = pickle.load(open('C:/Users/arnav/Desktop/Major Project/Major Project/Trained Model/heart_disease_model.sav','rb'))

parkinsons_model = pickle.load(open('C:/Users/arnav/Desktop/Major Project/Major Project/Trained Model/parkinsons_model.sav', 'rb'))

diabetes_model2 = pickle.load(open('C:/Users/arnav/Desktop/Major Project/Major Project/Trained Model/diabetes_model.sav', 'rb'))

cancer_model = pickle.load(open('C:/Users/arnav/Desktop/Major Project/Major Project/Trained Model/breast_cancer_model.sav', 'rb'))

lung_cancer = pickle.load(open('C:/Users/arnav/Desktop/Major Project/Major Project/Trained Model/lungcancer_model.sav', 'rb'))



# sidebar for navigation
with st.sidebar:
    '[Yes=1, No=0]'
    selected = option_menu('MENU-DRIVEN MULTIPLE DISEASE PREDICTION SYSTEM USING ML',
                          
                          ['Diabetes Prediction',
                           'Diabetes (Y/N)',
                           'Heart Disease Prediction',
                           'Parkinsons Disease Prediction',
                           'Breast Cancer Prediction',
                           'Lung Cancer Prediction (Y/N)'],
                          icons=['activity','clipboard2-pulse', 'suit-heart', 'p-square-fill', 'gender-female', 'lungs-fill'],
                          menu_icon=['hospital'],
                          default_index=0)
                          

# diabetes num Prediction Page
if (selected == 'Diabetes Prediction'):
    
    # page title
    st.title('Diabetes Prediction using ML')
    
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Age = st.text_input('Age of the Person')
        
    with col2:
        Glucose = st.text_input('Glucose Level')
    
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    
    with col2:
        Insulin = st.text_input('Insulin Level')
    
    with col3:
        BMI = st.text_input('BMI value')
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
            
    with col2:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    
    
    # code for Prediction
    diab_diagnosis2 = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diab_prediction2 = diabetes_model2.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        if (diab_prediction2[0] == 1):
          diab_diagnosis2 = 'The person is diabetic'
        else:
          diab_diagnosis2 = 'The person is not diabetic'
        
    st.success(diab_diagnosis2)
    
    
    
# Diabetes Prediction Page
if (selected == 'Diabetes (Y/N)'):
    
    # page title
    st.title('Diabetes Prediction (with symptoms)')
    
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
        
    with col2:
        gender = st.text_input('Gender (Male-0 / Female-1)')
    
    with col3:
        polyuria = st.text_input('Excess Urination (Polyuria)')
    
    with col1:
        polydipsia = st.text_input('Excess Thirst or not (Polydipsia)')
    
    with col2:
        sudden_weight_loss = st.text_input('Sudden Weight Loss')
    
    with col3:
        weakness = st.text_input('Weakness')
    
    with col1:
        polyphagia = st.text_input('Excess Hunger (Polyphagia)')
        
    with col2:
        genital_thrush = st.text_input('Yeast Infection (Genital Thrush)')
    
    with col3:
        visual_blurring = st.text_input('Blurred Vision')
    
    with col1:
        itching = st.text_input('Itching')
        
    with col2:
        irritability = st.text_input('Irritability')
    
    with col3:
        delayed_healing = st.text_input('Delayed Healing of Wound')
    
    with col1:
        partial_paresis = st.text_input('Muscle Weakening (Partial Paresis)')
        
    with col2:
        muscle_stiffness = st.text_input('Episode of Muscle Stiffness')
    
    with col3:
        alopecia = st.text_input('Hair Loss (Alopecia)')
    
    with col1:
        obesity = st.text_input('Obesity')
        
    
    
    
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




# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):
    
    # page title
    st.title('Heart Disease Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
        
    with col2:
        sex = st.text_input('Gender {Male-0 / Female-1}')
        
    with col3:
        cp = st.text_input('Chest Pain types')
        
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
        
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
        
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
        
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.text_input('Exercise Induced Angina')
        
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
        
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
        
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
        
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
        
     
     
    # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_disease_model.predict([[int(age),int(sex),int(cp),int(trestbps),int(chol), int(fbs), 
                                                         int(restecg), int(thalach),int(exang),float(oldpeak),int(slope),
                                                         int(ca),int(thal)]])                          
        
        if (heart_prediction[0] == 1):
          heart_diagnosis = 'The person is having heart disease'
        else:
          heart_diagnosis = 'The person does not have any heart disease'
        
    st.success(heart_diagnosis)
        
    
    

# Parkinson's Prediction Page
if (selected == "Parkinsons Disease Prediction"):
    
    # page title
    st.title("Parkinson's Disease Prediction using ML")
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.text_input('MDVP:Fo (Hz)')
        
    with col2:
        fhi = st.text_input('MDVP:Fhi (Hz)')
        
    with col3:
        flo = st.text_input('MDVP:Flo (Hz)')
        
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter (%)')
        
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter (Abs)')
        
    with col1:
        RAP = st.text_input('MDVP:RAP')
        
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
        
    with col3:
        DDP = st.text_input('Jitter:DDP')
        
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
        
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer (dB)')
        
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
        
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
        
    with col3:
        APQ = st.text_input('MDVP:APQ')
        
    with col4:
        DDA = st.text_input('Shimmer:DDA')
        
    with col5:
        NHR = st.text_input('NHR')
        
    with col1:
        HNR = st.text_input('HNR')
        
    with col2:
        RPDE = st.text_input('RPDE')
        
    with col3:
        DFA = st.text_input('DFA')
        
    with col4:
        spread1 = st.text_input('spread1')
        
    with col5:
        spread2 = st.text_input('spread2')
        
    with col1:
        D2 = st.text_input('D2')
        
    with col2:
        PPE = st.text_input('PPE')
        
    
    
    # code for Prediction
    parkinsons_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
        
        #BACKUP ([[float(fo), float(fhi), float(flo), float(Jitter_percent), float(Jitter_Abs),
        #                                                   float(RAP), float(PPQ), float(DDP), float(Shimmer), float(Shimmer_dB),
         #                                                  float(APQ3), float(APQ5), float(APQ), float(DDA), float(NHR), float(HNR),
          #                                                 float(RPDE), float(DFA), float(spread1), float(spread2), float(D2), float(PPE)]]) 
        
        if (parkinsons_prediction[0] == 1):
          parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
          parkinsons_diagnosis = "The person doesn't have Parkinson's disease"
        
    st.success(parkinsons_diagnosis)


#Breast Cancer Prediction Page:

if(selected == "Breast Cancer Prediction"):
    
    #page title
    st.title("Breast Cancer Prediction using ML")



# getting the input data from the user

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        mean_radius = st.text_input('Mean Radius')

        mean_smoothness = st.text_input('Mean Smoothness')
        
        mean_symmetry = st.text_input('Mean Symmetry')

        perimeter_error = st.text_input('Perimeter Error')

    with col2:
        mean_texture = st.text_input('Mean Texture')

        mean_compactness = st.text_input('Mean Compactness')

        mean_fractal_dimension = st.text_input('Mean Fractal Dimension')
        
        area_error = st.text_input('Area Error')

    with col3:
        mean_perimeter = st.text_input('Mean Perimeter')

        mean_concavity = st.text_input('Mean Concavity')
        
        radius_error = st.text_input('Radius Error')    

        smoothness_error = st.text_input('Smoothness Error')

    with col4:
        mean_area = st.text_input('Mean Area')

        mean_concave_points = st.text_input('Mean Concave Points')
        
        texture_error = st.text_input('Texture Error')

        compactness_error = st.text_input('Compactness Error')


    with col1:
        concavity_error = st.text_input('Concavity Error')
        
        worst_radius = st.text_input('Worst Radius')
        
        worst_smoothness = st.text_input('Worst Smoothness')
        
        worst_symmetry = st.text_input('Worst Symmetry')

    with col2:        
        concave_points_error = st.text_input('Concave Points Error')
        
        worst_texture = st.text_input('Worst Texture')
        
        worst_compactness = st.text_input('Worst Compactness')
        
        worst_fractal_dimension = st.text_input('Worst Fractal Dimension')

    with col3:
        symmetry_error = st.text_input('Symmetry Error')
        
        worst_perimeter = st.text_input('Worst Perimeter')
        
        worst_concavity = st.text_input('Worst Concavity')

    with col4:
        fractal_dimension_error = st.text_input('Fractal Dimension Error')
        
        worst_area = st.text_input('Worst Area')
        
        worst_concave_points = st.text_input('Worst Concave Points')
        
    
    #code for Prediction
    breast_cancer_check = " "

    if st.button("Breast Cancer Test Result"):
        breast_cancer_prediction = cancer_model.predict([[float(mean_radius), float(mean_texture), float(mean_perimeter), float(mean_area),
                                                          float(mean_smoothness), float(mean_compactness), float(mean_concavity),
                                                          float(mean_concave_points), float(mean_symmetry), float(mean_fractal_dimension),
                                                          float(radius_error), float(texture_error), float(perimeter_error), float(area_error),
                                                          float(smoothness_error), float(compactness_error), float(concavity_error),float(concave_points_error),
                                                          float(symmetry_error), float(fractal_dimension_error), float(worst_radius), float(worst_texture), 
                                                          float(worst_perimeter), float(worst_area), float(worst_smoothness), float(worst_compactness),
                                                          float(worst_concavity), float(worst_concave_points), float(worst_symmetry), float(worst_fractal_dimension)]])
        
        if (breast_cancer_prediction[0] == 0):
        
           breast_cancer_check = "You don't have Breast Cancer."
        else:
         breast_cancer_check = "Sorry! You have Breast Cancer."

    st.success(breast_cancer_check) 


#Lung Cancer Prediction Page:

if(selected == "Lung Cancer Prediction (Y/N)"):
    
    #page title
    st.title("Lung Cancer Prediction using ML")



# getting the input data from the user
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        GENDER = st.text_input("GENDER (M-0/F-1)")
        
    with col2:
        AGE = st.text_input("AGE")
    
    with col3:
        SMOKING = st.text_input("SMOKING")
    
    with col4:
        YELLOW_FINGERS = st.text_input("YELLOW FINGERS")
    
    with col1:
        ANXIETY = st.text_input("ANXIETY")
    
    with col2:
        PEER_PRESSURE = st.text_input("PEER PRESSURE")
    
    with col3:
        CHRONIC_DISEASE = st.text_input("CHRONIC DISEASE")
    
    with col4:
        FATIGUE = st.text_input("FATIGUE")
    
    with col1:
        ALLERGY = st.text_input("ALLERGY")
    
    with col2:
        WHEEZING = st.text_input("WHEEZING")
    
    with col3:
        ALCOHOL_CONSUMING = st.text_input("ALCOHOL CONSUMING")
    
    with col4:
        COUGHING = st.text_input("COUGHING")
    
    with col1:
        SHORTNESS_OF_BREATH = st.text_input("SHORTNESS OF BREATH")
    
    with col2:
        SWALLOWING_DIFFICULTY = st.text_input("SWALLOWING DIFFICULTY")
    
    with col3:
        CHEST_PAIN = st.text_input("CHEST PAIN")
    


# code for Prediction
    lung_cancer_result = " "
    
    # creating a button for Prediction
    
    if st.button("Lung Cancer Test Result"):
        lung_cancer_report = lung_cancer.predict([[int(GENDER), int(AGE), int(SMOKING), int(YELLOW_FINGERS), int(ANXIETY),
                                                   int(PEER_PRESSURE), int(CHRONIC_DISEASE), int(FATIGUE), int(ALLERGY),
                                                   int(WHEEZING), int(ALCOHOL_CONSUMING), int(COUGHING),
                                                   int(SHORTNESS_OF_BREATH),int(SWALLOWING_DIFFICULTY), int(CHEST_PAIN)]])
        
        if (lung_cancer_report[0] == 0):
          lung_cancer_result = "Safe! You don't have Lung Cancer."
        else:
          lung_cancer_result = "Oops! You have Lung Cancer."
        
    st.success(lung_cancer_result)
 



# Background
def set_bg_from_url(url, opacity=1):
    footer = """
    <footer>
        <div style='visibility: visible;margin-top:2rem;justify-content:center;display:flex;'>
            <p style="font-size:1.5rem;">
                "Made by ARNAV TANWAR"                            
            </p>
        </div>
    </footer>
"""
    st.markdown(footer, unsafe_allow_html=True)
    
    
    # Set background image using HTML and CSS
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

# Set background image from URL
set_bg_from_url("https://www.shutterstock.com/image-vector/girl-boy-kids-using-swords-260nw-1753976051.jpg", opacity=0.8)

