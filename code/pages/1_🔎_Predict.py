from io import StringIO
import time
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")

title = st.container()
model1 = st.container()
menu1 = st.container()
menu2 = st.container()
menu3 = st.container()
menu4 = st.container()
predict = st.container()

with title:
    st.markdown("# Predict your heart disease status")
    st.success("""#### To predict your heart disease status, simply follow the steps bellow:
    -> Select the model you want to use;
    -> Enter the parameters that best describe you;
    -> Press the 'Predict' button and wait for the result.""")
    st.error("Keep in mind that this results is not equivalent to a medical diagnosis! This model would never be adopted by health care facilities because of its less than perfect accuracy, so if you have any problems, consult a human doctor.")
    st.write("---")

with model1:
    st.markdown("### *Select the model you want to use:*")
    model = st.selectbox("", options=["k-nearest neighbors", "Logistic Regression", "Random Forest", "Support Vector Machine"])
    st.write("---")

with menu1:
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    with col1:
        st.markdown("### *BMI*")
        st.write("Computed body mass index")
        bmi = st.number_input("Computed body mass index", 1.0, 9999.0, 24.0, 0.1)
    with col2:
        st.markdown("### *Smoking*")
        st.write("Smoked at Least 100 Cigarettes")
        smoking = st.radio("Have you smoked at least 100 cigarettes in your entire life? [Note: 5 packs = 100 cigarettes]", ["Yes", "No"])
    with col3:
        st.markdown("### *AlcoholDrinking*")
        st.write("Heavy Alcohol Consumption Calculated Variable")
        alcohol_drinking = st.radio("Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week)", ["Yes", "No"])
    with col4:
        st.markdown("### *Stroke*")
        st.write("Ever Diagnosed with a Stroke")
        stroke = st.radio("(Ever told) (you had) a stroke.", ["Yes", "No"])
    with col5:
        st.markdown("### *PhysicalHealth*")
        st.write("Number of Days Physical Health Not Good")
        physical_health = st.slider("Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good?", 1, 30, 5, 1)


with menu2:
    col12, col22, col32, col42, col52 = st.columns([1, 1, 1, 1, 1])
    with col12:
        st.write('---')
        st.markdown("### *MentalHealth*")
        st.write("Number of Days Mental Health Not Good")
        mental_health = st.slider("Now thinking about your mental health, which includes stress, depression, and problems with emotions, for how many days during the past 30 days was your mental health not good?", 1, 30, 5, 1)
    with col22:
        st.write('---')
        st.markdown("### *DiffWalking*")
        st.write("Difficulty Walking or Climbing Stairs")
        diff_walking = st.radio("Do you have serious difficulty walking or climbing stairs?", ["Yes", "No"])
    with col32:
        st.write('---')
        st.markdown("### *Sex*")
        st.write("Are you male or female?")
        sex = st.radio('Are you male or female?', ['Male', 'Female'])
    with col42:
        st.write('---')
        st.markdown("### *AgeCategory*")
        st.write("Reported age in five-year age categories calculated variable")
        age = st.select_slider("Fourteen-level age category",
        options=[ '18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older'],
        value='18-24')
    with col52:
        st.write('---')
        st.markdown("### *Race*")
        st.write('Imputed race/ethnicity value')
        race = st.selectbox('Imputed race/ethnicity value (This value is the reported race/ethnicity or an imputed race/ethnicity, if the respondent refused to give a race/ethnicity. The value of the imputed race/ethnicity will be the most common race/ethnicity response for that region of the state)', options=['White', 'Black', 'Hispanic', 'Other', 'Asian', 'American Indian/Alaska Native'], index=0)

with menu3:
    col13, col23, col33, col43, col53 = st.columns([1, 1, 1, 1, 1])
    with col13:
        st.write('---')
        st.markdown("### *Diabetic*")
        st.write("(Ever told) you had diabetes")
        diabetic = st.selectbox("(Ever told) (you had) diabetes? (If 'Yes' and respondent is female, ask 'Was this only when you were pregnant?'. If Respondent says pre-diabetes or borderline diabetes, use response code 4.)", options=['Yes', 'No', 'No, borderline diabetes', 'Yes (during pregnancy)'], index=0)
    with col23:
        st.write('---')
        st.markdown("### *PhysicalActivity*")
        st.write("Exercise in Past 30 Days")
        physical_activity = st.radio("During the past month, other than your regular job, did you participate in any physical activities or exercises such as running, calisthenics, golf, gardening, or walking for exercise?", ["Yes", "No"])
    with col33:
        st.write('---')
        st.markdown("### *GenHealth*")
        st.write("General Health")
        gen_health = st.selectbox("Would you say that in general your health is:", options=['Excellent', 'Very good', 'Good', 'Fair', 'Poor'], index=0)
    with col43:
        st.write('---')
        st.markdown("### *SleepTime*")
        st.write("How Much Time Do You Sleep")
        sleep_time = st.slider("On average, how many hours of sleep do you get in a 24-hour period?", 1, 24, 8, 1)
    with col53:
        st.write('---')
        st.markdown("### *Asthma*")
        st.write("Ever Told Had Asthma")
        asthma = st.radio("(Ever told) (you had) asthma?", ["Yes", "No"])
        
with menu4:
    col14, col24, col34, col44, col54 = st.columns([1, 1, 1, 1, 1])
    with col14:
        st.write('---')
        st.markdown("### *KidneyDisease*")
        st.write("Ever told you have kidney disease?")
        kidney_disease = st.radio("Not including kidney stones, bladder infection or incontinence, were you ever told you had kidney disease?", ["Yes", "No"])
    with col24:
        st.write('---')
        st.markdown("### *SkinCancer*")
        st.write("(Ever told) you had skin cancer?")
        skin_cancer = st.radio("(Ever told) (you had) skin cancer?", ["Yes", "No"])

ss = 14


with predict:
    st.write('---')

    c1, c2 = st.columns([2, 5])
    
    with c1:
        st.markdown("## **Heart Disease Prediction**")
        predict = st.button("Predict")
    with c2:
        if predict:
            st.markdown("## ➕")
        else:
            st.markdown("## ➖")

