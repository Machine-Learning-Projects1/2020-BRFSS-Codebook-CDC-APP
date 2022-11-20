import streamlit as st

# title = st.container()
# body = st.container()

st.set_page_config(
    page_title="Machine Learning App",
    page_icon="👋",
)

with st.container():
    col1, col2 = st.columns([1,9])
    with col1:
        st.image('https://raw.githubusercontent.com/Machine-Learning-Projects1/CDC_ML/test-ui/assets/logo.png', width=80)
    with col2:
        st.markdown("# 2020-BRFSS-Codebook-CDC-APP")
    st.markdown("### A machin learning project to predict Heart disease using 17 Risk Factor provided by BRFSS")
    
with st.container():
    st.markdown('''
    According to the CDC, heart disease is one of the leading causes of death for people of most races in the US. Our ML project leads to a better understanding of how we can predict heart disease.

    Factors assessed by the BRFSS in 2020 included health status and healthy days, exercise, inadequate sleep, chronic health conditions,oral health, tobacco use, cancer screenings, and health-care access (core section). 
    Optional Module topics for 2020 included prediabetes and diabetes, cognitive decline, electronic cigarettes, cancer survivorship (type, treatment, pain management) and sexual orientation/gender identity (SOGI).
    ''')
    st.write("---")

with st.container():
    # st.markdown("### *Select* **'🔎Predict'** *page to get started!*")
    st.success("👈 Select 🔎 **Predict** page to get started!")