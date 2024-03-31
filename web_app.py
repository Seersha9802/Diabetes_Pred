

import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import streamlit as st

# loading the save model
loaded_model=pickle.load(open('D:/python/PyCharm/PyCharm Community Edition 2023.3.2/Streamlit_deployment/trained_model.sav','rb'))


# creating a function
def diabetes_prediction(input_data):

    to_an_array = np.asarray(input_data)
    reshape_array = to_an_array.reshape(1, -1)
    prediction = loaded_model.predict(reshape_array)

    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'




def main():

    # giving a title
    st.title('Diabetes Prediction App')

    #getting input data from user

    Pregnancies=st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI Value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    Age = st.text_input('Age of the person')


    #code for prediction
    diagnosis=''

    #creating a button
    if st.button('Diabetes Test result'):
        diagnosis=diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])

    st.success(diagnosis)


if __name__ =='__main__':
    main()

