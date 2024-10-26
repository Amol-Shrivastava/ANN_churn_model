import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle


## load the trained model
model = tf.keras.models.load_model('model.h5')

## load encoder and scaler
with open('label_encoder_gender_reg.pkl', 'rb') as file:
    label_encoder_gender=pickle.load(file)
with open('onehot_geography_encoder_reg.pkl', 'rb') as file:
    onehot_geography_encoder=pickle.load(file)
with open('scaler_reg.pkl', 'rb') as file:
    scaler = pickle.load(file)

## streamlit app

st.title('Estimated Salary Prediction')

#User Input
age= st.slider('Age', 18, 92)
gender=st.selectbox('Gender', label_encoder_gender.classes_)
geography=st.selectbox('Geography', onehot_geography_encoder.categories_[0])
Tenure=st.slider('Tenure', 0, 10)
Balance=st.number_input('Balance')
NumOfProducts=st.slider('Number of products', 1, 4)
HasCrCard=st.selectbox('Has credit card', [0,1])
IsActiveMember=st.selectbox('Is active member?', [0,1])
exited=st.selectbox('Exited', [0,1])
# estimated_salary=st.number_input('Estimated Salary')

## input data preparation
input_data = pd.DataFrame({
    'Age': [age],
    'Gender': [label_encoder_gender.transform([gender])[0]], 
    'Tenure':[Tenure],
    'Balance':[Balance],
    'NumOfProducts':[NumOfProducts],
    'HasCrCard':[HasCrCard],
    'IsActiveMember':[IsActiveMember],
    'Exited':[exited]
})

### one hot encoder for Geography column
geo_encoded =onehot_geography_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_geography_encoder.get_feature_names_out(['Geography']))

# ## combine encoded geography with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# ## scale the input data
input_data_scaled = scaler.transform(input_data)

# input_scaled

### Predict churn
salary_prediction = model.predict(input_data_scaled)
predicted_salary= salary_prediction[0][0]
st.write(f'Estimated Salary: ${predicted_salary}')