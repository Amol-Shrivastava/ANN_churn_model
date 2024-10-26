import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle


## load the trained model
model = tf.keras.models.load_model('model.h5')

## load encoder and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender=pickle.load(file)
with open('onehot_geography_encoder.pkl', 'rb') as file:
    onehot_geography_encoder=pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

## streamlit app

st.title('Customer Churn Prediction')

#User Input
age= st.slider('Age', 18, 92)
gender=st.selectbox('Gender', label_encoder_gender.classes_)
geography=st.selectbox('Geography', onehot_geography_encoder.categories_[0])
Tenure=st.slider('Tenure', 0, 10)
Balance=st.number_input('Balance')
NumOfProducts=st.slider('Number of products', 1, 4)
HasCrCard=st.selectbox('Has credit card', [0,1])
IsActiveMember=st.selectbox('Is active member?', [0,1])
estimated_salary=st.number_input('Estimated Salary')

## input data preparation
input_data = pd.DataFrame({
    'Age': [age],
    'Gender': [label_encoder_gender.transform([gender])[0]], 
    'Tenure':[Tenure],
    'Balance':[Balance],
    'NumOfProducts':[NumOfProducts],
    'HasCrCard':[HasCrCard],
    'IsActiveMember':[IsActiveMember],
    'EstimatedSalary':[estimated_salary]
})

### one hot encoder for Geography column
geo_encoded =onehot_geography_encoder.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_geography_encoder.get_feature_names_out(['Geography']))

# ## combine encoded geography with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# ## scale the input data
input_data_scaled = scaler.transform(input_data)

# input_scaled

### Predict churn
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

st.write(f'Churn Probability: {prediction_prob:.5f}')
print(prediction_prob)


if prediction_prob > 0.5:
    st.write('The customer is likely to chrun.')
else:
    st.write('The customer is not likely to churn.')