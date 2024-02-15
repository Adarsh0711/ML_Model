import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

with open('lr.pkl', 'rb') as file:
    model = pickle.load(file)
with open('scalar.pkl', 'rb') as file:
    scaler = pickle.load(file)

def preprocess_input(df):
    df_processed = pd.get_dummies(df, columns=['METRO3', 'REGION', 'ZADEQ', 'ASSISTED', 'TENURE'], 
                                  prefix=['METRO3', 'REGION', 'ZADEQ', 'ASSISTED', 'TENURE'])
    
    num_cols = ['AGE1', 'LMED', 'FMR', 'BEDRMS', 'VALUE', 'ROOMS', 'ZINC2', 'ZSMHC', 'UTILITY', 'OTHERCOST', 'COSTMED', 'BURDEN', 'INCRELAMIPCT']
    df_processed[num_cols] = scaler.transform(df_processed[num_cols])
    
    return df_processed

st.title('Predict STRUCTURETYPE')

METRO3 = st.selectbox('METRO3', ["1", "2", "3", "4", "5"])
REGION = st.selectbox('REGION', ["1", "2", "3", "4"])
ZADEQ = st.selectbox('ZADEQ', ['adequate', 'moderately inadequate', 'severely inadequate'])
ASSISTED = st.selectbox('ASSISTED', ['Unknown', 'Not Assisted', 'Assisted'])
TENURE = st.selectbox('TENURE', ['Owner-occupied', 'Rented'])

AGE1 = st.number_input('AGE1')
LMED = st.number_input('LMED')
FMR = st.number_input('FMR')
BEDRMS = st.number_input('BEDRMS')
VALUE = st.number_input('VALUE')
ROOMS = st.number_input('ROOMS')
ZINC2 = st.number_input('ZINC2')
ZSMHC = st.number_input('ZSMHC')
UTILITY = st.number_input('UTILITY')
OTHERCOST = st.number_input('OTHERCOST')
COSTMED = st.number_input('COSTMED')
BURDEN = st.number_input('BURDEN')
INCRELAMIPCT = st.number_input('INCRELAMIPCT')

if st.button('Predict STRUCTURETYPE'):
    input_data = {
        'METRO3': [METRO3], 'REGION': [REGION], 'ZADEQ': [ZADEQ], 'ASSISTED': [ASSISTED], 'TENURE': [TENURE],
        'AGE1': [AGE1], 'LMED': [LMED], 'FMR': [FMR], 'BEDRMS': [BEDRMS], 'VALUE': [VALUE], 'ROOMS': [ROOMS], 
        'ZINC2': [ZINC2], 'ZSMHC': [ZSMHC], 'UTILITY': [UTILITY], 'OTHERCOST': [OTHERCOST], 'COSTMED': [COSTMED], 
        'BURDEN': [BURDEN], 'INCRELAMIPCT': [INCRELAMIPCT]
    }
    df_user_input = pd.DataFrame.from_dict(input_data)
    
    processed_input = preprocess_input(df_user_input)

    prediction = model.predict(processed_input)

    st.write(f'Predicted STRUCTURETYPE: {prediction[0]}')
