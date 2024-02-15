import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

with open('lr.pkl', 'rb') as file:
    model = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

def preprocess_input(df):
    df_processed = pd.get_dummies(df, columns=['METRO3', 'REGION', 'ZADEQ', 'ASSISTED', 'TENURE'], 
                                  prefix=['METRO3', 'REGION', 'ZADEQ', 'ASSISTED', 'TENURE'])
    num_cols = ['AGE1', 'LMED', 'FMR', 'BEDRMS', 'VALUE', 'ROOMS', 'ZINC2', 'ZSMHC', 'UTILITY', 'OTHERCOST', 'COSTMED', 'BURDEN', 'INCRELAMIPCT']
    df_processed[num_cols] = scaler.transform(df_processed[num_cols])
    return df_processed

st.title('Predict Building Structure Type')

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

if st.button('Predict Structure Type'):
    input_data = {
        'METRO3': [METRO3], 'REGION': [REGION], 'ZADEQ': [ZADEQ], 'ASSISTED': [ASSISTED], 'TENURE': [TENURE],
        'AGE1': [AGE1], 'LMED': [LMED], 'FMR': [FMR], 'BEDRMS': [BEDRMS], 'VALUE': [VALUE], 'ROOMS': [ROOMS], 
        'ZINC2': [ZINC2], 'ZSMHC': [ZSMHC], 'UTILITY': [UTILITY], 'OTHERCOST': [OTHERCOST], 'COSTMED': [COSTMED], 
        'BURDEN': [BURDEN], 'INCRELAMIPCT': [INCRELAMIPCT]
    }
    df_user_input = pd.DataFrame.from_dict(input_data)
    processed_input = preprocess_input(df_user_input)

    expected_columns = [
    'AGE1', 'LMED', 'FMR', 'BEDRMS', 'VALUE', 'ROOMS', 'ZINC2', 'ZSMHC', 'UTILITY', 'OTHERCOST', 'COSTMED', 'BURDEN', 'INCRELAMIPCT', 
    "METRO3_'1'", "METRO3_'2'", "METRO3_'3'", "METRO3_'4'", "METRO3_'5'", 
    "REGION_'1'", "REGION_'2'", "REGION_'3'", "REGION_'4'", 
    'ZADEQ_adequate', 'ZADEQ_moderately inadequate', 'ZADEQ_severely inadequate', 
    'ASSISTED_Assisted', 'ASSISTED_Not Assisted', 'ASSISTED_Unknown', 
    'TENURE_Owner-occupied', 'TENURE_Rented']

    for col in expected_columns:
        if col not in processed_input.columns:
            processed_input[col] = 0
    
    processed_input = processed_input[expected_columns]

    prediction = model.predict(processed_input)

    predicted_value=''
    
    if prediction[0]==0:
        predicted_value="Single Unit"
    elif prediction[0]==1:
        predicted_value="2-4 Unit Building"
    elif prediction[0]==2:
        predicted_value="5-19 Unit Building"
    elif prediction[0]==3:
        predicted_value= "20-49 Unit Building"
    elif prediction[0]==4:
        predicted_value="50+ Unit Building"
    elif prediction[0]==5:
        predicted_value= "Mobile Home"
    else:
        predicted_value="Unknown"

    st.write('Predicted Structure Type: ', predicted_value)
