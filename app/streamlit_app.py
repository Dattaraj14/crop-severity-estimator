import streamlit as st
import pickle
import numpy as np
import os

model_xgb = pickle.load(open('models/xgb_model.pkl', 'rb'))
scaler    = pickle.load(open('models/scaler.pkl',    'rb'))

@st.cache_resource
def load_features():
    features = np.load('data/features/features.npy')
    labels   = np.load('data/features/labels.npy')
    yields   = np.load('data/features/yields.npy')
    return features, labels, yields

features, labels, yields = load_features()

st.title('Crop Disease Severity Estimator')
st.write('Select a disease and click predict to estimate yield loss.')

disease_list = sorted(list(set(labels)))
selected_disease = st.selectbox('Select disease type', disease_list)

col1, col2 = st.columns(2)
with col1:
    temp     = st.slider('Temperature (C)',   5,  45, 22)
    humidity = st.slider('Humidity (%)',     20, 100, 70)
with col2:
    rainfall = st.slider('Rainfall (mm)',     0,  50,  5)
    wind     = st.slider('Wind speed (km/h)', 0,  60, 15)

if st.button('Predict yield loss'):
    indices = np.where(labels == selected_disease)[0]
    idx     = np.random.choice(indices)

    feat     = features[idx].reshape(1, -1)
    weather  = np.array([[temp, humidity, rainfall, wind]])
    w_scaled = scaler.transform(weather)
    X        = np.concatenate([feat, w_scaled], axis=1)

    loss_pct   = round(float(model_xgb.predict(X)[0]), 1)
    true_loss  = round(float(yields[idx]), 1)

    st.metric('Predicted Yield Loss', f'{loss_pct}%')
    st.caption(f'Reference yield loss for {selected_disease}: {true_loss}%')

    if loss_pct < 20:
        st.success('Low severity — crop is mostly safe')
    elif loss_pct < 50:
        st.warning('Moderate severity — treatment recommended')
    else:
        st.error('High severity — act immediately!')