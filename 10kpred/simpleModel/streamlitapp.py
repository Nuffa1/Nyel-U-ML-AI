import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import sklearn
import streamlit as st
from pathlib import Path

st.title("Predictions App: Based on Logistic Regression with 2019 data from 10k")

datafile = Path(__file__).parents[1] / '10kpred/newdata.csv'

data = pd.read_csv(datafile)

target = data.pop("winner")
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(data, target)
input=[st.number_input("team 1 (red alliance)", step=1), st.number_input("team 2 (red alliance)", step=1), st.number_input("team 3 (red alliance)", step=1), st.number_input("team 4 (blue alliance)", step=1), st.number_input("team 5 (blue alliance)", step=1), st.number_input("team 6 (blue alliance)", step=1)]



if st.button("Predict"):
    if(model.predict(np.array([input])) == 0):
        st.write("Predicted Outcome: Red Alliance Win")
    
    if(model.predict(np.array([input])) == 1):
        st.write("Predicted Outcome: Blue Alliance Win")


