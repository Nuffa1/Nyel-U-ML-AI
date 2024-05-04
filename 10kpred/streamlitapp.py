import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import sklearn
import streamlit as st
from pathlib import Path

datafile = Path(__file__).parents[1] / '10kpred/newdata.csv'

data = pd.read_csv(datafile)

target = data.pop("winner")
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(data, target)
input=[st.number_input("team 1"), st.number_input("team 2"), st.number_input("team 3"), st.number_input("team 4"), st.number_input("team 5"), st.number_input("team 6")]



if st.button("Predict"):
    st.write(model.predict(np.array([input])))


