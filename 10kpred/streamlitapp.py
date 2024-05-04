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



if st.button("Predict"):
    st.write(model.predict(np.array([[3023,2501,5172,3026,2502,3871]])))


