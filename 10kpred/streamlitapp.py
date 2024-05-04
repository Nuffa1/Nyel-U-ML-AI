import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import sklearn
import streamlit as st

data = pd.read_csv('newdata.csv')
target = data.pop("winner")
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(data, target)
model.predict(np.array([[3023,2501,5172,3026,2502,3871]]))

st.title("Streamlit App")