import streamlit as st
import pandas as pd
import numpy as np
#for displaying images
from PIL import Image
import seaborn as sns

st.title("Red Wine App")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    image_path = Image.open("image.png")
    st.image(image_path, width=400)

df = pd.read_csv("wine_quality_red.csv")

st.dataframe(df.head(3))

st.subheader("01 Description of the Dataset")

st.dataframe(df.describe())

st.subheader("02 Missing values")

dfnull = df.isnull()/len(df)*100
total_missing = dfnull.sum().round(2)
st.write(total_missing)
if total_missing[0] == 0.0:
    st.success("Congrats, you have no missing values!")
else:
    st.error("U cooked twin")

st.subheader("03 Data Visualization")

list_columns = df.columns
values = st.multiselect("Select 2 variables:", list_columns, ["quality","citric acid"])

#creation of the line chart
st.line_chart(df, x=values[0], y=values[1])

#creation of bar chart
st.bar_chart(df, x=values[0], y=values[1])


values_pairplot = st.multiselect("Select 4 variables:", list_columns, ["quality","citric acid", "alcohol", "chlorides"])
#creation of pairplot

df2 = df[[values_pairplot[0],values_pairplot[1],values_pairplot[2],values_pairplot[3]]]
st.pyplot(sns.pairplot(df2))