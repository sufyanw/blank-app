import streamlit as st
import pandas as pd
import numpy as np
#for displaying images
from PIL import Image
import seaborn as sns
import codecs
import streamlit.components.v1 as components
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

st.title("Red Wine App")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    image_path = Image.open("image.png")
    st.image(image_path, width=400)
app_page = st.sidebar.selectbox("Select Page", ['Data Exploration', 'Visualization', 'Prediction'])

df = pd.read_csv("wine_quality_red.csv")

if app_page == 'Data Exploration':
    
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

    if st.button("Generate Report"):
        #function to load html fiel
        def read_html_report(file_path):
            with codecs.open(file_path, 'r', encoding="utf-8") as f:
                return f.read()
        
        html_report = read_html_report('report.html')
        
        #displaying file
        st.title("Streamlit Quality Report")
        
        st.components.v1.html(html_report, height=1000,scrolling=True)



if app_page == 'Visualization':
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

if app_page == 'Prediction':
    
    st.title("04 Prediction")
    list_columns = df.columns
    input = st.multiselect("Select variables:", list_columns, ["quality","citric acid"])
    df2 = df[input]
   
    #Step 1 splitting into x and y
    X = df2
    #target variable
    y = df["alcohol"]

    #step 2 splitting into 4 chunks X_train X_test y_train y_test
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

    #step 3 initilaze the linear regression
    lr = LinearRegression()

    #step 4 train the model
    lr.fit(X_train, y_train)

    #step 5 prediction
    predictions = lr.predict(X_test)

    #step 6 evaulate
    mae = metrics.mean_absolute_error(predictions, y_test)
    r2 =  metrics.r2_score(predictions, y_test)
    st.write("Mean Absolute Error:", mae)
    st.write("R2 Output:",r2)