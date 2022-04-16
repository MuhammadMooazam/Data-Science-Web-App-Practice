from click import option
import streamlit as st
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error , mean_absolute_error , r2_score

header = st.container()
data_sets = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title("Titanic Application")
    st.text("We will work on Titanic dataset")

with data_sets:
    st.header("Titanic Sunk")
    st.text("We will work on Titanic dataset")
    df = sns.load_dataset("titanic")
    df = df.dropna()
    st.write(df.head())

    st.subheader("Count of people according to gender")
    st.bar_chart(df["sex"].value_counts())

    st.subheader("Count of people according to class")
    st.bar_chart(df["class"].value_counts())

    st.subheader("Count of people according to age")
    st.bar_chart(df["age"].sample(20))

with features:
    st.header("This is our features")
    st.text("These are some of our app features.")
    st.markdown("1. **Feature 1:** This is feature # 1.")
    st.markdown("2. **Feature 2:** This is feature # 2.")
    st.markdown("3. **Feature 3:** This is feature # 3.")

with model_training:
    st.header("The model is training")
    st.text("We will work on Titanic dataset")

    input , display = st.columns(2)
    max_depth = input.slider("How many people?" , min_value=0 , max_value=100 , value=20 , step=5)

n_estimators = input.selectbox("How many tree should be in RF?" , options=[50,100,150,200,250,300,"No Limit"])

input.write(df.columns)

input_features = input.text_input("How many features you want to choose?" , value="age")

model = RandomForestRegressor(max_depth=max_depth , n_estimators=n_estimators)

if n_estimators=="No Limit":
    model = RandomForestRegressor(max_depth=max_depth)
else:
    model = RandomForestRegressor(max_depth=max_depth , n_estimators=n_estimators)

x=df[[input_features]]
y=df[["fare"]]

model.fit(x,y)
pred = model.predict(y)

display.subheader("Mean absolute error of the model is: ")
display.write(mean_squared_error(y,pred))
display.subheader("Mean squared error of the model is: ")
display.write(mean_absolute_error(y,pred))
display.subheader("R squared error of the model is: ")
display.write(r2_score(y,pred))

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 