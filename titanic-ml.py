import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

train=pd.read_csv("LogicTrain.csv")

st.write("""
# Titanic dataset App
This app predicts the **Survival**!
""")
st.warning(""" Pclass: Class in which the person was travelling""")
st.warning(""" Gender: 0 : Male and 1 : Female""")
st.sidebar.header('User Input Parameters')
def user_input_features():
    Pclass = st.sidebar.slider('Pclass', 1, 3, 2)
    Age = st.sidebar.slider('Age', 0.0, 80.0, 26.0)
    Gender = st.sidebar.slider('Gender', 0,1,0)
    data = {'Pclass': Pclass,
            'Age': Age,
            'Gender': Gender
            }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

def f(s):
    if s=="male":
        return 0
    else:
        return 1
train["Gender"]=train.Sex.apply(f)

train.drop(['Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked','Sex'],axis=1,inplace=True)

y_train=train.Survived

train.Age.fillna(train.Age.mean(),inplace=True)

train.drop(['Survived'],axis=1,inplace=True)

x_train=train.to_numpy()
y_train=y_train.to_numpy()

cls=LogisticRegression()
cls.fit(x_train,y_train)

prediction = cls.predict(df)
prediction_proba = cls.predict_proba(df)

def pre(prediction):
    if prediction==1:
        st.success('Survive')
    else:
        st.success('Can not Survive')
    
st.subheader('Class labels and their corresponding index number')

st.subheader('Prediction')
pre(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)