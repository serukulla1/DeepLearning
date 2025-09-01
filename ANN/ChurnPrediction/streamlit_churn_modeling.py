
import numpy as np
import pandas as pd
import streamlit as st 
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

##Load the dataset
dataset = pd.read_csv(r'C:\Swapna\Learning\PYTHON\FS DataScience\GenAI\Practise\ChurnPrediction\Churn_Modelling.csv')

##Data Preprocessing
X = dataset.iloc[:,3:-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)

##converting Gender col to numnerical value
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2]) 
print(X)

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])],remainder='passthrough')
X=np.array(ct.fit_transform(X))

sc = StandardScaler()
X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=0)

#Build ANN

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=5, activation='relu'))
ann.add(tf.keras.layers.Dense(units=4, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

##Train ANN

ann.fit(X_train, y_train, batch_size=32, epochs=15, validation_data=(X_test, y_test))


#Streamlit app
st.title("Churn Prediction App")

#User inputs
st.sidebar.header("Input Features")
credit_score = st.sidebar.number_input("Credit Score", min_value=0)
geography = st.sidebar.selectbox("Geography", ("France", "Germany","Spain"))
gender = st.sidebar.selectbox("Gender", ("Female","Male"))
age = st.sidebar.number_input("Age", min_value=0)
tenure = st.sidebar.number_input("Tenure", min_value=0)
balance = st.sidebar.number_input("Balance", min_value=0.0, format="%.2f")
num_of_products = st.sidebar.number_input("Number of Products", min_value=1, max_value=4)
has_cr_card = st.sidebar.selectbox("Has Credit Card", (0, 1))
is_active_member = st.sidebar.selectbox("Is Active Member", (0, 1))
estimated_salary = st.sidebar.number_input("Estimated Salary", min_value=0.0, format="%.2f")


#Transform user input
user_data = np.array([[credit_score, geography, gender,age,tenure,balance,num_of_products,has_cr_card,is_active_member,estimated_salary]])
user_data[:,2] = le.transform(user_data[:,2])
user_data = np.array(ct.transform(user_data))
user_data = sc.transform(user_data)

#predict churn
if st.button("Predict"):
    prediction = ann.predict(user_data)
    print(f"Churn Prediction value: **{prediction}**")
    prediction = (prediction > 0.5)
    result = "Churn" if prediction else "No Churn"
    st.write(f"The prediction is: **{result}**")
    
    #Evaluate Model
    y_pred = ann.predict(X_test)
    y_pred = (y_pred > 0.5)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: **{accuracy:.2f}**")
    
    cm = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix:")
    st.write(cm)
    