import pandas as pd
import numpy as np
import streamlit as st
import pickle

df=pickle.load(open("df.pkl","rb"))
pipeline=pickle.load(open("trained_model.pkl","rb"))

st.title("Laptop Predictor")

company=st.selectbox("Select A Company",df["Company"].unique())
Type=st.selectbox("Select A Type",df["TypeName"].unique())
Ram=st.selectbox("Select A Ram(in GB)",sorted(df["Ram"].unique()))
weight=st.number_input("Weight (in KG)")
touchscreen=st.selectbox("TouchScreen",["No","Yes"])
ips = st.selectbox('IPS',['No','Yes'])
screen_size = st.number_input('Screen Size')
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])
cpu = st.selectbox('CPU',df['CPU Brands'].unique())
hdd = st.number_input('HDD(in GB)')
ssd = st.number_input('SSD(in GB)')
gpu = st.selectbox('GPU',df['Gpu Brand'].unique())
os = st.selectbox('OS',df['os'].unique())


if st.button("Predict Price"):
    ppi=None
    if touchscreen=="Yes":
        touchscreen=1
    else:
        touchscreen=0
    if ips=="Yes":
        ips=1
    else:
        ips=0
    
    if screen_size<=0:
        st.subheader("Screen Size Cant be zero or less than zero")
    if weight<=0:
        st.subheader(" Weight Cant be zero or less than zero")
   
    else:

        X_res=int(resolution.split("x")[0])
        y_res=int(resolution.split("x")[1])
        ppi=((X_res**2)+(y_res**2))**0.5/screen_size
        query = np.array([company,Type,Ram,weight,touchscreen,ips,ppi,cpu,os,gpu,hdd,ssd])


        query_data = {
        'Company': [company],
        'TypeName': [Type],
        'Ram': [Ram],
        'Weight': [weight],
        'TouchScreen': [touchscreen],
        'IPS': [ips],
        'ppi': [ppi],
        'CPU Brands': [cpu],
        'os': [os],
        'Gpu Brand': [gpu],
        'HDD': [hdd],
        'SSD': [ssd]
        }

        query_df = pd.DataFrame(query_data)

        prediction = float(round(np.exp(pipeline.predict(query_df)[0]),2))

        st.subheader(f"The Predicted Price For Your Laptop Is: {prediction}")




