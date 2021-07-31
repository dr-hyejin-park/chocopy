# app1.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

dict_path = '../../../data/wine_quality/processed/wine_quality_data_dict.pkl'

def app():
    st.title('와인의 품질은 어떤 속성으로 결정되나요?')
    st.subheader('알코올, 황산염, 잔류설탕, 밀도, 휘발성 산도와 같은 화학물질  11개 속성으로 이루어집니다.')
    
    df = joblib.load(dict_path)
    st.table(df)