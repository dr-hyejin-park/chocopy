# app2.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import sys
sys.path.append("../../../utils/")
import utils
from ebm_chart_util import contribution_figure, ebm_score_figure
from ebm_util import tabularize_global_explanation, tabularize_local_explanation

data_path = '../../../data/wine_quality/raw/wine_quality.csv'
model_path = '../../../data/wine_quality/model/wine_quality_ebm_wip.pkl'

def app():
    wine_id = st.sidebar.text_input("Wine ID: (0~1597)", "1")
    st.title('와인 Quality 이해하기')
    st.subheader('')
    
    
    df = pd.read_csv(data_path)
    model = joblib.load(model_path)

    global_df = tabularize_global_explanation(df.iloc[:,:-1], model)
    local_df = tabularize_local_explanation(df.iloc[:,:-1], df.quality, model)

    instance = int(wine_id)
    col1, col2 = st.beta_columns(2)
    waterfall = contribution_figure(df, local_df, instance, 200, 450)
    col1.subheader("와인 {}의 품질은 다음과 같은 속성으로 이루어집니다.".format(wine_id))
    col1.altair_chart(waterfall)

    col2.subheader("")
    
    feature_list = df.iloc[:,:-1].columns.tolist()
    # feature = 'alcohol'
    i=0
    for col in st.beta_columns(4):    
        col.altair_chart(ebm_score_figure(df, feature_list[i], global_df, local_df, instance, 150, 150))
        i+=1
    i=4
    for col in st.beta_columns(4):    
        col.altair_chart(ebm_score_figure(df, feature_list[i], global_df, local_df, instance, 150, 150))
        i+=1
    i=8
    for col in st.beta_columns(4):    
        if i==11:
            col.write("")
        else:
            col.altair_chart(ebm_score_figure(df, feature_list[i], global_df, local_df, instance, 150, 150))
        i+=1
    
    # col1, col2, col3 = st.beta_columns(3)
    
    # col1.subheader('fixed acidity 한글로')
    # col1.write(c)

    # col2.subheader("alcohol ")
    # col2.write(c)

    # col3.subheader("citric acid")
    # col3.write(c)
    
