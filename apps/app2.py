# app2.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from interpret.glassbox import ExplainableBoostingRegressor
import joblib
import sys
sys.path.append("../../../utils/")
import utils
from ebm_chart_util import feature_importance_figure, ebm_pdp_figure
from ebm_util import tabularize_global_explanation, tabularize_local_explanation

data_path = '../../../data/wine_quality/raw/wine_quality.csv'
model_path = '../../../data/wine_quality/model/wine_quality_ebm_wip.pkl'

def app():
    st.title('Red 와인의 맛을 결정하는 요인은?')
    st.subheader('알코올 도수 (alcohol), 황산염 (sulphates), 휘발성 산도 (volatile acidity), 총 이산화황 (total sulfur dioxide) 이 크게 영향을 미치며 어떤 값을 가지냐에 따라 Quality 가 좌우됩니다.')

    # st.subheader('와인 table')
    df = pd.read_csv(data_path)
    st.write("")
    
    model = joblib.load(model_path)
    
    col1, col2, col3 = st.beta_columns((2,1,1))
    
    new_title0 = '<p style="font-family:sans-serif; font-size: 21px;">Quality에 미치는 성분별 영향도</p>'
    col1.markdown(new_title0, unsafe_allow_html=True)
    col1.write("")
    df_feature_importance = utils.tabularize_feature_importance(df.iloc[:,:-1], model)
    df_feature_importance.sort_values(by='importance', ascending=False, inplace=True)
    col1.altair_chart(feature_importance_figure(df_feature_importance, 20))
    
    
    feature_top4 = df_feature_importance[:4].feature.tolist()
    global_df = tabularize_global_explanation(df.iloc[:,:-1], model)
    
    new_title1 = '<p style="font-family:sans-serif; font-size: 21px;">주요 성분별 품질 기여도</p>'
    col2.markdown(new_title1, unsafe_allow_html=True)
    col2.write("")
    col2.altair_chart(ebm_pdp_figure(df, feature_top4[0], global_df, 150, 150))
    col2.altair_chart(ebm_pdp_figure(df, feature_top4[2], global_df, 150, 150))
    
    new_title2 = '<p style="font-family:sans-serif; color:white; font-size: 21px;">.</p>'
    col3.markdown(new_title2, unsafe_allow_html=True)
    col3.write("")
    col3.altair_chart(ebm_pdp_figure(df, feature_top4[1], global_df, 150, 150))
    col3.altair_chart(ebm_pdp_figure(df, feature_top4[3], global_df, 150, 150))
    