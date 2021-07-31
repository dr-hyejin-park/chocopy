# app2.py

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import dice_ml
from dice_ml import Dice
# Import custom pkgs
import sys
sys.path.append("../../../utils/")
import utils
from dice_util import (calculate_max_score, tabularize_sparse_cfs, 
                        tabularize_value_gap, tabularize_org_cfs, 
                        tabularize_dice_pred_result)
from dice_chart_util import expected_score_figure, counterfactual_figure


data_path = '../../../data/wine_quality/raw/wine_quality.csv'
model_path = '../../../data/wine_quality/model/wine_quality_ebm_wip.pkl'
dice_path = '../../../data/wine_quality/model/dice_explainer.pkl'


def app():
    
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write('Simulation하고 싶은 와인 ID를 입력하세요')
    wine_id = st.sidebar.text_input("Wine ID: (0~1597)", "1")
    st.title('와인 Quality 높이기')
    
    df_train = pd.read_csv(data_path) 
    model = joblib.load(model_path)
    
    # Dataset for training an ML model
    d = dice_ml.Data(dataframe=df_train,
                    continuous_features= df_train.drop('quality', axis=1).columns.tolist(), 
                    outcome_name='quality')
    # Pre-trained ML model
    m = dice_ml.Model(model=model, backend='sklearn', model_type='regressor')
    # DiCE explanation instance
    dice_explainer = dice_ml.Dice(d,m, method='random')
    
    idx_X = int(wine_id)
    df_X = df_train.drop('quality', axis=1)
    max_score = calculate_max_score(df_X, idx_X, model, dice_explainer)

    st.sidebar.write("")
    st.sidebar.write('Simulation 하고 싶은 Quality 를 입력하세요')
    current_score = round(model.predict(df_X.iloc[[idx_X]])[0],1)
    desired_score = current_score + 1

    if max_score == None:
        default_score = str(current_score + 0.5)
    elif desired_score >= max_score:
        default_score = str(max_score - 0.1)
    else:
        default_score = str(desired_score)
    
    desired_score_string = st.sidebar.text_input("Desired Quality (Max: {})".format(max_score), default_score)
    
    
    desired_score = float(desired_score_string)
    st.subheader('{}번 와인 품질을 {}에서 {}로 바꾸려면 다음과 같은 속성들을 변화시켜야 합니다'.format(wine_id, current_score, desired_score))
    st.write("")
    st.write("")


    # set parameters
    # idx_X = 106 # 관측하고싶은 샘플의 Index

    total_cfs=1
    df_org, df_cfs = tabularize_org_cfs(df_X, idx_X, model, 
                                                dice_explainer, 
                                                desired_score, total_cfs)

    df_gap = tabularize_value_gap(df_org, df_cfs)
    df_pred = tabularize_dice_pred_result(df_org, df_cfs, model)
    df_sparse_cfs = tabularize_sparse_cfs(df_org, df_cfs)
    
    st.altair_chart(counterfactual_figure(df_sparse_cfs))
    total_cfs=5
    df_org, df_cfs = tabularize_org_cfs(df_X, idx_X, model, 
                                                dice_explainer, 
                                                desired_score, total_cfs)
    st.table(df_cfs)
