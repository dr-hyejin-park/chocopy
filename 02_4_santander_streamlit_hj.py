import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly import tools
import plotly.offline as py
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import joblib
from functools import reduce
from itertools import product
from sklearn.metrics import roc_curve, roc_auc_score, classification_report
import sys
sys.path.append("../../../../utils/")
import utils

st.set_page_config(
    page_title="1st App",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title('What-if : santander transaction prediction')
st.write("@ Digital Lab")

st.sidebar.title("Choose your scope")
selectbox = st.sidebar.radio("",("Global", "Local"))

# data path
result_data = ('../scores/02_1_ebm_score_global.pickle')
raw_csv = ('../../../../data/santander/raw/train.csv')
local_df = ('../scores/02_1_ebm_score_local.pickle')
rank_df_pickle = ('predicted_prob_w_rank.pickle')
model_pickle = ('../model/02_1_model_ebm_no_interaction.pickle')

# raw data 
raw = pd.read_csv(raw_csv)
# ebm model 
model = joblib.load(model_pickle)
# predicted probability 
rank_df = joblib.load(rank_df_pickle)

if selectbox == "Global":
    st.subheader("Model briefing")
    fpr, tpr, thresholds = roc_curve(raw.target, rank_df[1])
    auc_score = roc_auc_score(raw.target, rank_df[1])
    # conf_matrix = confusion_matrix(raw.target, rank_df['predicted_label'])
    report = classification_report(raw.target, rank_df['predicted_label'], output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    st.table(df_report)
   
    def model_brief_plot():
        fig = make_subplots(rows=1, cols=3, specs=[[
            {"type":"bar"}, {"type":"scatter"}, {"type":"bar"}]], 
            column_widths=[0.25, 0.3, 0.45])
        fig.add_trace(go.Bar(x=['Label 0', 'Label 1'], y=rank_df.target.value_counts().to_numpy(), marker_color='#20639B', opacity=0.3), row=1, col=1)
        fig.update_xaxes(title_text='#.Customers by labels', row=1, col=1)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', marker_color='#20639B', name=f"AUC: {auc_score:.2f}"), row=1, col=2)
        fig.update_xaxes(title_text=f'AUC: {auc_score:.2f}', row=1, col=2)
        fig.add_trace(go.Histogram(x=rank_df[rank_df.target==0][1], histnorm='percent', marker_color='#4ecdc4', nbinsx=100, opacity=0.35), row=1, col=3)
        fig.add_trace(go.Histogram(x=rank_df[rank_df.target==1][1], histnorm='percent', marker_color='#ff366d', nbinsx=100, opacity=0.35), row=1, col=3)
        fig.update_xaxes(title_text='Probability density', row=1, col=3)
        fig.update_layout(height=500, width=2000, showlegend=False)
        return fig
    model_fig = model_brief_plot()
    st.plotly_chart(model_fig)
    
    st.subheader("Feature importance")
    def feature_importance_plot():
        df_feature_importance = utils.get_feature_importance_df(raw.iloc[:,2:], model)
        df_feature_importance.sort_values(by=['importance'], key=abs, ascending=True, inplace=True)
        fig = go.Figure(go.Bar(
                x=df_feature_importance.importance,
                y=df_feature_importance.feature,
                orientation='h', marker_color='#6D9197'))
        fig.update_layout(height=600, width=1100, showlegend=False)
        return fig
    feature_importance_fig = feature_importance_plot()
    st.plotly_chart(feature_importance_fig)
        
elif selectbox == "Local":
#     st.write("local score")
    cust_index = st.sidebar.text_input("Customer ID: (0~199999)", "1")
    cust_index = int(cust_index)
#   
    customer_selected = "Customer " + str(cust_index)
    st.subheader(customer_selected)

    # load for charting
    # global score

    @st.cache
    def load_data(DATA_URL):
        data = pd.read_pickle(DATA_URL)
        lowercase = lambda x: str(x).lower()
        data.rename(lowercase, axis='columns', inplace=True)
        return data
    df = load_data(result_data)

    # data for waterfall chart and local value/score table
    def load_local_data(DATA_URL, index):
        data = pd.read_pickle(DATA_URL)
        lowercase = lambda x: 'var_'+str(x).lower()
        data.rename(lowercase, axis='columns', inplace=True)

        # sort by absolute score 
        data = data.iloc[index,:200].sort_values(key=abs,ascending=False)
        sorted_col = data.index[:40].to_list()
        data = pd.DataFrame(data.iloc[:40]).transpose()
        return data, sorted_col

    
    cust_local, feat_list = load_local_data(local_df, cust_index)
    # table formatting
    cust_table = pd.DataFrame(raw.loc[cust_index, feat_list]).transpose()
    st.table(cust_table.style.format("{:.3}"))

    # intercept value 
    # intercept = l['specific'][0]['extra']['scores']
    intercept = [-3.189393983845777]

    cust_prob = rank_df.loc[cust_index, 1]
    # cust_rank = rank_df.loc[cust_index, 'PercentRank']

    # predicted prob and contribution 
    def upper_figure():
        fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.05, vertical_spacing=0.05)
        fig.add_trace(go.Histogram(x=rank_df[1], histnorm='percent', name=cust_index, marker_color='#304D63', nbinsx=100, opacity=0.25), row=1, col=1)
        fig.add_scatter(x=[cust_prob,cust_prob], y=[0,20], mode='lines', marker_color=['#ff366d'], name=customer_selected, row=1, col=1)
        fig.update_xaxes(title_text='Predicted probability (label = 1) of Customer {}'.format(cust_index), row=1, col=1)
        fig.update_layout(barmode='overlay')
        fig.update_layout(height=400, width=1900, showlegend=False)
        fig.update_xaxes(title = "Contribution", row=1, col=2)
        fig.add_trace(go.Waterfall(name = customer_selected, orientation = "v", 
                                   increasing_marker_color = '#4ecdc4', decreasing_marker_color='#ff366d', opacity=0.7,
    #                                   x = ['Intercept'] + feat_list[:40], y = list(intercept) + list(cust_local.iloc[0,:40])),row=1,col=2)
                                        x = feat_list[:40], y = list(cust_local.iloc[0,:40])),row=1,col=2)
        return fig

    fig_up = upper_figure()
    st.plotly_chart(fig_up)

    st.subheader('Scores')
    save = []
    # global score chart and dots of local score
    def lower_figure(num_rows=3, num_cols=4):
        fig = make_subplots(specs=[[{"secondary_y": True},{"secondary_y": True},{"secondary_y": True},{"secondary_y": True}],
                                   [{"secondary_y": True},{"secondary_y": True},{"secondary_y": True},{"secondary_y": True}],
                                   [{"secondary_y": True},{"secondary_y": True},{"secondary_y": True},{"secondary_y": True}]
                                  ], 
                            rows=num_rows, cols=num_cols, horizontal_spacing=0.05, vertical_spacing=0.08)
        
        min_score = min(df.scores)
        max_score = max(df.scores)
        k=0
        for i in range(0,num_rows):
            for j in range(0,num_cols):
#                 x = df[df.col_name == feat_list[k]].names
#                 y = df[df.col_name == feat_list[k]].scores
                dist = raw.loc[:,feat_list[k]]
                local_score = pd.DataFrame(data={'names':[dist[cust_index]], 'scores':[cust_local.loc[cust_index,feat_list[k]]]})
                global_score = pd.concat([df[df.col_name == feat_list[k]].iloc[:,:2], local_score], axis=0, ignore_index=True)
                global_score.sort_values(by=['names'], ascending=True, inplace=True)
                fig.add_trace(go.Histogram(x=dist,name=feat_list[k], histnorm='probability', nbinsx=30, opacity=0.25, marker_color='#6D9197'), 
                              secondary_y=True, row=i+1, col=j+1)
                fig.add_scatter(x=global_score.names, y=global_score.scores, name=feat_list[k],
                                mode='lines', marker_color='#20639B', secondary_y=False, row=i+1, col=j+1)
                fig.add_scatter(x=local_score.names, y=local_score.scores, name=customer_selected, 
                                mode='markers', marker_size=[15], marker_color=['#ff366d'], row=i+1, col=j+1)
                fig.update_xaxes(title_text=feat_list[k], row=i+1, col=j+1)
                fig.update_yaxes(range=[min_score, max_score], secondary_y=False, row=i+1, col=j+1)
                fig.update_yaxes(secondary_y=True,showgrid=False)
                fig.update_layout(barmode='overlay',bargap=0.1)
                save.append([feat_list[k], min(global_score.names), max(global_score.names), dist[cust_index]])
                k+=1

        fig.update_layout(height=1200, width=2000, showlegend=False)
        return fig, save

    fig_low, save_df = lower_figure()
    st.plotly_chart(fig_low)
    # if st.checkbox('Submission file'):
    #     st.subheader('Final submission')
    #     st.write(finaldata)

    # Create row, column, and value inputs
    sidebar_expander = st.sidebar.beta_expander("Simulation box")
    with sidebar_expander:
       _, slider_col, _ = st.beta_columns([0.02, 30.96, 0.02])
       with slider_col:
            top1 = st.number_input(save_df[0][0], round(save_df[0][1],1), round(save_df[0][2],1), value=round(save_df[0][3],1), step=0.5)
            top2 = st.number_input(save_df[1][0], round(save_df[1][1],1), round(save_df[1][2],1), value=round(save_df[1][3],1), step=0.5)
            top3 = st.number_input(save_df[2][0], round(save_df[2][1],1), round(save_df[2][2],1), value=round(save_df[2][3],1), step=0.5)
            top4 = st.number_input(save_df[3][0], round(save_df[3][1],1), round(save_df[3][2],1), value=round(save_df[3][3],1), step=0.5)
            top5 = st.number_input(save_df[4][0], round(save_df[4][1],1), round(save_df[4][2],1), value=round(save_df[4][3],1), step=0.5)
            top6 = st.number_input(save_df[5][0], round(save_df[5][1],1), round(save_df[5][2],1), value=round(save_df[5][3],1), step=0.5)
            # top7 = st.number_input(save_df[6][0], round(save_df[6][1],1), round(save_df[6][2],1), value=round(save_df[6][3],1), step=0.5)

    simul_x = pd.DataFrame(raw.loc[cust_index,:]).transpose()
    simul_x[save_df[0][0]] = top1
    simul_x[save_df[1][0]] = top2
    simul_x[save_df[2][0]] = top3
    simul_x[save_df[3][0]] = top4
    simul_x[save_df[4][0]] = top5
    simul_x[save_df[5][0]] = top6
    # simul_x[save_df[6][0]] = top7

    simul_y = pd.DataFrame(model.predict_proba(simul_x.iloc[:,2:]))
    simul_y.rename({0:'Prob 0',1:'Prob 1'}, axis='columns', inplace=True)
    st.sidebar.table(simul_y)
    # st.sidebar.slider("Standard layout slider")


    st.subheader('Solutions : To be continued ')
