import streamlit as st
from multiapp import MultiApp
from apps import app1, app2, app3, app4
import joblib

st.set_page_config(
    page_title="what-if",
    layout="wide",
    initial_sidebar_state="expanded",
)


app = MultiApp()

# Add all your application here
app.add_app("와인 Data 설명", app1.app)
app.add_app("와인 Quality 요인 분석", app2.app)
app.add_app("와인 Quality status", app3.app)
app.add_app("와인 Quality simulation", app4.app)

# The main app
app.run()
