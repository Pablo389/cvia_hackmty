import streamlit as st
import pandas as pp
import numpy as np
import pickle


original_title = '<p style="font-family:Georgia; color:red; font-size: 100px;">CVIA</p>'
st.markdown(original_title, unsafe_allow_html=True)
st.divider('Red')

uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
#if uploaded_file is not None:
   # df = extract_data(uploaded_file)
