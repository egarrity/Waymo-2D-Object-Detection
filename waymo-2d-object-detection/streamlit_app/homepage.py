import time
import streamlit as st

import numpy as np 
import pandas as pd 
import cv2


def app():
    st.title("Welcome to Object Detection Central")
    st.header("Detecting cars, cyclists and pedestrians 🚗🚴‍♂🚶‍♂️")

    img = cv2.imread('/Users/peterfagan/Code/Waymo-2D-Object-Detection/assets/waymo.jpeg')
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img, width=600)

    page_selection = pd.DataFrame({
      'first column': [1, 2, 3, 4],
      'second column': [10, 20, 30, 40]
    })

    expander = st.beta_expander("FAQ")
    expander.write("Here you could put in some really, really long explanations...")

