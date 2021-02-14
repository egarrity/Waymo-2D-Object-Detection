import homepage
import about
import streamlit as st
PAGES = {
    "Homepage": homepage,
    "About": about
}
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()