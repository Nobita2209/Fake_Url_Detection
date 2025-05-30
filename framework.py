import streamlit as st
from urllib.request import urlopen,Request
from urllib.error import URLError
from bs4 import BeautifulSoup
x = st.slider("Select a value")
st.write(x, "squared is", x * x)
clicked = st.button("Click me")
#st.form_submit_button("sign up")
st.link_button("go to galary",'''url''')

st.button("Reset", type="primary")
if st.button('Say hello'):
    st.write('Why hello there')
else:
    st.write('Goodbye')

def getSoup(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0'}
        req = Request(url, headers=headers)
        req = urlopen(req)
        soup = BeautifulSoup(req, 'html.parser')
        return soup
    except URLError as e:
        print("Error opening URL:", e)
        return None
    
getSoup("https://www.google.com/")