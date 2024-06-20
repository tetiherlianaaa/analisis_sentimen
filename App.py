import streamlit as st
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

# Set the page configuration at the top of the script
st.set_page_config(page_title="Analisis Sentimen", page_icon="icon/classification.png")

import pandas as pd
import matplotlib.pyplot as plt

import page.home
import page.Pelabelan
import page.wordcloud
import page.pembobotan 
import page.klasifikasiNaiveBayes
import page.klasifikasiKnn

PAGES = {
    "Beranda": page.home,
    "Pelabelan InSet": page.Pelabelan,
    "WordCloud dan Grafik":page.wordcloud,
    "Proses Pembobotan TFIDF":page.pembobotan,
    "Klasifikasi Model Naive Bayes":page.klasifikasiNaiveBayes,
    "Klasifikasi Model KNN":page.klasifikasiKnn,
}

def set_page_config():
    page_font = "Times New Roman"
    st.markdown(
        f"""
        <style>
            body {{
                font-family: "{page_font}", sans-serif;
            }}
            .reportview-container .main .block-container {{
                max-width: 1200px;
                padding-top: 5rem;
                padding-right: 5rem;
                padding-left: 5rem;
                padding-bottom: 5rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def main():
    set_page_config()

    #st.sidebar.markdown('<p style="font-family: Times New Roman; font-size: 24px; font-weight: bold;">Analisis Sentimen</p>', unsafe_allow_html=True)
    st.sidebar.markdown('<p style="font-family: Times New Roman; font-size: 24px; font-weight: bold; margin-top: 0; margin-bottom: 0;">Analisis Sentimen KNN & Naive Bayes</p>', unsafe_allow_html=True)

    page = st.sidebar.radio("", list(PAGES.keys()))

    with st.spinner(f"Loading {page} ..."):
        PAGES[page].main()
    
   #st.sidebar.markdown('<p style="font-family: Times New Roman; font-size: 24px; font-weight: bold;">About App</p>', unsafe_allow_html=True)
    
    st.sidebar.info(
        """

        """
    )

if __name__ == "__main__":
    main()
