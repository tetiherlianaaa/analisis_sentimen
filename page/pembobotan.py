import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Fungsi untuk memproses dan menghitung TF-IDF
def process_tfidf(df):
    df['kalimat'] = df['tweet_tokens_WSW']
    texts = df['kalimat']
    labels = df['polarity']
    
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, min_df=5, max_df=0.8, stop_words=None)
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    tfidf_total_list = []
    for i in range(len(df)):
        tfidf_values = tfidf_matrix[i].toarray().flatten()
        total_tfidf = sum(tfidf_values)
        tfidf_total_list.append(total_tfidf)
    
    output_df = pd.DataFrame({
        'kalimat': df['kalimat'],
        'tfidf': tfidf_total_list,
        'polarity': df['polarity']
    })
    
    return output_df, tfidf_vectorizer

# Fungsi untuk menghitung TF-IDF untuk input teks tunggal
def compute_tfidf_single(input_text, tfidf_vectorizer):
    tfidf_values = tfidf_vectorizer.transform([input_text]).toarray().flatten()
    total_tfidf = sum(tfidf_values)
    return total_tfidf

# Streamlit App
def main():
    st.title("TF-IDF Calculation")

    # Unggah file CSV
    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if 'tweet_tokens_WSW' not in df.columns or 'polarity' not in df.columns:
            st.error("File CSV harus memiliki kolom 'tweet_tokens_WSW' dan 'polarity'.")
            return
        
        output_df, tfidf_vectorizer = process_tfidf(df)
        
        st.write("Data dengan nilai TF-IDF:")
        st.write(output_df)

        # Unduh file hasil
        csv = output_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Unduh CSV dengan TF-IDF", data=csv, file_name='labeled_sentiment_tfidf.csv', mime='text/csv')

    # Input teks tunggal untuk menghitung TF-IDF
    input_text = st.text_area("Masukkan teks untuk menghitung TF-IDF", "")

    if input_text and uploaded_file:
        total_tfidf = compute_tfidf_single(input_text, tfidf_vectorizer)
        st.write(f"Total nilai TF-IDF untuk teks yang dimasukkan: {total_tfidf}")

if __name__ == "__main__":
    main()