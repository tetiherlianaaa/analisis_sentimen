import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import seaborn as sns

# Fungsi untuk menghasilkan WordCloud
def generate_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    st.pyplot(plt)

# Fungsi untuk memproses dan menampilkan hasil
def process_file(df):
    # Mengunduh daftar stop words bahasa Indonesia
    stop_words = set(stopwords.words('indonesian'))

    # Pisahkan teks berdasarkan sentimen
    positive_text = ' '.join(df[df['polarity'] == 'positive']['tweet_tokens_WSW'].explode().astype(str))
    negative_text = ' '.join(df[df['polarity'] == 'negative']['tweet_tokens_WSW'].explode().astype(str))

    # Membuat WordCloud untuk sentimen positif
    st.subheader('Word Cloud Sentimen Positif')
    generate_wordcloud(positive_text, 'Word Cloud Sentimen Positif')

    # Membuat WordCloud untuk sentimen negatif
    st.subheader('Word Cloud Sentimen Negatif')
    generate_wordcloud(negative_text, 'Word Cloud Sentimen Negatif')

    # Menghitung jumlah sentimen
    sentiment_counts = df['polarity'].value_counts()

    # Plot donut dengan inset
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
    ax.set_title('Persentase Sentimen')

    # Inset untuk label persentase
    axin = ax.inset_axes([0.6, 0.6, 0.35, 0.35])
    axin.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
    axin.set_title('Sentimen')
    st.pyplot(fig)

    # Plot bar menggunakan seaborn
    plt.figure(figsize=(8, 6))
    sns.countplot(x='polarity', data=df, palette='viridis')
    plt.title('Jumlah Sentimen')
    plt.xlabel('Sentimen')
    plt.ylabel('Jumlah')
    st.pyplot(plt)

# Streamlit App
def main():
    st.title("Analisis Sentimen dengan WordCloud dan Grafik")

    # Unggah file CSV
    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if 'tweet_tokens_WSW' not in df.columns or 'polarity' not in df.columns:
            st.error("File CSV harus memiliki kolom 'tweet_tokens_WSW' dan 'polarity'.")
            return
        
        process_file(df)

if __name__ == "__main__":
    main()
