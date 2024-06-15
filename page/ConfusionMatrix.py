import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from page.fetch import display_pie_chart
from page import ConfusionMatrix

# Then use it in your code
PAGES = {
    "Evaluasi Matrix": ConfusionMatrix,
    # Other pages
}

with open('styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def load_data(file):
    if file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    elif file.name.endswith('.csv'):
        try:
            return pd.read_csv(file, encoding='utf-8', on_bad_lines='skip')
        except pd.errors.ParserError as e:
            st.error(f"Error reading CSV file: {e}")
            return None
    else:
        st.error("Tipe file tidak didukung. Harap unggah file CSV atau Excel.")
        return None

def preprocess_label(dt_tweet, df_tweet):
    required_columns_dt = ['label']
    required_columns_df = ['polarity']

    if not all(col in dt_tweet.columns for col in required_columns_dt):
        st.error(f"Kolom yang diperlukan tidak ada dalam dt_tweet. Diharapkan kolom: {required_columns_dt}")
        return None, None

    if not all(col in df_tweet.columns for col in required_columns_df):
        st.error(f"Kolom yang diperlukan tidak ada dalam df_tweet. Diharapkan kolom: {required_columns_df}")
        return None, None

    label_dict = {'neutral': 0, 'positive': 1, 'negative': -1}
    dt_tweet['label'] = dt_tweet['label'].map(label_dict)
    dt_tweet['label'] = dt_tweet['label'].map({-1: 'negative', 0: 'neutral', 1: 'positive'})
    dt_tweet['label'] = dt_tweet['label'].fillna('unknown')  # Replace NaN values with 'unknown'
    st.write("DataFrame Asli:")
    st.write(dt_tweet)

    polarity_dict = {'neutral': 0, 'positive': 1, 'negative': -1}
    polarity_dict1 = {'netral': 0, 'positif': 1, 'negatif': -1}
    df_tweet['polarity'] = df_tweet['polarity'].replace(polarity_dict1).replace(polarity_dict)
    df_tweet['polarity'] = df_tweet['polarity'].map({-1: 'negative', 0: 'neutral', 1: 'positive'})
    df_tweet['polarity'] = df_tweet['polarity'].fillna('unknown')  # Replace NaN values with 'unknown'
    st.write("DataFrame Hasil Analisis:")
    st.write(df_tweet)

    return dt_tweet, df_tweet

def evaluate_sentiment(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return accuracy, recall, precision, f1

def display_metrics(accuracy, recall, precision, f1):
    st.subheader("Metrics")
    st.write("Accuracy:", accuracy)
    st.write("Recall:", recall)
    st.write("Precision:", precision)
    st.write("F1 Score:", f1)

def evaluate_sentiment_macro(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    return accuracy, recall, precision, f1

def display_metrics_macro(accuracy, recall, precision, f1):
    st.subheader("Metrics")
    st.write("Accuracy:", accuracy)
    st.write("Recall:", recall)
    st.write("Precision:", precision)
    st.write("F1 Score:", f1)

def display_dataframe(dt_tweet, df_tweet):
    st.write(dt_tweet, df_tweet)

def main():
    st.markdown('<p style="font-family: Times New Roman; font-size: 32px; font-weight: bold;">Evaluasi Matrix InSet</p>', unsafe_allow_html=True)

    dt_tweet_file = st.file_uploader("Upload file asli", type=["csv", "xlsx"])
    df_tweet_file = st.file_uploader("Upload file hasil analisis", type=["csv", "xlsx"])

    if dt_tweet_file and df_tweet_file:
        # Load data
        dt_tweet = load_data(dt_tweet_file)
        df_tweet = load_data(df_tweet_file)

        if dt_tweet is None or df_tweet is None:
            return

        # Preprocess label
        dt_tweet, df_tweet = preprocess_label(dt_tweet, df_tweet)
        
        if dt_tweet is None or df_tweet is None:
            return

        # Display original data
        st.header("Data Asli dan Data Hasil Analisis")
        display_dataframe(dt_tweet, df_tweet)

        # Display pie chart for label distribution
        st.header("Distribusi Sentimen pada Data Asli")
        display_pie_chart(dt_tweet, 'label')

        # Display pie chart for polarity distribution
        st.header("Distribusi Sentimen pada Data Hasil Analisis Sentimen")
        if 'polarity' in df_tweet.columns:
            display_pie_chart(df_tweet, 'polarity')
        else:
            st.warning("Kolom 'polarity' tidak tersedia dalam data frame. Periksa data Anda.")
        
        # Calculate count of identified labels in dt_tweet
        dt_tweet_count = dt_tweet['label'].value_counts()
        st.header("Jumlah Data Sentimen pada Data Asli")
        st.write(dt_tweet_count)

        # Calculate count of identified labels in df_tweet
        if 'polarity' in df_tweet.columns:
            df_tweet_count = df_tweet['polarity'].value_counts()
            st.header("Jumlah Data Sentimen pada Data Hasil Analisis Sentimen")
            st.write(df_tweet_count)
        else:
            st.warning("Kolom 'polarity' tidak tersedia dalam data frame. Periksa data Anda.")

        # Evaluate sentiment analysis (macro)
        accuracy_macro, recall_macro, precision_macro, f1_macro = evaluate_sentiment_macro(dt_tweet['label'], df_tweet['polarity'])

        # Display evaluation metrics (macro)
        st.header("Evaluasi Analisis Sentimen Macro")
        display_metrics_macro(accuracy_macro, recall_macro, precision_macro, f1_macro)
        
        # Display evaluation metrics in percentages (macro)
        accuracy_macro_percent = round(accuracy_macro * 100, 2)
        recall_macro_percent = format(recall_macro * 100, ".2f")
        precision_macro_percent = format(precision_macro * 100, ".2f")
        f1_macro_percent = format(f1_macro * 100, ".2f")
        
        st.subheader("Metrics (Macro) dalam Persentase")
        st.write("Accuracy (%):", accuracy_macro_percent)
        st.write("Recall (%):", recall_macro_percent)
        st.write("Precision (%):", precision_macro_percent)
        st.write("F1 Score (%):", f1_macro_percent)

        # Evaluate sentiment analysis (weighted)
        accuracy, recall, precision, f1 = evaluate_sentiment(dt_tweet['label'], df_tweet['polarity'])

        # Display evaluation metrics (weighted)
        st.header("Evaluasi Analisis Sentimen Weighted")
        display_metrics(accuracy, recall, precision, f1)
        
        # Display evaluation metrics in percentages (weighted)
        accuracy_percent = round(accuracy * 100, 2)
        recall_percent = format(recall * 100, ".2f")
        precision_percent = format(precision * 100, ".2f")
        f1_percent = format(f1 * 100, ".2f")
        
        st.subheader("Metrics (Weighted) dalam Persentase")
        st.write("Accuracy (%):", accuracy_percent)
        st.write("Recall (%):", recall_percent)
        st.write("Precision (%):", precision_percent)
        st.write("F1 Score (%):", f1_percent)

if __name__ == "__main__":
    main()
