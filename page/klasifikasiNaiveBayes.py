import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Fungsi untuk memuat data dari file CSV
def load_data(filename):
    data = pd.read_csv(filename)
    return data['kalimat'].tolist(), data['polarity'].tolist(), data

# Fungsi untuk menghitung TF-IDF
def compute_tfidf(corpus):
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, min_df=1, max_df=0.8)
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    return tfidf_matrix, tfidf_vectorizer

# Streamlit App
def main():
    st.title("Klasifikasi Model Naive Bayes Multinominal")

    # Unggah file CSV
    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

    if uploaded_file:
        texts, labels, data = load_data(uploaded_file)

        # Check columns in the uploaded file
        st.write("Columns in the uploaded file:")
        st.write(data.columns)

        # Allow user to specify text column if 'kalimat' column is not found
        text_column = 'kalimat'
        if text_column not in data.columns:
            st.error("CSV file must contain a 'kalimat' column for TF-IDF vectorization.")
            text_column = st.selectbox("Select the column that contains the text data:", data.columns)
            if not text_column:
                return

        # Menghitung TF-IDF
        tfidf_matrix, tfidf_vectorizer = compute_tfidf(data[text_column])

        # Membagi data menjadi set pelatihan dan pengujian (80-20)
        X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, labels, test_size=0.2, random_state=42)

        # Inisialisasi dan latih model Multinomial Naive Bayes
        model = MultinomialNB()
        model.fit(X_train, y_train)

        # Mengklasifikasikan data pengujian
        predictions = model.predict(X_test)

        # Menghitung akurasi
        accuracy = accuracy_score(y_test, predictions)
        st.write(f"Akurasi: {accuracy * 100:.2f}%")

        # Membuat confusion matrix
        conf_matrix = confusion_matrix(y_test, predictions)
        classes = model.classes_
        st.write("Confusion Matrix:")
        st.write(pd.DataFrame(conf_matrix, index=classes, columns=classes))

        # Visualisasi confusion matrix
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis', ax=ax,
                    xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        st.pyplot(fig)

        # Memprediksi seluruh data dan menyimpan hasil di kolom baru
        full_predictions = model.predict(tfidf_matrix)
        data['polarity_predicted'] = full_predictions

        @st.cache_data
        def convert_df(df):
            return df.to_csv().encode('utf-8')

        csv = convert_df(data)
        st.download_button(
            label="Download hasil sebagai CSV",
            data=csv,
            file_name='data_tfidf_with_predictions_naivebayes.csv',
            mime='text/csv',
        )

        # Memprediksi kalimat baru
        st.subheader("Prediksi Kalimat Baru")
        input_text = st.text_input("Masukkan kalimat:")
        if input_text:
            new_input_tfidf = tfidf_vectorizer.transform([input_text])
            new_prediction = model.predict(new_input_tfidf)
            st.write(f"Kalimat: {input_text}")
            st.write(f"Prediksi Polarity: {new_prediction[0]}")

if __name__ == "__main__":
    main()