import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

# Fungsi untuk memuat data dari file CSV
def load_data(filename):
    data = pd.read_csv(filename)
    return data['tfidf'].tolist(), data['polarity'].tolist(), data

# Fungsi untuk membagi data menjadi set pelatihan dan pengujian (80-20)
def train_test_split(tfidf, labels, train_ratio=0.9):
    split_index = int(train_ratio * len(tfidf))
    return tfidf[:split_index], labels[:split_index], tfidf[split_index:], labels[split_index:]

# Fungsi untuk menghitung probabilitas prior
def calculate_prior(labels):
    total = len(labels)
    classes = list(set(labels))
    prior = {}
    for cls in classes:
        count = labels.count(cls)
        prior[cls] = count / total
    return prior

# Fungsi untuk menghitung rata-rata dan varians
def calculate_mean_variance(tfidf, labels):
    classes = list(set(labels))
    mean = {}
    variance = {}
    for cls in classes:
        tfidf_class = [tfidf[i] for i in range(len(tfidf)) if labels[i] == cls]
        mean[cls] = sum(tfidf_class) / len(tfidf_class)
        variance[cls] = sum((x - mean[cls]) ** 2 for x in tfidf_class) / len(tfidf_class)
    return mean, variance

# Fungsi untuk menghitung distribusi Gaussian
def gaussian(x, mean, variance):
    return (1 / (2 * variance) ** 0.5) * (2.71828 ** -((x - mean) ** 2 / (2 * variance)))

# Fungsi Naive Bayes untuk mengklasifikasikan data baru
def naive_bayes(tfidf, prior, mean, variance):
    posteriors = {}
    for cls in prior:
        likelihood = gaussian(tfidf, mean[cls], variance[cls])
        posteriors[cls] = prior[cls] * likelihood
    return max(posteriors, key=posteriors.get)

# Fungsi untuk menghitung akurasi
def calculate_accuracy(true_labels, predicted_labels):
    correct = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
    return correct / len(true_labels)

# Fungsi untuk membuat confusion matrix
def confusion_matrix(true_labels, predicted_labels):
    classes = list(set(true_labels))
    class_to_index = {cls: i for i, cls in enumerate(classes)}
    matrix = [[0] * len(classes) for _ in range(len(classes))]
    for true, pred in zip(true_labels, predicted_labels):
        matrix[class_to_index[true]][class_to_index[pred]] += 1
    return matrix, classes

# Fungsi untuk memproses TF-IDF untuk teks baru
def compute_tfidf_single(input_text, corpus):
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, min_df=1, max_df=0.8)
    tfidf_vectorizer.fit(corpus)
    tfidf_values = tfidf_vectorizer.transform([input_text]).toarray().flatten()
    total_tfidf = sum(tfidf_values)
    return total_tfidf

# Streamlit App
def main():
    st.title("Klasifikasi Model Naive Bayes")

    # Unggah file CSV
    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

    if uploaded_file:
        tfidf, labels, data = load_data(uploaded_file)

        # Check columns in the uploaded file
        st.write("Columns in the uploaded file:")
        st.write(data.columns)

        # Allow user to specify text column if 'text' column is not found
        text_column = 'kalimat'
        if text_column not in data.columns:
            st.error("CSV file must contain a 'kalimat' column for TF-IDF vectorization.")
            text_column = st.selectbox("Select the column that contains the text data:", data.columns)
            if not text_column:
                return

        # Membagi data
        train_tfidf, train_labels, test_tfidf, test_labels = train_test_split(tfidf, labels)

        # Menghitung probabilitas prior
        prior = calculate_prior(train_labels)

        # Menghitung rata-rata dan varians
        mean, variance = calculate_mean_variance(train_tfidf, train_labels)

        # Mengklasifikasikan data pengujian
        predictions = [naive_bayes(test_tfidf[i], prior, mean, variance) for i in range(len(test_tfidf))]

        # Menghitung akurasi
        accuracy = calculate_accuracy(test_labels, predictions)
        st.write(f"Akurasi: {accuracy * 100:.2f}%")

        # Membuat confusion matrix
        conf_matrix, classes = confusion_matrix(test_labels, predictions)
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
        data['polarity_predicted'] = [naive_bayes(tfidf[i], prior, mean, variance) for i in range(len(tfidf))]

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
            # Use the corpus from the uploaded data for fitting the vectorizer
            corpus = data[text_column].tolist()  # Ensure there's a 'text' column in the uploaded CSV
            new_input = compute_tfidf_single(input_text, corpus)
            new_prediction = naive_bayes(new_input, prior, mean, variance)
            st.write(f"Kalimat: {input_text}")
            st.write(f"Prediksi Polarity: {new_prediction}")

if __name__ == "__main__":
    main()
