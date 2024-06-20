import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Fungsi untuk memuat data dari file CSV
def load_data(filename):
    data = pd.read_csv(filename)
    return data['tfidf'].tolist(), data['polarity'].tolist(), data

# Fungsi untuk membagi data menjadi set pelatihan dan pengujian (80-20)
def train_test_split(tfidf, labels, train_ratio=0.8):
    split_index = int(train_ratio * len(tfidf))
    return tfidf[:split_index], labels[:split_index], tfidf[split_index:], labels[split_index:]

# Fungsi untuk menghitung jarak antara dua titik
def calculate_distance(x1, x2):
    return abs(x1 - x2)

# Fungsi untuk menghitung frekuensi kelas
def most_common(labels):
    frequency = {}
    for i in range(len(labels)):
        label = labels[i]
        if label in frequency:
            frequency[label] += 1
        else:
            frequency[label] = 1
    return max(frequency, key=frequency.get)

# Fungsi K-Nearest Neighbors untuk mengklasifikasikan data baru
def knn(train_tfidf, train_labels, test_tfidf, k=3):
    predictions = []
    for i in range(len(test_tfidf)):
        test_point = test_tfidf[i]
        distances = [calculate_distance(test_point, train_tfidf[j]) for j in range(len(train_tfidf))]
        distance_indices = sorted(range(len(distances)), key=lambda idx: distances[idx])[:k]
        k_nearest_labels = [train_labels[idx] for idx in distance_indices]
        predictions.append(most_common(k_nearest_labels))
    return predictions

# Fungsi untuk menghitung akurasi
def calculate_accuracy(true_labels, predicted_labels):
    return sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred) / len(true_labels)

# Fungsi untuk membuat confusion matrix
def create_confusion_matrix(true_labels, predicted_labels):
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
    return sum(tfidf_values)

# Streamlit App
def main():
    st.title("Klasifikasi Model K-Nearest Neighbors")

    # Cek apakah ada file CSV yang telah disimpan sebelumnya
    saved_file_path = "saved_data.csv"
    if os.path.exists(saved_file_path):
        uploaded_file = None
    else:
        uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

    if uploaded_file:
        uploaded_data = pd.read_csv(uploaded_file)
        uploaded_data.to_csv(saved_file_path, index=False)
    elif os.path.exists(saved_file_path):
        uploaded_data = pd.read_csv(saved_file_path)
    else:
        st.error("Belum ada file CSV yang diunggah atau disimpan sebelumnya.")
        return

    st.write("Columns in the uploaded file:")
    st.write(uploaded_data.columns)

    # Check if 'text' column is present, allow user to select if not found
    text_column = 'kalimat'
    if text_column not in uploaded_data.columns:
        st.error("CSV file must contain a 'kalimat' column for TF-IDF vectorization.")
        text_column = st.selectbox("Select the column that contains the text data:", uploaded_data.columns)
        if not text_column:
            return

    tfidf, labels = uploaded_data['tfidf'].tolist(), uploaded_data['polarity'].tolist()

    train_tfidf, train_labels, test_tfidf, test_labels = train_test_split(tfidf, labels)

    # Tambahkan slider untuk memilih nilai K
    k_value = st.slider("Pilih nilai K untuk KNN:", min_value=1, max_value=10, value=3, step=1)

    predictions = knn(train_tfidf, train_labels, test_tfidf, k=k_value)
    accuracy = calculate_accuracy(test_labels, predictions)
    st.write(f"Akurasi: {accuracy * 100:.2f}%")

    conf_matrix, classes = create_confusion_matrix(test_labels, predictions)
    st.write("Confusion Matrix:")
    st.write(pd.DataFrame(conf_matrix, index=classes, columns=classes))

    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis', ax=ax, xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    st.pyplot(fig)

    uploaded_data['polarity_predicted'] = knn(train_tfidf, train_labels, tfidf, k=k_value)

    @st.cache_data
    def convert_df(df):
        return df.to_csv().encode('utf-8')

    csv = convert_df(uploaded_data)
    st.download_button(label="Download hasil sebagai CSV", data=csv, file_name='data_tfidf_with_predictions_knn.csv', mime='text/csv')

    st.subheader("Prediksi Kalimat Baru")
    input_text = st.text_input("Masukkan kalimat:")
    if input_text:
        corpus = uploaded_data[text_column].tolist()
        new_input = compute_tfidf_single(input_text, corpus)
        new_prediction = knn(train_tfidf, train_labels, [new_input], k=k_value)
        st.write(f"Kalimat: {input_text}")
        st.write(f"Prediksi Polarity: {new_prediction[0]}")

if __name__ == "__main__":
    main()