import streamlit as st

def main():
    st.title("Klasifikasi Model K-Nearest Neighbors")

    file_uploaded = False
    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

    if uploaded_file is not None:
        file_uploaded = True
        data = pd.read_csv(uploaded_file)

        st.write("Columns in the uploaded file:")
        st.write(data.columns)

        # Check if 'text' column is present, allow user to select if not found
        text_column = 'kalimat'
        if text_column not in data.columns:
            st.error("CSV file must contain a 'kalimat' column for TF-IDF vectorization.")
            text_column = st.selectbox("Select the column that contains the text data:", data.columns)
            if not text_column:
                return

        tfidf, labels = data['tfidf'].tolist(), data['polarity'].tolist()

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

        data['polarity_predicted'] = knn(train_tfidf, train_labels, tfidf, k=k_value)

        @st.cache_data
        def convert_df(df):
            return df.to_csv().encode('utf-8')

        csv = convert_df(data)
        st.download_button(label="Download hasil sebagai CSV", data=csv, file_name='data_tfidf_with_predictions_knn.csv', mime='text/csv')

    if file_uploaded:
        st.subheader("Prediksi Kalimat Baru")
        input_text = st.text_input("Masukkan kalimat:")
        if input_text:
            corpus = data[text_column].tolist()
            new_input = compute_tfidf_single(input_text, corpus)
            new_prediction = knn(train_tfidf, train_labels, [new_input], k=k_value)
            st.write(f"Kalimat: {input_text}")
            st.write(f"Prediksi Polarity: {new_prediction[0]}")

if __name__ == "__main__":
    main()
