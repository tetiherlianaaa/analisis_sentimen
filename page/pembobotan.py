# Streamlit App
def main():
    st.title("TF-IDF Calculation")

    # Menyimpan informasi tentang file yang telah diunggah sebelumnya
    previous_uploaded_file = st.session_state.get('uploaded_file', None)

    # Unggah file CSV
    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

    if uploaded_file:
        # Memeriksa apakah file yang diunggah saat ini berbeda dengan yang sebelumnya
        if previous_uploaded_file is not None and uploaded_file.name == previous_uploaded_file.name:
            st.error("File yang diunggah harus berbeda dengan file yang telah diunggah sebelumnya.")
            return

        # Simpan informasi tentang file yang diunggah saat ini
        st.session_state.uploaded_file = uploaded_file

        df = read_csv_file(uploaded_file)

        if df is not None:
            output_df, tfidf_vectorizer = process_tfidf(df)
            
            st.write("Data dengan nilai TF-IDF:")
            st.write(output_df)

            # Unduh file hasil
            csv = output_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="Unduh CSV dengan TF-IDF", data=csv, file_name='labeled_sentiment_tfidf.csv', mime='text/csv')

    # Input teks tunggal untuk menghitung TF-IDF
    input_text = st.text_area("Masukkan teks untuk menghitung TF-IDF", "")

    if input_text and previous_uploaded_file:
        df_single = pd.DataFrame({'tweet_tokens_WSW': [input_text]})
        output_single, _ = process_tfidf(df_single)
        st.write(f"Total nilai TF-IDF untuk teks yang dimasukkan: {output_single['tfidf'][0]}")

if __name__ == "__main__":
    main()