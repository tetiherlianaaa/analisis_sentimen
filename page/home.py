import streamlit as st
    
def main():
    st.title("BERANDA")
    st.write("Selamat Datang Di Aplikasi Analisis Sentimen!!!")
    st.markdown('<p style="font-family: Times New Roman; font-size: 25px; font-weight: bold;">Pendeteksi Kata Slang</p>', unsafe_allow_html=True)

    # Menambahkan penjelasan mengenai aplikasi
    st.markdown("""
    
 Berikut adalah tahapan yang dilakukan dalam aplikasi ini:
                
    a. User mengunggah file kamus KBBI dalam format CSV, TXT, XLS, atau XLSX.
    b. 
    c. 

    """)
    
    st.markdown('<p style="font-family: Times New Roman; font-size: 25px; font-weight: bold;">Analisis Sentimen Menggunakan InSet</p>', unsafe_allow_html=True)

    # Menambahkan penjelasan mengenai aplikasi
    st.markdown("""
Aplikasi ini adalah program Python yang menggunakan library Streamlit, Pandas, NLTK, dan beberapa library lainnya. Aplikasi ini digunakan untuk melakukan analisis sentimen menggunakan InSet. Beberapa fitur yang tersedia di aplikasi ini antara lain:
                
    a. Memuat kamus positif dan negatif: Program memungkinkan pengguna untuk mengunggah file positive.csv dan negative.csv menggunakan st.file_uploader(). Setelah file-file tersebut diunggah, fungsi load_lexicon() akan digunakan untuk memuat kamus positif dan negatif dari file-file tersebut.
    b. Memuat file CSV untuk analisis sentimen: Program memungkinkan pengguna untuk mengunggah file CSV yang akan dianalisis sentimen menggunakan st.file_uploader(). Jika file tersebut diunggah, file CSV akan dibaca dan dimasukkan ke dalam DataFrame menggunakan pd.read_csv().
    c. Analisis sentimen menggunakan kamus: Program melakukan analisis sentimen pada teks yang ada dalam file CSV. Jika kolom 'tweet_clean' ada dalam DataFrame, analisis sentimen akan dilakukan pada kolom 'tweet_clean'. Jika kolom tersebut tidak ada, analisis sentimen akan dilakukan pada kolom 'text_clean'.
    d. Menampilkan hasil analisis sentimen: Program menampilkan hasil analisis sentimen dalam bentuk jumlah polaritas positif, negatif, dan netral menggunakan st.write(). Hasil ini memberikan gambaran umum tentang sebaran sentimen dalam data.
    e. Menampilkan sampel teks dan polaritas: Program menampilkan sampel teks dari setiap label sentimen (negatif, positif, netral) beserta nilai polaritas menggunakan st.subheader() dan st.write(). Ini memungkinkan pengguna untuk melihat contoh teks dengan nilai polaritas terkait.
    f. Export ke CSV dan XLSX: Program menyediakan tombol untuk mengexport hasil analisis sentimen ke dalam file CSV dan XLSX menggunakan st.button(). Jika tombol tersebut ditekan, DataFrame yang telah dianalisis akan disimpan ke dalam file CSV atau XLSX menggunakan to_csv() atau to_excel().

    """)

    st.markdown('<p style="font-family: Times New Roman; font-size: 32px; font-weight: bold;">Evaluasi Matrix InSet</p>', unsafe_allow_html=True)

    # Menambahkan penjelasan mengenai aplikasi
    st.markdown("""
Aplikasi ini adalah program Python yang menggunakan library Streamlit dan Pandas untuk melakukan evaluasi matrix InSet pada data hasil analisis sentimen:

    a. Mengunggah file sebelum analisis sentimen dan sesudah analisis sentimen: User dapat memilih file yang akan diunggah untuk dievaluasi matrixnya.
    b. Menampilkan data asli: Sistem menampilkan data sebelum analisis sentimen dan sesudah analisis sentimen yang sudah dimasukan.
    c. Menampilkan distribusi label pada data asli: Sistem menampilkan grafik pie yang menunjukkan distribusi label pada data sebelum analisis sentimen.
    d. Menampilkan distribusi sentimen pada data hasil analisis: Sistem menampilkan grafik pie yang menunjukkan distribusi sentimen pada data sesudah analisis sentimen.
    e. Menghitung evaluasi matrix weighted-average: Sistem menghitung dan menampilkan evaluasi matrix seperti akurasi, recall, presisi, dan F1 score untuk analisis sentimen menggunakan matrix weighted.
    f. Menampilkan evaluasi matrix weighted-average dalam persentase: Sistem menampilkan evaluasi matrix seperti akurasi, recall, presisi, dan F1 score dalam bentuk persentase.
    g. Menghitung evaluasi matrix macro-average: Sistem menghitung dan menampilkan evaluasi matrix seperti akurasi, recall, presisi, dan F1 score untuk analisis sentimen menggunakan matrix macro-averaged.
    h. Menampilkan evaluasi matrix macro-average dalam persentase: Sistem menampilkan evaluasi matrix seperti akurasi, recall, presisi, dan F1 score dalam bentuk persentase.

    """)
if __name__ == "__main__":
    main()