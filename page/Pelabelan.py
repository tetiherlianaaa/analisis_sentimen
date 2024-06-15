import streamlit as st
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import csv
import io
import nltk

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('indonesian'))

# Function to load lexicon from CSV file
def load_lexicon(file):
    lexicon = {}
    # Wrap BytesIO file with TextIOWrapper
    file = io.TextIOWrapper(file, encoding='utf-8')
    reader = csv.reader(file, delimiter=',')
    for row in reader:
        if row[0] != 'word':  # skip header row
            lexicon[row[0]] = int(row[1])
    return lexicon

# Function to clean and process text
def preprocess_text(text):
    cleaned_text = ''
    for char in text:
        if (48 <= ord(char) <= 57) or (65 <= ord(char) <= 90) or (97 <= ord(char) <= 122):
            cleaned_text += char
        else:
            cleaned_text += ' '  # Replace non-alphanumeric characters with space
    text = cleaned_text
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    
    casefolding = text.lower()
    handling_tandabaca = re.sub(r'[^\w\s]', '', casefolding)
    handling_urls_mentions_hashtags = re.sub(r'http\S+|@\S+|#\S+', '', handling_tandabaca)
    tokenize = word_tokenize(handling_urls_mentions_hashtags)
    tweet_tokens_WSW = [word for word in tokenize if word not in stop_words]

    return casefolding, handling_tandabaca, handling_urls_mentions_hashtags, tokenize, tweet_tokens_WSW

# Function for sentiment analysis
def sentiment_analysis_lexicon_indonesia(text, lexicon_positive, lexicon_negative):
    score = 0
    for word in text:
        if word in lexicon_positive:
            score += lexicon_positive[word]
        if word in lexicon_negative:
            score += lexicon_negative[word]
    if score > 0:
        polarity = 'positive'
    elif score < 0:
        polarity = 'negative'
    else:
        polarity = 'neutral'
    return score, polarity

def main():
    st.title("Pelabelan InSet")

    # Upload Excel file
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

    # Upload positive lexicon file
    uploaded_positive_lexicon = st.file_uploader("Upload positive lexicon (CSV)", type=["csv"])

    # Upload negative lexicon file
    uploaded_negative_lexicon = st.file_uploader("Upload negative lexicon (CSV)", type=["csv"])

    if uploaded_file and uploaded_positive_lexicon and uploaded_negative_lexicon:
        # Load data from uploaded file
        df = pd.read_excel(uploaded_file)

        # Limit data to 3000 rows
        df = df.head(3000)
        
        # Load positive and negative lexicon
        lexicon_positive = load_lexicon(uploaded_positive_lexicon)
        lexicon_negative = load_lexicon(uploaded_negative_lexicon)
        
        # Process each sentence
        sentences = df['Content'].tolist()
        casefoldings = []
        handling_tandabacas = []
        handling_urls_mentions_hashtags_list = []
        tokenizes = []
        tweet_tokens_WSWs = []
        polarity_scores = []
        labels = []

        for sentence in sentences:
            if isinstance(sentence, str):  # Ensure processing only strings
                casefolding, handling_tandabaca, handling_urls_mentions_hashtags, tokenize, tweet_tokens_WSW = preprocess_text(sentence)
                score, polarity = sentiment_analysis_lexicon_indonesia(tweet_tokens_WSW, lexicon_positive, lexicon_negative)
            else:
                casefolding = handling_tandabaca = handling_urls_mentions_hashtags = ''
                tokenize = tweet_tokens_WSW = []
                score = 0
                polarity = 'neutral'

            casefoldings.append(casefolding)
            handling_tandabacas.append(handling_tandabaca)
            handling_urls_mentions_hashtags_list.append(handling_urls_mentions_hashtags)
            tokenizes.append(tokenize)
            tweet_tokens_WSWs.append(tweet_tokens_WSW)
            polarity_scores.append(score)
            labels.append(polarity)

        df['casefolding'] = casefoldings
        df['handling_tandabaca'] = handling_tandabacas
        df['handling_urls_mentions_hashtags'] = handling_urls_mentions_hashtags_list
        df['tokenize'] = tokenizes
        df['tweet_tokens_WSW'] = tweet_tokens_WSWs
        df['polarity_score'] = polarity_scores
        df['polarity'] = labels

        # Remove unnecessary columns
        if 'Score' in df.columns: df.drop(columns=['Score'], inplace=True)
        if 'At' in df.columns: df.drop(columns=['At'], inplace=True)
        if 'Unnamed: 4' in df.columns: df.drop(columns=['Unnamed: 4'], inplace=True)
        if 'Unnamed: 5' in df.columns: df.drop(columns=['Unnamed: 5'], inplace=True)
        if 'Unnamed: 6' in df.columns: df.drop(columns=['Unnamed: 6'], inplace=True)

        # Display DataFrame
        st.write(df)

        # Download result file
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download CSV", data=csv_data, file_name='labeled_sentiment_fixed.csv', mime='text/csv')

        # Create buffer to store Excel data
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False)
        excel_buffer.seek(0)
        st.download_button(label="Download Excel", data=excel_buffer, file_name='labeled_sentiment_fixed.xlsx')

    # Single text input for sentiment analysis
    input_text = st.text_area("Enter text for sentiment analysis", "")

    if input_text:
        casefolding, handling_tandabaca, handling_urls_mentions_hashtags, tokenize, tweet_tokens_WSW = preprocess_text(input_text)
        score, polarity = sentiment_analysis_lexicon_indonesia(tweet_tokens_WSW, lexicon_positive, lexicon_negative)
        
        st.write("Analysis Result:")
        st.write("Casefolding:", casefolding)
        st.write("Handling Punctuation:", handling_tandabaca)
        st.write("Handling URLs, Mentions, Hashtags:", handling_urls_mentions_hashtags)
        st.write("Tokenize:", tokenize)
        st.write("Tokens without Stop Words:", tweet_tokens_WSW)
        st.write("Polarity Score:", score)
        st.write("Polarity:", polarity)

if __name__ == '__main__':
    main()
