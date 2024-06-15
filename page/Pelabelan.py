import streamlit as st
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import csv
import io
import nltk

# Download NLTK data
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
    # Replace non-alphanumeric characters with space and lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text).lower()
    
    # Remove URLs, mentions, hashtags
    text = re.sub(r'http\S+|@\S+|#\S+', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stop words
    tokens_without_stopwords = [word for word in tokens if word not in stop_words]

    return text, tokens, tokens_without_stopwords

# Function for sentiment analysis
def sentiment_analysis_lexicon_indonesia(tokens, lexicon_positive, lexicon_negative):
    score = 0
    for word in tokens:
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
        
        # Initialize lists
        casefoldings = []
        tokenizes = []
        tweet_tokens_WSWs = []
        polarity_scores = []
        labels = []

        for sentence in df['Content']:
            if isinstance(sentence, str):
                casefolding, tokens, tokens_without_stopwords = preprocess_text(sentence)
                score, polarity = sentiment_analysis_lexicon_indonesia(tokens_without_stopwords, lexicon_positive, lexicon_negative)
            else:
                casefolding = ''
                tokens = []
                tokens_without_stopwords = []
                score = 0
                polarity = 'neutral'
            
            casefoldings.append(casefolding)
            tokenizes.append(tokens)
            tweet_tokens_WSWs.append(tokens_without_stopwords)
            polarity_scores.append(score)
            labels.append(polarity)

        # Ensure the lists match the length of the DataFrame
        assert len(casefoldings) == len(df), f"Length of casefoldings ({len(casefoldings)}) does not match DataFrame rows ({len(df)})"
        assert len(tokenizes) == len(df), f"Length of tokenizes ({len(tokenizes)}) does not match DataFrame rows ({len(df)})"
        assert len(tweet_tokens_WSWs) == len(df), f"Length of tweet_tokens_WSWs ({len(tweet_tokens_WSWs)}) does not match DataFrame rows ({len(df)})"
        assert len(polarity_scores) == len(df), f"Length of polarity_scores ({len(polarity_scores)}) does not match DataFrame rows ({len(df)})"
        assert len(labels) == len(df), f"Length of labels ({len(labels)}) does not match DataFrame rows ({len(df)})"

        df['casefolding'] = casefoldings
        df['tokenize'] = tokenizes
        df['tweet_tokens_WSW'] = tweet_tokens_WSWs
        df['polarity_score'] = polarity_scores
        df['polarity'] = labels

        # Remove unnecessary columns
        columns_to_remove = ['Score', 'At', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6']
        df.drop(columns=[col for col in columns_to_remove if col in df.columns], inplace=True)

        # Display DataFrame
        st.write(df)

        # Download result file as CSV
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download CSV", data=csv_data, file_name='labeled_sentiment_fixed.csv', mime='text/csv')

        # Create buffer to store Excel data
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False)
        excel_buffer.seek(0)
        st.download_button(label="Download Excel", data=excel_buffer, file_name='labeled_sentiment_fixed.xlsx')

    # Single text input for sentiment analysis
    input_text = st.text_area("Enter text for sentiment analysis", "")

    if input_text and uploaded_positive_lexicon and uploaded_negative_lexicon:
        _, tokens, tokens_without_stopwords = preprocess_text(input_text)
        score, polarity = sentiment_analysis_lexicon_indonesia(tokens_without_stopwords, lexicon_positive, lexicon_negative)
        
        st.write("Analysis Result:")
        st.write("Casefolding:", input_text.lower())
        st.write("Tokenize:", tokens)
        st.write("Tokens without Stop Words:", tokens_without_stopwords)
        st.write("Polarity Score:", score)
        st.write("Polarity:", polarity)

if __name__ == '__main__':
    main()
