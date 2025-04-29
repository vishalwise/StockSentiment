import streamlit as st
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from fuzzywuzzy import process
import nltk
nltk.download('vader_lexicon')

# Load the sector-to-stock mapping CSV
df = pd.read_csv("ind_nifty500list.csv")  # Adjust the path if needed

# Initialize Sentiment Analyzer
sid = SentimentIntensityAnalyzer()

# News API Key
API_KEY = "467901d6f0884b8dbd1fde2bab77061f"  # Replace with your key from newsapi.org

# Function to get sector from stock name
def get_sector_from_stock(stock_name):
    match, score = process.extractOne(stock_name, df["Stock"].tolist())
    if score > 70:  # Confidence threshold
        sector = df[df["Stock"] == match]["Sector"].values[0]
        return sector
    return None

def get_news(query):
    url = (
        f"https://newsapi.org/v2/everything?q={query}&"
        f"sortBy=publishedAt&language=en&apiKey={API_KEY}"
    )
    response = requests.get(url)
    articles = response.json().get("articles", [])
    return articles

def analyze_sentiment(text):
    score = sid.polarity_scores(text)
    return score["compound"]

def summarize_sentiment(articles):
    sentiments = {"positive": 0, "neutral": 0, "negative": 0}
    for article in articles[:10]:
        title = article["title"]
        sentiment_score = analyze_sentiment(title)
        sentiment_label = (
            "positive" if sentiment_score > 0.2 else
            "negative" if sentiment_score < -0.2 else
            "neutral"
        )
        sentiments[sentiment_label] += 1
    return sentiments

# Streamlit UI
st.title("📊 Indian Stock News Sentiment Analyzer")

stock = st.text_input("Enter Stock Name (e.g., Infosys, Reliance)")

if st.button("Analyze"):
    if not API_KEY or API_KEY == "YOUR_NEWS_API_KEY":
        st.error("Please set your NewsAPI key in the code.")
    elif stock:
        with st.spinner("Analyzing stock and sector sentiment..."):
            stock_articles = get_news(f"{stock} stock")
            stock_sentiment = summarize_sentiment(stock_articles)

            # Get sector for the stock
            sector = get_sector_from_stock(stock)
            if sector:
                sector_articles = get_news(f"{sector} sector India")
                sector_sentiment = summarize_sentiment(sector_articles)
            else:
                sector_articles = []
                sector_sentiment = None

            # Display results
            st.subheader("📈 Stock Sentiment")
            st.write(f"**Stock:** {stock}")
            st.write(stock_sentiment)

            if sector_sentiment:
                st.subheader("🏭 Sector Sentiment")
                st.write(f"**Sector:** {sector}")
                st.write(sector_sentiment)
            else:
                st.warning("Could not detect sector for this stock.")
