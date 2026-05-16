import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
import re

# =========================
# LOAD FINAL DATASET
# =========================

df = pd.read_csv("../data/final_amazon_reviews.csv")

print("Dataset Loaded:", df.shape)

# =========================
# INITIALIZE VADER
# =========================

analyzer = SentimentIntensityAnalyzer()

# =========================
# SENTIMENT SCORE FUNCTION
# =========================

def get_sentiment_score(text):
    score = analyzer.polarity_scores(str(text))
    return score['compound']

# =========================
# APPLY SENTIMENT SCORE
# =========================

df['sentiment_score'] = df['clean_review'].apply(get_sentiment_score)

# =========================
# SENTIMENT LABEL
# =========================

def get_sentiment_label(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df['ai_sentiment'] = df['sentiment_score'].apply(get_sentiment_label)

# =========================
# SENTIMENT DISTRIBUTION
# =========================

print("\nAI Sentiment Distribution:")
print(df['ai_sentiment'].value_counts())

# =========================
# COMMON POSITIVE WORDS
# =========================

positive_reviews = df[df['ai_sentiment'] == 'Positive']

positive_text = " ".join(positive_reviews['clean_review'])

positive_words = re.findall(r'\b[a-z]{4,}\b', positive_text)

positive_common = Counter(positive_words).most_common(15)

print("\nTop Positive Words:")
print(positive_common)

# =========================
# COMMON NEGATIVE WORDS
# =========================

negative_reviews = df[df['ai_sentiment'] == 'Negative']

negative_text = " ".join(negative_reviews['clean_review'])

negative_words = re.findall(r'\b[a-z]{4,}\b', negative_text)

negative_common = Counter(negative_words).most_common(15)

print("\nTop Negative Words:")
print(negative_common)

# =========================
# HIDDEN DISSATISFACTION
# =========================

hidden_issues = df[
    (df['reviews.rating'] >= 4) &
    (df['ai_sentiment'] == 'Negative')
]

print("\nHidden Dissatisfaction Cases:")
print(hidden_issues[['name', 'reviews.rating', 'reviews.text']].head())

# =========================
# SAVE NLP DATASET
# =========================

df.to_csv("../data/nlp_amazon_reviews.csv", index=False)

print("\nNLP dataset saved successfully!")