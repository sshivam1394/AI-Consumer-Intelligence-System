import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px

# =========================
# LOAD NLP DATASET
# =========================

df = pd.read_csv("../data/nlp_amazon_reviews.csv")

print("Dataset Loaded Successfully!")

# =========================
# SENTIMENT DISTRIBUTION
# =========================

sentiment_counts = df['ai_sentiment'].value_counts()

print("\nSentiment Counts:")
print(sentiment_counts)

# =========================
# PLOTLY PIE CHART
# =========================

fig = px.pie(
    names=sentiment_counts.index,
    values=sentiment_counts.values,
    title="AI Sentiment Distribution",
    template='plotly_dark'
)

fig.show()

# =========================
# POSITIVE WORD CLOUD
# =========================

positive_reviews = df[df['ai_sentiment'] == 'Positive']

positive_text = " ".join(positive_reviews['clean_review'])

positive_wc = WordCloud(
    width=1200,
    height=600,
    background_color='black'
).generate(positive_text)

plt.figure(figsize=(15,7))
plt.imshow(positive_wc, interpolation='bilinear')
plt.axis('off')
plt.title("Positive Review Word Cloud")
plt.show()

# =========================
# NEGATIVE WORD CLOUD
# =========================

negative_reviews = df[df['ai_sentiment'] == 'Negative']

negative_text = " ".join(negative_reviews['clean_review'])

negative_wc = WordCloud(
    width=1200,
    height=600,
    background_color='black'
).generate(negative_text)

plt.figure(figsize=(15,7))
plt.imshow(negative_wc, interpolation='bilinear')
plt.axis('off')
plt.title("Negative Review Word Cloud")
plt.show()

# =========================
# HIDDEN DISSATISFACTION
# =========================

hidden_cases = df[
    (df['reviews.rating'] >= 4) &
    (df['ai_sentiment'] == 'Negative')
]

hidden_counts = hidden_cases['name'].value_counts().head(10)

fig2 = px.bar(
    x=hidden_counts.values,
    y=hidden_counts.index,
    orientation='h',
    title="Top Hidden Dissatisfaction Products",
    template='plotly_dark'
)

fig2.show()

print("\nVisualization Module Completed Successfully!")