import pandas as pd

# =========================
# LOAD CLEANED DATASET
# =========================

df = pd.read_csv("../data/cleaned_reviews.csv")

print("Original Shape:", df.shape)

# =========================
# REMOVE UNKNOWN PRODUCTS
# =========================

df = df[df['name'] != 'Unknown Product']

print("After Removing Unknown Products:", df.shape)

# =========================
# CREATE MAIN CATEGORY
# =========================

df['main_category'] = df['categories'].apply(
    lambda x: str(x).split(",")[0]
)

# =========================
# CREATE SENTIMENT LABEL
# BASED ON RATING
# =========================

def rating_sentiment(rating):
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"

df['rating_sentiment'] = df['reviews.rating'].apply(rating_sentiment)

# =========================
# CREATE PRODUCT REVIEW COUNT
# =========================

review_counts = df['name'].value_counts()

df['product_review_count'] = df['name'].map(review_counts)

# =========================
# CREATE PRODUCT HEALTH SCORE
# =========================

df['health_score'] = (
    (df['reviews.rating'] * 0.7) +
    (df['product_review_count'] / df['product_review_count'].max()) * 5 * 0.3
)

# =========================
# CLASSIFY PRODUCTS
# =========================

def classify_product(score):
    if score >= 4.5:
        return "Scale"
    elif score >= 3.5:
        return "Fix"
    else:
        return "Drop"

df['product_strategy'] = df['health_score'].apply(classify_product)

# =========================
# SHOW RESULTS
# =========================

print("\nMain Categories:")
print(df['main_category'].value_counts().head())

print("\nSentiment Distribution:")
print(df['rating_sentiment'].value_counts())

print("\nProduct Strategy Distribution:")
print(df['product_strategy'].value_counts())

print("\nTop Health Scores:")
print(
    df[['name', 'health_score']]
    .sort_values(by='health_score', ascending=False)
    .head(10)
)

# =========================
# SAVE FINAL DATASET
# =========================

df.to_csv("../data/final_amazon_reviews.csv", index=False)

print("\nFinal dataset saved successfully!")