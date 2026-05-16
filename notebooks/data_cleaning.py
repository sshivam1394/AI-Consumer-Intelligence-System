import pandas as pd
import numpy as np
import re

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("../data/1429_1.csv", low_memory=False)

print("Original Shape:", df.shape)

# =========================
# REMOVE DUPLICATES
# =========================

df.drop_duplicates(inplace=True)

print("After Removing Duplicates:", df.shape)

# =========================
# DROP USELESS COLUMNS
# =========================

columns_to_drop = [
    'reviews.userCity',
    'reviews.userProvince',
    'reviews.didPurchase',
    'reviews.id',
    'reviews.dateAdded',
    'reviews.dateSeen',
    'reviews.sourceURLs'
]

df.drop(columns=columns_to_drop, inplace=True)

print("\nColumns after dropping useless columns:")
print(df.columns)

# =========================
# HANDLE MISSING VALUES
# =========================

df['name'] = df['name'].fillna("Unknown Product")
df['reviews.text'] = df['reviews.text'].fillna("")
df['reviews.title'] = df['reviews.title'].fillna("")
df['reviews.rating'] = df['reviews.rating'].fillna(df['reviews.rating'].median())

# =========================
# CLEAN TEXT FUNCTION
# =========================

def clean_text(text):
    text = str(text).lower()

    # remove URLs
    text = re.sub(r"http\S+", "", text)

    # remove special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

# =========================
# APPLY TEXT CLEANING
# =========================

df['clean_review'] = df['reviews.text'].apply(clean_text)

# =========================
# CREATE REVIEW LENGTH
# =========================

df['review_length'] = df['clean_review'].apply(len)

# =========================
# BASIC DATASET INFO
# =========================

print("\nFinal Dataset Shape:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSample Clean Reviews:")
print(df['clean_review'].head())

# =========================
# SAVE CLEAN DATASET
# =========================

df.to_csv("../data/cleaned_reviews.csv", index=False)

print("\nCleaned dataset saved successfully!")