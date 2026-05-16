import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("../data/nlp_amazon_reviews.csv")

print("Dataset Loaded:", df.shape)

# =========================
# REMOVE DUPLICATES
# =========================

df = df.drop_duplicates(subset='name')

print("After Removing Duplicate Products:", df.shape)

# =========================
# CREATE CONTENT FEATURES
# =========================

df['content'] = (
    df['name'].astype(str) + " " +
    df['main_category'].astype(str) + " " +
    df['brand'].astype(str)
)

# =========================
# TF-IDF VECTORIZATION
# =========================

tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(df['content'])

print("\nTF-IDF Matrix Shape:")
print(tfidf_matrix.shape)

# =========================
# COSINE SIMILARITY
# =========================

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# =========================
# PRODUCT INDEX MAPPING
# =========================

indices = pd.Series(df.index, index=df['name']).drop_duplicates()

# =========================
# RECOMMENDATION FUNCTION
# =========================

def recommend_products(product_name, num_recommendations=5):

    if product_name not in indices:
        print("Product not found!")
        return

    idx = indices[product_name]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:num_recommendations + 1]

    product_indices = [i[0] for i in sim_scores]

    recommendations = df[['name', 'brand', 'main_category']].iloc[product_indices]

    return recommendations

# =========================
# TEST RECOMMENDATION
# =========================

sample_product = df['name'].iloc[0]

print("\nSample Product:")
print(sample_product)

print("\nRecommended Products:")

recommendations = recommend_products(sample_product)

print(recommendations)

print("\nRecommendation System Completed Successfully!")