import pandas as pd
import plotly.express as px

# =========================
# LOAD CLEANED DATASET
# =========================

df = pd.read_csv("../data/cleaned_reviews.csv")

print("Dataset Loaded Successfully!")
print(df.shape)

# =========================
# TOP BRANDS ANALYSIS
# =========================

top_brands = df['brand'].value_counts().head(10)

print("\nTop Brands:")
print(top_brands)

# =========================
# CATEGORY ANALYSIS
# =========================

top_categories = df['categories'].value_counts().head(10)

print("\nTop Categories:")
print(top_categories)

# =========================
# AVERAGE RATING BY BRAND
# =========================

brand_ratings = (
    df.groupby('brand')['reviews.rating']
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

print("\nAverage Rating by Brand:")
print(brand_ratings)

# =========================
# MOST REVIEWED PRODUCTS
# =========================

top_products = df['name'].value_counts().head(10)

print("\nMost Reviewed Products:")
print(top_products)

# =========================
# VISUALIZATION 1
# TOP BRANDS
# =========================

fig1 = px.bar(
    x=top_brands.index,
    y=top_brands.values,
    title="Top 10 Brands by Review Count",
    labels={'x': 'Brand', 'y': 'Number of Reviews'},
    template='plotly_dark'
)

fig1.show()

# =========================
# VISUALIZATION 2
# BRAND RATINGS
# =========================

fig2 = px.bar(
    x=brand_ratings.index,
    y=brand_ratings.values,
    title="Top Brands by Average Rating",
    labels={'x': 'Brand', 'y': 'Average Rating'},
    template='plotly_dark',
    color=brand_ratings.values
)

fig2.show()

# =========================
# VISUALIZATION 3
# RATING DISTRIBUTION
# =========================

fig3 = px.histogram(
    df,
    x='reviews.rating',
    nbins=5,
    title="Customer Rating Distribution",
    template='plotly_dark'
)

fig3.show()

print("\nMarket Analysis Completed Successfully!")