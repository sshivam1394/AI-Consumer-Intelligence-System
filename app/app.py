import streamlit as st
import pandas as pd
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="AI Product Intelligence Dashboard",
    layout="wide"
)

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("data/nlp_amazon_reviews.csv")

# =========================
# TITLE
# =========================

st.title("🚀 AI-Powered Consumer Intelligence Dashboard")
st.markdown("""
<style>
.big-font {
    font-size:28px !important;
    font-weight:700;
    color:#c084fc;
}
.metric-card {
    background-color:#1e293b;
    padding:15px;
    border-radius:15px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">AI-Powered Consumer Intelligence System</p>', unsafe_allow_html=True)

st.markdown("### Advanced Product Analytics & Sentiment Intelligence")

# =========================
# SIDEBAR FILTERS
# =========================

st.sidebar.header("Dashboard Filters")

brand_filter = st.sidebar.selectbox(
    "Select Brand",
    options=df['brand'].unique()
)

filtered_df = df[df['brand'] == brand_filter]

# =========================
# KPI SECTION
# =========================

total_reviews = filtered_df.shape[0]
avg_rating = round(filtered_df['reviews.rating'].mean(), 2)

positive_percent = round(
    (
        filtered_df[filtered_df['ai_sentiment'] == 'Positive'].shape[0]
        / total_reviews
    ) * 100,
    2
)

top_category = filtered_df['main_category'].mode()[0]

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Reviews", total_reviews)
col2.metric("Average Rating", avg_rating)
col3.metric("Positive Sentiment %", f"{positive_percent}%")
col4.metric("Top Category", top_category)

# =========================
# SENTIMENT DISTRIBUTION
# =========================

st.subheader("📊 AI Sentiment Distribution")

sentiment_counts = filtered_df['ai_sentiment'].value_counts()

fig1 = px.pie(
    names=sentiment_counts.index,
    values=sentiment_counts.values,
    template='plotly_dark'
)

st.plotly_chart(fig1, use_container_width=True)

# =========================
# RATING DISTRIBUTION
# =========================

st.subheader("⭐ Customer Rating Distribution")

fig2 = px.histogram(
    filtered_df,
    x='reviews.rating',
    nbins=5,
    template='plotly_dark'
)

st.plotly_chart(fig2, use_container_width=True)

# =========================
# PRODUCT STRATEGY
# =========================

st.subheader("📈 Product Strategy Distribution")

strategy_counts = filtered_df['product_strategy'].value_counts()

fig3 = px.bar(
    x=strategy_counts.index,
    y=strategy_counts.values,
    template='plotly_dark',
    color=strategy_counts.index
)

st.plotly_chart(fig3, use_container_width=True)

# =========================
# LIVE SENTIMENT PREDICTION
# =========================

st.subheader("🤖 Live AI Sentiment Prediction")

user_review = st.text_area("Enter Customer Review")

analyzer = SentimentIntensityAnalyzer()

if st.button("Analyze Sentiment"):

    score = analyzer.polarity_scores(user_review)['compound']

    if score >= 0.05:
        sentiment = "Positive 😀"
    elif score <= -0.05:
        sentiment = "Negative 😡"
    else:
        sentiment = "Neutral 😐"

    st.success(f"Predicted Sentiment: {sentiment}")

# =========================
# RECOMMENDATION SYSTEM
# =========================

st.subheader("🛍️ Product Recommendation Engine")

product_df = df.drop_duplicates(subset='name')

product_df['content'] = (
    product_df['name'].astype(str) + " " +
    product_df['main_category'].astype(str) + " " +
    product_df['brand'].astype(str)
)

tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(product_df['content'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(
    product_df.index,
    index=product_df['name']
).drop_duplicates()

selected_product = st.selectbox(
    "Select Product",
    product_df['name'].values
)

def recommend_products(product_name):

    idx = indices[product_name]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:6]

    product_indices = [i[0] for i in sim_scores]

    return product_df[['name', 'brand', 'main_category']].iloc[product_indices]

if st.button("Get Recommendations"):

    recommendations = recommend_products(selected_product)

    st.dataframe(recommendations)

# =========================
# FOOTER
# =========================

st.markdown("---")
st.markdown("AI-Powered Consumer Intelligence System | MBA Data Science Project")