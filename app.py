import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="🎬 Movie Recommendation System",
    page_icon="🎥",
    layout="wide"
)

# ==========================================
# Professional CSS
# ==========================================
st.markdown("""
<style>

.stApp{
    background: linear-gradient(to right,#141E30,#243B55);
}

h1,h2,h3{
    color:#FFD700;
    text-align:center;
}

section[data-testid="stSidebar"]{
    background:#111827;
}

div.stButton > button{
    background:#E50914;
    color:white;
    border:none;
    border-radius:12px;
    height:50px;
    width:100%;
    font-size:18px;
    font-weight:bold;
}

div.stButton > button:hover{
    background:#B20710;
}

.movie-card{
    background:#1E293B;
    padding:20px;
    border-radius:15px;
    margin-top:15px;
    margin-bottom:15px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.4);
}

.movie-title{
    color:#FFD700;
    font-size:24px;
    font-weight:bold;
}

.genre{
    color:#E5E7EB;
    font-size:16px;
}

.footer{
    text-align:center;
    color:white;
    margin-top:40px;
    font-size:15px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# Load Dataset
# ==========================================


@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    return movies, ratings


df_m, df_r = load_data()


# ==========================================
# Build Recommendation Model
# ==========================================

@st.cache_resource
def build_model(df_m):

    tfidf = TfidfVectorizer(
        token_pattern=r'[^|]+'
    )

    tfidf_matrix = tfidf.fit_transform(df_m["genres"])

    model = NearestNeighbors(
        metric="cosine",
        algorithm="brute"
    )

    model.fit(tfidf_matrix)

    return tfidf_matrix, model


tfidf_matrix, model = build_model(df_m)


# ==========================================
# Recommendation Function
# ==========================================

def recommend(movie_name):

    movie_name = movie_name.lower()

    idx = df_m[
        df_m["title"].str.lower() == movie_name
    ].index

    if len(idx) == 0:
        return []

    idx = idx[0]

    distances, indices = model.kneighbors(
        tfidf_matrix[idx],
        n_neighbors=11
    )

    recommendations = []

    for i in range(1, len(indices[0])):

        movie_idx = indices[0][i]

        recommendations.append({
            "title": df_m.iloc[movie_idx]["title"],
            "genre": df_m.iloc[movie_idx]["genres"]
        })

    return recommendations


# ==========================================
# Sidebar
# ==========================================
st.sidebar.title("🎬 Movie Recommendation")

menu = st.sidebar.radio(
    "Select Menu",
    ["🏠 Home", "📊 Analytics"]
)

# ==========================================
# HOME PAGE
# ==========================================

if menu == "🏠 Home":

    st.markdown("""
    <h1>🎬 Movie Recommendation System</h1>
    <h4 style='text-align:center;color:white;'>
    Find Similar Movies Using AI & Machine Learning
    </h4>
    <hr>
    """, unsafe_allow_html=True)

    movie_list = sorted(df_m["title"].unique())

    selected_movie = st.selectbox(
        "🔍 Select a Movie",
        movie_list
    )

    if st.button("⭐ Recommend Movies"):

        recommendations = recommend(selected_movie)

        if len(recommendations) == 0:

            st.error("Movie Not Found!")

        else:

            st.success(f"Top {len(recommendations)} Recommendations")

            for i, movie in enumerate(recommendations, start=1):

                st.markdown(f"""
                <div class="movie-card">

                <div class="movie-title">
                🎬 {movie['title']}
                </div>

                <br>

                <div class="genre">
                <b>Genre :</b> {movie['genre']}
                </div>

                </div>
                """, unsafe_allow_html=True)


# ==========================================
# ANALYTICS PAGE
# ==========================================

elif menu == "📊 Analytics":

    st.title("📊 Movie Analytics Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("🎬 Total Movies", len(df_m))

    with col2:
        st.metric("⭐ Total Ratings", len(df_r))

    st.markdown("---")

    st.subheader("User-Movie Ratings Heatmap")

    top_movies = df_r["movieId"].value_counts().head(30).index
    top_users = df_r["userId"].value_counts().head(30).index

    filtered = df_r[
        (df_r["movieId"].isin(top_movies)) &
        (df_r["userId"].isin(top_users))
    ]

    movie_matrix = filtered.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    )

    fig, ax = plt.subplots(figsize=(14, 8))

    sns.heatmap(
        movie_matrix,
        cmap="coolwarm",
        linewidths=0.5,
        ax=ax
    )

    ax.set_title("User-Movie Ratings Heatmap")

    st.pyplot(fig)

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown("""
<div class="footer">

<h4>🎬 Movie Recommendation System</h4>

<p>
Developed using <b>Python</b>, <b>Streamlit</b>,
<b>Scikit-Learn</b> and <b>Pandas</b>
</p>

<p>Made with ❤️ for Machine Learning Project</p>

</div>
""", unsafe_allow_html=True)
