import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

@st.cache_resource
def load_recommender():
    class NetflixRecommender:
        def __init__(self):
            self.df = pd.read_pickle("df_metadata.pkl")
            self.norm_features = self._load_pickle("norm_features.pkl")
            self.title_to_indices = self._load_pickle("title_to_indices.pkl")

            try:
                self.nn = self._load_pickle("nn_model.pkl")
                self.use_ann = True
            except:
                self.use_ann = False
                self.cosine_sim = cosine_similarity(self.norm_features)

        def _load_pickle(self, path):
            with open(path, 'rb') as f:
                return pickle.load(f)

        def recommend(self, title: str, top_n: int = 5):
            title = title.lower()

            if title in self.title_to_indices:
                idx = self.title_to_indices[title][0]

                if self.use_ann:
                    distances, indices = self.nn.kneighbors(
                        self.norm_features[idx].reshape(1, -1), n_neighbors=top_n + 1
                    )
                    indices = indices[0][1:]
                    similarity_scores = 1 - distances[0][1:]
                else:
                    sim_scores = list(enumerate(self.cosine_sim[idx]))
                    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
                    indices = [i[0] for i in sim_scores]
                    similarity_scores = [round(i[1], 6) for i in sim_scores]
            else:
                popular = self.df.sort_values("release_year", ascending=False).head(100)
                vectorizer = TfidfVectorizer(max_features=500).fit(self.df["genres"])
                query_genre = vectorizer.transform([popular.iloc[0]["genres"]])
                sim_scores = cosine_similarity(query_genre, vectorizer.transform(popular["genres"]))[0]
                top_indices = np.argsort(sim_scores)[-top_n:][::-1]
                similarity_scores = sim_scores[top_indices]
                indices = popular.iloc[top_indices].index.tolist()

            recs = self.df.iloc[indices].copy()
            recs["similarity"] = similarity_scores
            return recs

    return NetflixRecommender()

st.set_page_config(
    page_title="🎬 Netflix Movie Recommender",
    page_icon="🎞️",
    layout="wide"
)

st.markdown("""
<style>
body {
    background-color: #0f0f0f;
    color: #ffffff;
}
.stTextInput>div>div>input {
    font-size: 1.1rem;
}
.stButton>button {
    background-color: #e50914;
    color: white;
    font-weight: bold;
}
.sim-bar {
    height: 10px;
    background: linear-gradient(90deg, #e50914 var(--val), #333 var(--val));
    border-radius: 4px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🎬 Netflix Title Recommender")
st.caption("Powered by hybrid similarity search and AI-driven suggestions.")

with st.sidebar:
    st.header("🔍 Search Settings")
    movie_title = st.text_input("Enter a Netflix title", placeholder="Stranger Things")
    top_n = st.slider("Number of Recommendations", 3, 10, 5)

recommender = load_recommender()

if movie_title:
    with st.spinner("Generating recommendations..."):
        results = recommender.recommend(movie_title, top_n=top_n)

    if "Error" in results.columns:
        st.warning(results["Error"].values[0])
    else:
        for _, row in results.iterrows():
            with st.expander(f"🎥 {row['title']} ({row['release_year']})", expanded=True):
                st.markdown(f"**Director:** {row['director']}")
                st.markdown(f"**Cast:** {row['cast']}")
                st.markdown(f"**Genres:** {row['genres']}")
                st.markdown(f"**Country:** {row['country']}")
                st.markdown(f"**Rating:** {row['rating']}")
                st.markdown(f"**Similarity Score:** `{round(row['similarity']*100, 2)}%`")
                st.markdown(f"<div class='sim-bar' style='--val:{row['similarity']*100}%;'></div>", unsafe_allow_html=True)
                st.markdown(f"**Description:** {row['description']}")
else:
    st.info("Enter a title on the left to get started.")

