import argparse
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


class NetflixRecommender:
    def __init__(self,
                 metadata_path: str,
                 nn_path: str,
                 norm_feat_path: str,
                 index_map_path: str):
       
        self.df = pd.read_pickle(metadata_path)
        self.norm_features = self._load_pickle(norm_feat_path)
        self.title_to_indices = self._load_pickle(index_map_path)

        try:
            self.nn = self._load_pickle(nn_path)
            self.use_ann = True
        except Exception:
            self.use_ann = False
            print("[INFO] NearestNeighbors not available. Using cosine similarity.")
            self.cosine_sim = cosine_similarity(self.norm_features)

    def _load_pickle(self, path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def recommend(self, title: str, top_n: int = 5, fallback_to_popular: bool = True) -> pd.DataFrame:
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
            if not fallback_to_popular:
                return pd.DataFrame({'Error': [f"Title '{title}' not found."]})

            print(f"[INFO] Title '{title}' not found. Recommending popular items by genre...")
            popular = self.df.sort_values('release_year', ascending=False).head(100)

            if len(popular) == 0:
                return pd.DataFrame({'Error': ["No fallback items found."]})

            genre_vectorizer = TfidfVectorizer(max_features=500).fit(self.df['genres'])
            gen_vec_popular = genre_vectorizer.transform(popular['genres'])
            query_idx = np.random.choice(len(popular))
            query_genre_vec = genre_vectorizer.transform([popular.iloc[query_idx]['genres']])
            sim_scores = cosine_similarity(query_genre_vec, gen_vec_popular)[0]

            top_indices = np.argsort(sim_scores)[-top_n:][::-1]
            similarity_scores = sim_scores[top_indices]
            indices = popular.iloc[top_indices].index.tolist()

        recommendations = self.df.iloc[indices].copy()
        recommendations['similarity'] = similarity_scores

        return recommendations[[
            'title', 'director', 'cast', 'genres', 'country',
            'release_year', 'rating', 'description', 'similarity'
        ]]


def main():
    parser = argparse.ArgumentParser(description="Netflix Title Recommender")
    parser.add_argument("title", type=str, help="Title of the movie/show")
    parser.add_argument("--top_n", type=int, default=5, help="Number of recommendations to return")
    parser.add_argument("--metadata", default="df_metadata.pkl")
    parser.add_argument("--nn_model", default="nn_model.pkl")
    parser.add_argument("--norm_features", default="norm_features.pkl")
    parser.add_argument("--index_map", default="title_to_indices.pkl")

    args = parser.parse_args()

    recommender = NetflixRecommender(
        metadata_path=args.metadata,
        nn_path=args.nn_model,
        norm_feat_path=args.norm_features,
        index_map_path=args.index_map
    )

    result_df = recommender.recommend(args.title, top_n=args.top_n)
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()