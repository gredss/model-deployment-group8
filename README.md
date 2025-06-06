# model-deployment-group8

---

# 📽️ Content-Based Movie Recommendation System (Netflix Titles)

This project is a **content-based filtering recommendation engine** that suggests Netflix titles similar to a given input based solely on item metadata. It leverages **TF-IDF vectorization**, **cosine similarity**, and optional **approximate nearest neighbor search** to recommend similar content.

---

## 📌 Features

* Preprocessing with lemmatization, POS tagging, and stopword removal
* Vectorization using `TfidfVectorizer` on key metadata fields
* Weighted combination of multiple content features
* Dimensionality reduction using `TruncatedSVD`
* Normalized vector space for similarity search
* Option to use `NearestNeighbors` for scalable approximate matching
* Fallback mechanism for cold-start queries
* Clean and interpretable recommendations with similarity scores
* **Streamlit web app** for interactive usage

---

## 📂 Dataset

The dataset used is [`netflix_titles.csv`](https://www.kaggle.com/datasets/shivamb/netflix-shows), which includes metadata on movies and TV shows available on Netflix.

**Key Fields Used**:

* `title`
* `description`
* `genres` (formerly `listed_in`)
* `director`
* `cast`
* `country`
* `rating`
* `release_year`

---

## 🛠️ Installation and Setup

1. Clone the repository or open the Colab notebook.
2. Install the required dependencies:

```bash
pip install pandas numpy scikit-learn nltk scipy streamlit
```

3. Download the NLTK resources (only required once):

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

4. Place the `netflix_titles.csv` file in your working directory.

---

## 🧼 Data Preprocessing

All text-based fields (`description`, `director`, `cast`, `country`, `genres`) are:

* Lowercased
* Stripped of non-alphanumeric characters
* Tokenized and POS tagged
* Lemmatized using `WordNetLemmatizer`
* Cleaned of stopwords and short tokens

---

## 🔍 Feature Engineering

Each field is vectorized separately using **TF-IDF**:

| Feature     | Max Features | Weight |
| ----------- | ------------ | ------ |
| Description | 3000         | 1.0    |
| Genres      | 500          | 0.8    |
| Director    | 300          | 0.5    |
| Cast        | 500          | 0.4    |
| Country     | 200          | 0.2    |

All vectors are horizontally stacked using `scipy.sparse.hstack()` and reduced using **SVD (n=200)** for efficiency.

---

## 🤖 Recommendation Logic

### ✅ If the title exists:

* Find its nearest neighbors in feature space using:

  * `NearestNeighbors` with cosine distance (preferred)
  * Or `cosine_similarity` if ANN is unavailable
* Return the top `n` most similar titles excluding the input

### ❌ If the title does **not** exist (cold-start):

* Fallback to top 100 most recent titles
* Use TF-IDF on genres to find semantically similar popular items
* Return top `n` similar titles based on genre similarity

---

## 🧪 Example Usage

```python
recommend("Stranger Things")
recommend("Non-Existent Movie")
```

---

## 🌐 Streamlit Web Application

This project includes a user-friendly **Streamlit** app that allows users to input a movie title and receive instant recommendations via a web interface.

### 🎮 How to Run the App Locally

```bash
streamlit run app.py
```

The application supports:

* Input box for movie title
* Interactive results with similarity scores
* Handles cold-start gracefully if the title does not exist in the dataset

### 🛰️ Deployment

We deploy the app publicly using: **Streamlit Community Cloud**. The app is accessible via:

```
[Visit our app!](https://netflix-hybrid-recommender.streamlit.app/)
```

---

## 📈 Output Format

Returns a `pandas.DataFrame` with the following fields:

* `title`
* `director`
* `cast`
* `genres`
* `country`
* `release_year`
* `rating`
* `description`
* `similarity` (score between 0 and 1)

---

## 📄 Function Overview

| Function                             | Description                    |
| ------------------------------------ | ------------------------------ |
| `clean_text(text)`                   | Normalizes and lemmatizes text |
| `load_and_preprocess_data(filepath)` | Reads CSV and processes fields |
| `vectorize_text(series)`             | Generates TF-IDF vectors       |
| `recommend(title, top_n)`            | Main recommendation engine     |
