# Persian Text Clustering — Golestan Stories

Unsupervised clustering of Persian prose from **Saadi's Golestan**, using TF-IDF features and both K-Means and DBSCAN.

## Pipeline

### 1. Loading
- Reads `golestan.csv`, renames the `hekayt` column to `hekayat`, and strips stray quotes/newlines.

### 2. Preprocessing (Hazm)
- **Normalization** — `Normalizer`
- **Tokenization** — `word_tokenize`
- **Stopword removal** — `stopwords_list`
- **Lemmatization** — `Lemmatizer` (token kept before the `#` delimiter)
- Strips Persian punctuation and drops tokens that are single characters or pure Latin.

### 3. TF-IDF
- `TfidfVectorizer` over the processed text.
- Prints the top 10 highest-scoring words for the first story.

### 4. K-Means
- Initial run with `k = 5`, then the **elbow method** over `k = 2…15` plotted against inertia.
- Final run with the chosen `k = 8`.
- Reports cluster sizes and a `cluster × bab` (chapter) crosstab.
- Prints the top 5 words per cluster from the cluster centers.

### 5. DBSCAN
- Cosine metric with `eps = 0.87`, `min_samples = 3`.
- Reports number of clusters, number of noise points, member counts, and a `cluster × bab` crosstab.
- Prints top keywords per (non-noise) cluster.

## Requirements

```
pandas numpy hazm nltk scikit-learn scipy matplotlib tabulate
```

The dataset (`golestan.csv`) is expected in the working directory.

## Run

```bash
python persian-text-clustering.py
```
