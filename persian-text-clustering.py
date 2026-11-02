import pandas as pd
from hazm import Normalizer, Lemmatizer, word_tokenize, stopwords_list
import re
import string
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


pd.set_option('display.max_colwidth', 150) 

# Load the CSV file
df = pd.read_csv("golestan.csv")
#rename hakyt column
df = df.rename(columns={'hekayt': 'hekayat'})

df['hekayat'] = df['hekayat'].str.strip('"\r\n')

# Initialize Hazm tools
normalizer = Normalizer()
lemmatizer = Lemmatizer()
stop_words = stopwords_list()

# Define the preprocessing function
def preprocess_text(text):
    normalized_text = normalizer.normalize(text)
    
    tokens = word_tokenize(normalized_text)
    
    processed_tokens = []
    
    for token in tokens:
        token_cleaned = token.strip(string.punctuation + '«»؟!:؛ـ،.“”()[]{}…')
        

        if token_cleaned and token_cleaned not in stop_words:
            lemma = lemmatizer.lemmatize(token_cleaned)
            final_word = lemma.split('#')[0]
            
            if len(final_word) > 1 and not re.match(r'^[a-zA-Z]+$', final_word): 
                processed_tokens.append(final_word)

    return ' '.join(processed_tokens)

df['hekayat_processed'] = df['hekayat'].apply(preprocess_text)

print("Displaying the first 5 rows with original and processed text:")
print(df[['hekayat', 'hekayat_processed']].head(5))

print(f"\nTotal number of processed stories: {len(df)}")


tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['hekayat_processed'])
feature_names = tfidf_vectorizer.get_feature_names_out()


doc_index = 0

feature_vector = tfidf_matrix[doc_index, :]

coo_rep = coo_matrix(feature_vector)
present_indices = coo_rep.col
tfidf_scores = coo_rep.data

words_present = [feature_names[i] for i in present_indices]

results = pd.DataFrame({
    'Word': words_present,
    'TF-IDF Score': tfidf_scores
})

results = results.sort_values(by='TF-IDF Score', ascending=False)

print("\nTop 10 Most Important Words in first Hekayat:")
# Displaying the top 20 words that are present in this specific Hekayat
print(results.head(10).to_markdown(index=False))



# --- Apply K-Means Clustering ---
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

kmeans.fit(tfidf_matrix)

df['cluster'] = kmeans.labels_

print("\nCluster assignment for first 10 hekayat:")
print(df[['hekayat_processed', 'cluster']].head(10))

print("\nNumber of hekayat in each cluster:")
print(df['cluster'].value_counts())



def get_top_terms_from_centers(kmeans_model, feature_names, top_n=5):
    centers = kmeans_model.cluster_centers_   
    top_terms = {}

    for cluster_id in range(kmeans_model.n_clusters):
        center_vector = centers[cluster_id]   

        top_indices = center_vector.argsort()[::-1][:top_n]

        words_scores = [(feature_names[i], center_vector[i]) for i in top_indices]
        top_terms[cluster_id] = words_scores

    return top_terms


top5_from_centers = get_top_terms_from_centers(kmeans, feature_names, top_n=5)

print("\n--- TOP 5 WORDS PER CLUSTER (FROM CLUSTER CENTERS) ---\n")
for cid, items in top5_from_centers.items():
    print(f"Cluster {cid}:")
    for word, score in items:
        print(f"   {word:20s} -> {score:.4f}")
    print("-" * 40)



inertia = []
k_range = range(2, 16)  

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(tfidf_matrix)
    inertia.append(km.inertia_)


plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, 'bo-', markersize=8)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (SSE)')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()


selected_k = 8

kmeans = KMeans(n_clusters=selected_k, random_state=42, n_init=10)

df['cluster'] = kmeans.fit_predict(tfidf_matrix)

print(f"\n--- Cluster Sizes (k={selected_k}) ---")
cluster_counts = df['cluster'].value_counts().sort_index()
print(cluster_counts)

print(f"\n--- Cluster vs Bab Distribution (k={selected_k}) ---")
crosstab_res = pd.crosstab(df['cluster'], df['bab'])
print(crosstab_res)
print("-" * 40)

dbscan = DBSCAN(eps=0.87, min_samples=3, metric='cosine')
labels = dbscan.fit_predict(tfidf_matrix)

df['dbscan_cluster'] = labels

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("-" * 30)
print(f"number of cluster: {n_clusters_}")
print(f"number of noises: {n_noise_}")
print("-" * 30)

print("Cluster Member Counts::")
print(df['dbscan_cluster'].value_counts().sort_index())
print("-" * 30)

print("\n Cluster vs Bab Distribution:")
print(pd.crosstab(df['dbscan_cluster'], df['bab']))



def get_top_keywords(data, clusters, vectorizer, n_terms=5):
    df_temp = pd.DataFrame(data.toarray(), columns=vectorizer.get_feature_names_out())
    df_temp['cluster'] = clusters
    
    cluster_means = df_temp.groupby('cluster').mean()
    
    print("\n--- Top 5 Words per Cluster ---")
    for cluster_id in sorted(cluster_means.index):
        if cluster_id == -1: continue 
        
        print(f"\nCluster {cluster_id}:")
        top_words = cluster_means.loc[cluster_id].sort_values(ascending=False).head(n_terms)
        
        for word, score in top_words.items():
            print(f"   - {word}: {score:.4f}")

get_top_keywords(tfidf_matrix, df['dbscan_cluster'], tfidf_vectorizer)
