import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD


# Load the dataset
file_path = 'hundredPerListOut.json'

# Re-loading the updated dataset to analyze its structure again
try:
    with open(file_path, 'r') as file:
        reviews_df = pd.read_json(file_path)
        reviews_df.set_index(['user_id', 'business_id'], inplace=True)
except Exception as e:
    print(f"Error loading JSON: {e}")

# Display the first few rows of the updated dataset to understand its structure
# reviews_df.head()

# Filter the dataset for the specified cuisines
CUISINE_TYPES = ["American (Traditional)", "American (New)", "Cajun/Creole", "Southern", 
                 "Soul Food", "Mexican", "Latin American", "Cuban", "Italian", "Mediterranean", 
                 "Greek", "French", "Irish", "Spanish", "Chinese", "Japanese", "Thai", 
                 "Vietnamese", "Indian", "Korean"]
reviews_df = reviews_df[reviews_df['cuisine'].isin(CUISINE_TYPES)]

# Preprocessing the text

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def aggregate_synonyms(word):
    # Find synsets for the word
    synsets = wordnet.synsets(word)
    if not synsets:
        return word

    # Example strategy: return the name of the first lemma of the first synset
    # You can modify this strategy as needed
    return synsets[0].lemmas()[0].name()

# Defining allowed pos tags as defined by the nltk library
# we want all nouns, verbs, adjectives, and adverbs
allowed_pos_tags = ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 
                    'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']

# Define a function for text preprocessing
def preprocess_text(text):
    # Parse out non-english characters
    text = text.encode('ascii', 'ignore').decode('ascii')

    # Tokenize, remove stop words, and lemmatize
    tokens = nltk.word_tokenize(text)
    processed_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalpha() and token.lower() not in stop_words]

    # Aggregate synonyms  
    processed_tokens = [aggregate_synonyms(token) for token in processed_tokens]

    # allow only the allowed pos tags
    processed_tokens = [token for token in processed_tokens if nltk.pos_tag([token])[0][1] in allowed_pos_tags]

    return ' '.join(processed_tokens)

# Apply preprocessing to the review texts
reviews_df['processed_text'] = reviews_df['text'].apply(preprocess_text)

for cuisine in CUISINE_TYPES:
    cuisine_df = reviews_df[reviews_df['cuisine'] == cuisine]
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=5000, min_df=0.01)
    tfidf_matrix = vectorizer.fit_transform(cuisine_df['processed_text'])

    # Applying Truncated SVD for dimensionality reduction
    svd = TruncatedSVD(n_components=25)  # Using 25 components
    tfidf_reduced = svd.fit_transform(tfidf_matrix)

    # Extracting feature names and their weights for the top 25 dimensions
    feature_names = vectorizer.get_feature_names_out() 
    top_features = {feature_names[i]: svd.components_[0][i] for i in range(25)}

    print(f"Cuisine Type: {cuisine}")
    print(f"Top 25 Features: {top_features}\n")


# # TF-IDF Vectorization with dimensionality reduction
# vectorizer = TfidfVectorizer(max_features=5000, min_df=0.01)
# tfidf_matrix = vectorizer.fit_transform(reviews_df['processed_text'])

# # Applying Truncated SVD for dimensionality reduction
# svd = TruncatedSVD(n_components=100)  # Reducing to 100 components
# tfidf_reduced = svd.fit_transform(tfidf_matrix)

# # Compute cumulative explained variance ratio
# cumulative_variance = np.cumsum(svd.explained_variance_ratio_)

# # Plotting the cumulative explained variance
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, 101), cumulative_variance, marker='o')
# plt.title('Cumulative Explained Variance by SVD Components')
# plt.xlabel('Number of Components')
# plt.ylabel('Cumulative Explained Variance')
# plt.show()

# # Convert the reduced matrix back to a DataFrame
# tfidf_reduced_df = pd.DataFrame(tfidf_reduced, index=reviews_df.index)

# # Display the reduced TF-IDF DataFrame
# print(tfidf_reduced_df.head())

# # 1. Combine Text Features with Ratings
# reviews_df['rating'] = pd.to_numeric(reviews_df['rating'], errors='coerce')
# # combined_df = pd.concat([tfidf_df, reviews_df[['rating']]], axis=1)
# combined_df = pd.concat([tfidf_reduced_df, reviews_df[['rating']]], axis=1)
# combined_df.columns = combined_df.columns.astype(str)

# # 2. Normalize the Data
# scaler = StandardScaler()
# normalized_data = scaler.fit_transform(combined_df)

# # Assuming 'normalized_data' is the scaled TF-IDF data from the previous steps

# k_range = range(1,51)

# # 1. Run K-means for K from 1 to 20
# inertia_values = []
# for k in k_range:
#     kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
#     kmeans.fit(normalized_data)
#     inertia_values.append(kmeans.inertia_)

# # 2. Graph the Loss Values
# plt.figure(figsize=(10, 6))
# plt.plot(k_range, inertia_values, marker='o')
# plt.title('Elbow Method for Optimal K')
# plt.xlabel('Number of Clusters')
# plt.ylabel('Inertia')
# plt.xticks(k_range)
# plt.show()
