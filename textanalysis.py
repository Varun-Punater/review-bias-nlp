import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# Load the dataset
file_path = 'textListOut.json'

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

# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Define a function for text preprocessing
def preprocess_text(text):
    # Parse out non-english characters
    text = text.encode('ascii', 'ignore').decode('ascii')

    # Tokenize, remove stop words, and lemmatize
    tokens = nltk.word_tokenize(text)
    processed_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalpha() and token.lower() not in stop_words]
    return ' '.join(processed_tokens)

# Apply preprocessing to the review texts
reviews_df['processed_text'] = reviews_df['text'].apply(preprocess_text)

# Display the first few rows of the processed DataFrame
# print(reviews_df.head())

# Vectorizing the text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # Limiting to 5000 features for simplicity
tfidf_matrix = vectorizer.fit_transform(reviews_df['processed_text'])

# Save the vectorized text to a new DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out(), index=reviews_df.index)

# Display the TF-IDF DataFrame
# print(tfidf_df.head())


# 1. Combine Text Features with Ratings
reviews_df['rating'] = pd.to_numeric(reviews_df['rating'], errors='coerce')
combined_df = pd.concat([tfidf_df, reviews_df[['rating']]], axis=1)

# 2. Normalize the Data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(combined_df.fillna(0))

# Assuming 'normalized_data' is the scaled TF-IDF data from the previous steps

k_range = range(1,21)

# 1. Run K-means for K from 1 to 20
inertia_values = []
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
    kmeans.fit(normalized_data)
    inertia_values.append(kmeans.inertia_)

# 2. Graph the Loss Values
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia_values, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.xticks(k_range)
plt.show()
