import numpy as np
from sklearn.cluster import KMeans      #kmeans
import pandas as pd                     #dataframe
from sklearn.decomposition import PCA   #pca
import json                             #json

def kmeans_cluster(data, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=1)
    labels = kmeans.fit_predict(data)
    centers = kmeans.cluster_centers_ 
    inertia = kmeans.inertia_
    return labels, centers, inertia

def read_parsed_json(filename):
    with open(filename, 'r') as json_file:
        data = json.load(json_file)
    concatenated_data = []
    cuisine_type =[]
    i = 0
    for entry in data:
        i += 1
        cuisine_vector = entry['cuisine']
        rating = np.atleast_1d(entry['rating'])
        sentiment = entry['sentiment']
        emotion = entry['emotion']
        # print("Cuisine Vector:", cuisine_vector)
        # print("Rating:", rating)
        # print("Sentiment:", sentiment)
        # print("Emotion:", emotion)
        # print("\n")
        entry_data = np.concatenate([rating, sentiment, emotion])
        # print(entry_data)
        concatenated_data.append(entry_data)
        cuisine_index = np.argmax(cuisine_vector)
        cuisine_type.append(cuisine_index)
    print(str(i) + " reviews parsed.")
    return np.array(concatenated_data), np.array(cuisine_type)


#this is just for testing
# np.random.seed(1)
# n = 100000
# dimensionality = 32
# data = np.random.rand(n, dimensionality)
#data should contain output from fileread


#K-means
data, cuisine_type = read_parsed_json("parsed_kmeans_data.json")

#use inertia to determine best number of clusters
lowest_inertia = 0
lowesti_cluster = 1
for i in range(1, 20):
    t1, t2, inertia = kmeans_cluster(data, int(i))
    print("\nCluster " + str(i) + ", inertia " + str(inertia))
    if i == 1:
        lowest_inertia = inertia
    else:
        if inertia < lowest_inertia:
            lowest_inertia = inertia
            lowesti_cluster = i
print("\nLowest inertia is cluster size " + str(lowesti_cluster) + " with inertia " + str(lowest_inertia))

num_clusters = lowesti_cluster
cluster_labels, cluster_centers, inertia = kmeans_cluster(data, num_clusters)

#print out results
cluster_labels_with_cuisine = np.column_stack((cuisine_type, cluster_labels))
cluster_labels_df = pd.DataFrame(cluster_labels_with_cuisine, columns=['CuisineType', 'Cluster'])
print("\nCluster Labels:\n", cluster_labels_df)
column_labels = ['Rating:', 'Positive:', 'Negative:', 'Neutral:', 'Admiration', 'Amusement:', 'Anger:', 'Annoyance:', 'Approval:', 'Caring:', 'Confusion:', 'Curiosity:', 'Desire:', 'Disappointment:', 'Disapproval:', 'Disgust:', 'Embarrassment:', 'Excitement:', 'Fear:', 'Gratitude:', 'Grief:', 'Joy:', 'Love:', 'Nervousness:', 'Optimism:', 'Pride:', 'Realization:', 'Relief:', 'Remorse:', 'Sadness:', 'Surprise:', 'Neutral:']
cluster_centers_df = pd.DataFrame(cluster_centers, columns=column_labels)
print("\nCluster Centers:\n", cluster_centers_df) 

cluster_labels_df.to_csv(("cluster_labels_df.csv"), index=False)
cluster_centers_df.to_csv(("cluster_centers_df.csv"), index=False)

#PCA
# num_components = 3
# pca = PCA(n_components=num_components)
# cluster_centers_pca = pca.fit_transform(cluster_centers)
# cluster_centers_pca_df = pd.DataFrame(cluster_centers_pca)
# print("\nCluster Centers PCA:\n", cluster_centers_pca_df)