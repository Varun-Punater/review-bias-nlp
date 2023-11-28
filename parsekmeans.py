import numpy as np
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import os
import json
from pandas import DataFrame
from tqdm import tqdm

# file
DATASET_DIR = "/Users/varunpunater/Downloads/yelp_dataset"

CUISINE_TYPES = ["American (Traditional)", "American (New)", "Cajun/Creole", "Southern", "Soul Food", "Mexican", "Latin American", "Cuban", "Italian", "Mediterranean", "Greek", "French", "Irish", "Spanish", "Chinese", "Japanese", "Thai", "Vietnamese", "Indian", "Korean"]

def createjson():
    data = dict()
    nameCount = dict()
    tags = []
    nullCount = 0
    totCount = 0
    print("Loading restaurant data")
    with open(DATASET_DIR + "/yelp_academic_dataset_business.json", "r") as file:
        for line in file:
            entry = json.loads(line)
            if entry["categories"] != None: #converts categories string into a list
                categories = entry["categories"].split(",")
                categories = [i.strip() for i in categories]

                if "Restaurants" in categories and "Fast Food" not in categories and "Cafes" not in categories:#category checker
                    entry["categories"] = categories
                    data[entry["business_id"]] = entry
            else:
                nullCount+=1
            if entry["name"] not in nameCount:
                nameCount[entry["name"]] = 1
            else:
                nameCount[entry["name"]] += 1
            totCount += 1

    print(totCount, "restaurants in json file")
    print(nullCount, "stores without categories detected")

    print(len(data), "restaurants with the specified tags loaded")

    file.close()

    chainNames = set()

    for i in nameCount:
        if nameCount[i] >= 4:
            chainNames.add(i)
    print(len(chainNames), "chains detected")
    chainSum = 0
    validRestaurants = dict()

    for i in data:
        if data[i]["name"] not in chainNames:
            validRestaurants[data[i]["business_id"]] = data[i]
        else:
            chainSum += 1
    data = validRestaurants
    print(chainSum, "chain restaurants removed")

    cuisineTypes = CUISINE_TYPES
    # cuisineTypes = ["Chinese" , "French", "Mexican", "American", "Japanese", "Italian", "Greek", "Spanish", "Lebanese", "Thai", "Indian", "Korean", "Asian Fusion", "Cajun/Creole", "Southern", "Soul Food", "Latin American"]
    cuisineCount = [0] * len(cuisineTypes)

    dupCount = 0
    dupFlag = 0
    validCount = 0
    zeroCount = 0
    restaurantToCuisine = dict()
    for i in data:
        for j in range(len(cuisineTypes)):
            if cuisineTypes[j] in data[i]["categories"]:
                dupFlag += 1
                uniqueCuisine = cuisineTypes[j]
        if dupFlag == 0:
            zeroCount += 1
        if dupFlag == 1:
            validCount += 1
            cuisineCount[cuisineTypes.index(uniqueCuisine)] += 1
            restaurantToCuisine[data[i]["business_id"]] = uniqueCuisine
        elif dupFlag > 1:
            dupCount+=1
        dupFlag = 0
    print("Stats")
    print("Cuisine Type Stats")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    for i in range(len(cuisineCount)):
        print(cuisineTypes[i], ":", cuisineCount[i])
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    print(len(data), "total restaurants considered")
    print(zeroCount, "restaurants with no cuisine tags")
    print(dupCount, "doubled counted")
    print(len(restaurantToCuisine), "unique restaurants")

    output = dict()
    listOut = []
    # reviews = []
    cuisineReviews = dict()
    for i in cuisineTypes:
        cuisineReviews[i] = []
    reviewTot = 0
    print("Loading all reviews")
    with open(DATASET_DIR + "/yelp_academic_dataset_review.json", "r") as file:
        for line in tqdm(file):
            entry = json.loads(line)
            if entry["business_id"] in restaurantToCuisine:
                entryDict = dict()
                entryDict["cuisine"] = restaurantToCuisine[entry["business_id"]]
                entryDict["text"] = entry["text"]
                entryDict["stars"] = entry["stars"]
                listOut.append(entryDict)
                # output[entry["review_id"]] = entryDict
                # reviews.append(entry)
            reviewTot+=1
    # print(len(output), "reviews related to restaurants obtained")
    print(len(listOut), "reviews related to restaurants obtained")
    print("Total reviews:", reviewTot)
    file.close()

    # json_object = json.dumps(output, indent=4)
    json_object = json.dumps(listOut, indent=4)
    # Writing to sample.json
    with open("kmeansout.json", "w") as outfile:
        outfile.write(json_object)
    print("Job done")

def createdataarray():

    # robereta model for sentniment analysis
    sent_tknzr = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    sent_mdl = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

    emo_tknzr = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
    emo_mdl = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")

    # we want to create an array with the following infomation stored in the columns:
    # cuisine, review, sentiment, emotion
    # the cuisine value is a one hot encoding of the cuisine type 
    # the sentiment value is the entire sentiment tensor of the review, after it has been softmaxed
    # the emotion value is a the entire emotion tensor of the review, after it has been softmaxed


    # open the json file
    with open("kmeansout.json", "r") as f:
        data = json.load(f)

    # we want to save the results of the sentiment analysis to a dataframe
    result_dataframe = DataFrame(columns=["cuisine", "rating", "sentiment", "emotion"])

    cusine_counts = np.zeros(len(CUISINE_TYPES))

    for i in tqdm(range(len(data))):

        review_index = np.random.randint(0, len(data))

        review = data[review_index]["text"]
        cuisine = data[review_index]["cuisine"]

        # if we've seen over a thousand of this cuisine, we don't want to add it to the dataframe
        if cusine_counts[CUISINE_TYPES.index(cuisine)] >= 1000:
            continue
        else:
            cusine_counts[CUISINE_TYPES.index(cuisine)] += 1

        cusine_array = np.zeros(len(CUISINE_TYPES))
        cusine_array[CUISINE_TYPES.index(cuisine)] = 1

        rating = data[review_index]["stars"]

        # sentiment analysis
        sent_inputs = sent_tknzr(review, return_tensors="pt", add_special_tokens=True, padding=True, truncation=True, max_length=512)
        sent_outputs = sent_mdl(**sent_inputs)

        sent_probs = F.softmax(sent_outputs.logits, dim=-1)

        # emotion analysis
        emo_inputs = emo_tknzr(review, return_tensors="pt", add_special_tokens=True, padding=True, truncation=True, max_length=512)
        emo_outputs = emo_mdl(**emo_inputs)

        emo_probs = F.softmax(emo_outputs.logits, dim=-1)

        # we now need to add this to the dataframe
        result_dataframe.loc[i] = [cusine_array, rating, sent_probs.detach().cpu().numpy().flatten(), emo_probs.detach().cpu().numpy().flatten()]


    # we now need to save the dataframe to a new json file
    result_dataframe.to_json("parsed_kmeans_data.json", orient="records")

    print("Job done")
    

if __name__ == "__main__":
    # createjson()
    createdataarray()