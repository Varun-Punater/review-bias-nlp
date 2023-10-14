import numpy as np
import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import os
import json
from pandas import DataFrame

# pipe_yelp = pipeline("text-classification", model="mrcaelumn/yelp_restaurant_review_sentiment_analysis")
# pipe_roberta = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

yelp_tknzr = AutoTokenizer.from_pretrained("mrcaelumn/yelp_restaurant_review_sentiment_analysis")
yelp_mdl = AutoModelForSequenceClassification.from_pretrained("mrcaelumn/yelp_restaurant_review_sentiment_analysis")

rob_tknzr = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
rob_mdl = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

# we want to save the results of the sentiment analysis to a dataframe
sentiment_dataframe = DataFrame(columns=["cuisine", "yelp_avg_neg", "yelp_avg_nue", "yelp_avg_pos", "rob_avg_neg", "rob_avg_nue", "rob_avg_pos"])


# we now have a folder called cuisineData, which has all the reviews for each cuisine
# we now want to analyze 20 reviews from each cuisine json file

CUISINE_DIR = "cuisineData"

SAMPLES_PER_CUISINE = 1000

for filename in os.listdir(CUISINE_DIR):
    if filename.endswith('.json'):
        with open(os.path.join(CUISINE_DIR, filename)) as f:
            # print the name of the file
            print(f"File: {filename}")
            data = json.load(f)

            # we need to keep track of the sum of the probabilities from the tensors
            yelp_sum = np.zeros(3)
            rob_sum = np.zeros(3)
            
            # print the first 3 reviews
            for i in range(SAMPLES_PER_CUISINE):

                index = np.random.randint(0, len(data))
                review = data[index]
                
                # yelp
                yelp_inputs = yelp_tknzr(review, return_tensors="pt", add_special_tokens=True, padding=True, truncation=True, max_length=512)
                yelp_outputs = yelp_mdl(**yelp_inputs)

                yelp_probs = F.softmax(yelp_outputs.logits, dim=-1)
                yelp_sum += yelp_probs.detach().cpu().numpy().flatten()
                

                # roberta
                rob_inputs = rob_tknzr(review, return_tensors="pt", add_special_tokens=True, padding=True, truncation=True, max_length=512)
                rob_outputs = rob_mdl(**rob_inputs)

                rob_probs = F.softmax(rob_outputs.logits, dim=-1)
                rob_sum += rob_probs.detach().cpu().numpy().flatten()

            # now we need to average the results
            yelp_avg = yelp_sum / SAMPLES_PER_CUISINE
            roberta_avg = rob_sum / SAMPLES_PER_CUISINE

            cuisine_name = filename[:-5]

            # final list is the cuisine name and the averages
            final_list = [cuisine_name] + yelp_avg.tolist() + roberta_avg.tolist()

            # now we need to add the list to the dataframe as a row
            sentiment_dataframe.loc[len(sentiment_dataframe)] = final_list

print()
print("Sentiment dataframe:")
print(sentiment_dataframe)

# save the dataframe to sentiment.csv
sentiment_dataframe.to_csv("sentiment_table_" + str(SAMPLES_PER_CUISINE) + "_samples.csv", index=False)