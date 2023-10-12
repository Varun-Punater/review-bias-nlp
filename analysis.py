from transformers import pipeline
import json

pipe_yelp = pipeline("text-classification", model="mrcaelumn/yelp_restaurant_review_sentiment_analysis")

pipe_roberta = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

# we should analyze the first 20 reviews from out.json
f = open("out.json", "r")
data = json.load(f)

# now go through the first 20 reviews and print the results
limit = 10
samples = 0
for i in data:

    review = data[i]["text"]

    # now use both pipelines to analyze the sentiment and print the results
    print(f"Review: {review[:50]}...")
    print("Yelp pipeline:")
    print(pipe_yelp(review))
    print("Roberta pipeline:")
    print(pipe_roberta(review))
    print()

    if samples == limit:
        break
    samples += 1

