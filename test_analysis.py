from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import json


tokenizer = AutoTokenizer.from_pretrained("mrcaelumn/yelp_restaurant_review_sentiment_analysis")
model = AutoModelForSequenceClassification.from_pretrained("mrcaelumn/yelp_restaurant_review_sentiment_analysis")

# we should analyze the first 20 reviews from out.json
f = open("out.json", "r")
data = json.load(f)

# now go through the first 20 reviews and print the results
limit = 10
samples = 0
for i in data:

    review = data[i]["text"]
    inputs = tokenizer(review, return_tensors="pt")
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1)

    # we want to print the first 50 characters of the review
    print(f"Review: {review[:50]}...")
    print(f"Negative: {probs[0, 0]}")
    print(f"Neutral: {probs[0, 1]}")
    print(f"Positive: {probs[0, 2]}")
    
    if samples == limit:
        break
    samples += 1

