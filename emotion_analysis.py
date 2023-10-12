# #https://huggingface.co/SamLowe/roberta-base-go_emotions

#'0': admiration
# '1': amusement
# '2': anger
# '3': annoyance
# '4': approval
# '5': caring
# '6': confusion
# '7': curiosity
# '8': desire
# '9': disappointment
# '10': disapproval
# '11': disgust
# '12': embarrassment
# '13': excitement
# '14': fear
# '15': gratitude
# '16': grief
# '17': joy
# '18': love
# '19': nervousness
# '20': optimism
# '21': pride
# '22': realization
# '23': relief
# '24': remorse
# '25': sadness
# '26': surprise
# '27': neutral

from transformers import pipeline
import json
from pathlib import Path
import pandas as pd

print("Start")

pipe = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions")

folder_path = Path("cuisineData")

print("Finish huggingface import")

data_list = []

# Iterate through the files
for file in folder_path.iterdir():
    if file.is_file():

        print(file)

        # Initialize a dictionary to store emotion counts
        emotions = {}

        # Open and load the JSON data
        with open(file, "r") as f:
            data = json.load(f)

        count = 0
        for i in data:

            count += 1
            if (count%10 == 0):
                print("+")
            if (count == 100):
                break
            #print("----------")

            review = data[i]["text"]
            if len(review) > 511:
                review = review[:511]
            #print(review + "\n")
            result = pipe("Review = " + review)
            emotion = result[0]["label"]
            #print("Emotion: " + emotion)

            if emotion in emotions:
                emotions[emotion] += 1
            else:
                emotions[emotion] = 1

        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        
        # Print the emotion counts
        for emotion, count in sorted_emotions:
            print(f"{emotion}: {count}")

        # Append the emotion counts for this file to the list
        data_list.append(emotions)

# Create a DataFrame from the list of dictionaries
cuisinetoemotion = pd.DataFrame(data_list)

# If you want to add the file names as a column in the DataFrame:
cuisinetoemotion["file_name"] = [file.name for file in folder_path.iterdir() if file.is_file()]

# Print the DataFrame
print(cuisinetoemotion)


#cuisinetoemotion = pd.DataFrame(data, columns=['amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'])