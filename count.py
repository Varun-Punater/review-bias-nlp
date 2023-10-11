import json


#cuisineTypes = ["Chinese" , "French", "Mexican", "American", "Japanese", "Italian", "Greek", "Spanish", "Lebanese", "Thai", "Indian", "Korean", "Asian Fusion"]
cuisineTypes = ["American (Traditional)", "American (New)", "Cajun/Creole", "Southern", "Soul Food", "Mexican", "Latin American", "Cuban", "Italian", "Mediterranean", "Greek", "French", "Irish", "Spanish", "Chinese", "Japanese", "Thai", "Vietnamese", "Indian", "Korean" ]
cuisineCount = [0] * len(cuisineTypes)

f = open("yelpData.json", "r")
data = json.load(f)
# with open("test1.json", "r") as file:
#     for line in file:
#         entry = json.loads(line)
#         if entry["cuisine"] in cuisineTypes:
#             cuisineCount[cuisineTypes.index(entry["cuisine"])]+=1
for i in data:
    # print(i)
    for j in data[i]["cuisine"]:
        if j in cuisineTypes:
            cuisineCount[cuisineTypes.index(j)]+=1
print(sum(cuisineCount), "total reviews")
for i in range(len(cuisineCount)):
    print(cuisineTypes[i], ":", cuisineCount[i])