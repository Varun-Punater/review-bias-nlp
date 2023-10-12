import json


#cuisineTypes = ["Chinese" , "French", "Mexican", "American", "Japanese", "Italian", "Greek", "Spanish", "Lebanese", "Thai", "Indian", "Korean", "Asian Fusion"]
cuisineTypes = ["American (Traditional)", "American (New)", "Cajun/Creole", "Southern", "Soul Food", "Mexican", "Latin American", "Cuban", "Italian", "Mediterranean", "Greek", "French", "Irish", "Spanish", "Chinese", "Japanese", "Thai", "Vietnamese", "Indian", "Korean" ]
cuisineCount = [0] * len(cuisineTypes)

f = open("yelpData.json", "r")
data = json.load(f)

for i in data:
    # print(i)
    # for j in data[i]["cuisine"]:
    #     print(j)
    if data[i]["cuisine"] in cuisineTypes:
        cuisineCount[cuisineTypes.index(data[i]["cuisine"])]+=1
print(sum(cuisineCount), "total reviews")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
for i in range(len(cuisineCount)):
    print(cuisineTypes[i], ":", cuisineCount[i])
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")