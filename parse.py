import json

data = dict()
nameCount = dict()
tags = []
nullCount = 0
totCount = 0
print("Loading restaurant data")
with open("yelp_dataset/yelp_academic_dataset_business.json", "r") as file:
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

cuisineTypes = ["American (Traditional)", "American (New)", "Cajun/Creole", "Southern", "Soul Food", "Mexican", "Latin American", "Cuban", "Italian", "Mediterranean", "Greek", "French", "Irish", "Spanish", "Chinese", "Japanese", "Thai", "Vietnamese", "Indian", "Korean"]
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
# reviews = []
cuisineReviews = dict()
for i in cuisineTypes:
    cuisineReviews[i] = []
reviewTot = 0
print("Loading all reviews")
with open("yelp_dataset/yelp_academic_dataset_review.json", "r") as file:
    for line in file:
        entry = json.loads(line)
        if entry["business_id"] in restaurantToCuisine:
            entryDict = dict()
            entryDict["review_id"] = entry["review_id"]
            entryDict["cuisine"] = restaurantToCuisine[entry["business_id"]]
            entryDict["business_id"] = entry["business_id"]
            entryDict["text"] = entry["text"]
            entryDict["date"] = entry["date"]
            entryDict["name"] = data[entry["business_id"]]["name"]
            output[entry["user_id"]] = entryDict
            # reviews.append(entry)
        reviewTot+=1
print(len(output), "reviews related to restaurants obtained")
print("Total reviews:", reviewTot)
file.close()

json_object = json.dumps(output, indent=4)
 
# Writing to sample.json
with open("yelpData.json", "w") as outfile:
    outfile.write(json_object)
