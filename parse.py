import json

# rf = open("archive/yelp_academic_dataset_business.json", "r")
# restaurantData = json.load(rf)
# for i in restaurantData["business_id"]:
#     print(i)


data = dict()
dataCount = dict()
tags = []
nullCount = 0
print("Loading restaurant data")
with open("yelp_dataset/yelp_academic_dataset_business.json", "r") as file:
    for line in file:
        entry = json.loads(line)

        if entry["categories"] != None and "Restaurants" in entry["categories"]:
            print(entry["categories"])
            data[entry["business_id"]] = entry
            subTag = entry["categories"].split(",")
            subTag = [i.strip() for i in subTag]
            tags += subTag
            entry["categories"] = tags


            if entry["name"] not in dataCount:
                dataCount[entry["name"]] = 1
            else:
                dataCount[entry["name"]] += 1
        if entry["categories"] == None:
            nullCount += 1
        tags = []
print("Loading restaurant successful...", len(data), "entries loaded")
print(nullCount, "stores without categories detected")
file.close()
tags = set(tags)


# sol = 0
# solC = 0
# for i in dataCount:
#     if dataCount[i] <= 4:
#         print(i)
#         sol += dataCount[i]
#         solC += 1

# print("Total dup", sol)
# print("Dup names", solC)


# for i in tags:
#     print(i)
# print(len(tags), "unique tags")

cuisineTypes = ["American (Traditional)", "American (New)", "Cajun/Creole", "Southern", "Soul Food", "Mexican", "Latin American", "Cuban", "Italian", "Mediterranean", "Greek", "French", "Irish", "Spanish", "Chinese", "Japanese", "Thai", "Vietnamese", "Indian", "Korean" ]
# cuisineTypes = ["Chinese" , "French", "Mexican", "American", "Japanese", "Italian", "Greek", "Spanish", "Lebanese", "Thai", "Indian", "Korean", "Asian Fusion", "Cajun/Creole", "Southern", "Soul Food", "Latin American"]
cuisineCount = [0] * len(cuisineTypes)

dupCount = 0
dupFlag = 0
validCount = 0
restaurantToCuisine = dict()
for i in data:
    uniqueCuisine = []
    for j in range(len(cuisineTypes)):
        if cuisineTypes[j] in data[i]["categories"]:
            dupFlag += 1
            cuisineCount[j] += 1
            uniqueCuisine.append(cuisineTypes[j])
    if dupFlag == 1:
        validCount += 1
        restaurantToCuisine[data[i]["business_id"]] = uniqueCuisine
    elif dupFlag > 1:
        dupCount+=1
        restaurantToCuisine[data[i]["business_id"]] = uniqueCuisine

    dupFlag = 0
    uniqueCuisine = ""
print("Stats")

for i in range(len(cuisineCount)):
    print(cuisineTypes[i], ":", cuisineCount[i])
print(dupCount, "doubled counted")
print(sum(cuisineCount), "total counts")
print(len(restaurantToCuisine), " unique restaurants")

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
