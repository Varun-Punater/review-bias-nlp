import json
import os

print("Reading data")
f = open("out.json", "r")
data = json.load(f)
f.close()
if not os.path.exists("cuisineData"):
    os.makedirs("cuisineData")

print("Data successfully read")
cuisineSort = dict()
for i in data:
    if data[i]["cuisine"] not in cuisineSort:
        cuisineSort[data[i]["cuisine"]] = dict()
    cuisineSort[data[i]["cuisine"]][i] = data[i]
print(len(cuisineSort), "types of cuisines identified")
for i in cuisineSort:
    json_object = json.dumps(cuisineSort[i], indent= 4)
    outName = i.replace(" ", "")
    outName = outName.replace("/", "_")
    outName = "cuisineData/" + outName + ".json"
    with open(outName, 'w') as outfile:
        outfile.write(json_object)
print("Job done")

