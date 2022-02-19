import argparse
import numpy as np
import pandas as pd
import csv
import os
from tqdm import tqdm
from collections import Counter
import json

parser = argparse.ArgumentParser(description='preprocess ml 100k')
parser.add_argument("--data_path", 
                    type=str,
                    default='data/ml-1m', 
                    help="",
)
parser.add_argument("--item_filter_threshold", 
                    type=int,
                    default=10, 
                    help="item filter threshold",
)
parser.add_argument("--boundary_rating", 
                    type=float,
                    default=3.5, 
                    help="boundary_rating",
)
parser.add_argument("--user_filter_threshold", 
                    type=int,
                    default=32, 
                    help="item filter threshold",
)
parser.add_argument("--out_dir", 
                    type=str,
                    default='data/processed/ml-1m/', 
                    help="",
)
# parser.add_argument("--group_json_file", 
#                     type=str,
#                     default='user_gender_group_map.json', 
#                     help="",
# )
args = parser.parse_args()


# parameters go here
data_path = args.data_path
is_filter = True # filter the item with frequency less than XX
item_filter_threshold = args.item_filter_threshold
user_filter_threshold = args.user_filter_threshold
output_path = args.out_dir

# load rating data
rating_file = os.path.join(data_path, 'ratings.dat')

rating_data = pd.read_csv(rating_file, 
    sep='::',
    names=['userId', 'itemId', 'rating', 'timestamp'],
    dtype={'userId':np.int32,'itemId':np.int32,'rating':np.float64,'timestamp':np.int32},
    engine='python',
    )

print("Rating data: ")
print(rating_data)

# create filter item id list
itemID_rating_cnt =  Counter(rating_data['itemId'].values.tolist())

remove_items = set()
for item_id, cnt in itemID_rating_cnt.items():
    if cnt < item_filter_threshold:
        remove_items.add(item_id)

print("\n# of total item", len(itemID_rating_cnt))
print("# of item to remove", len(remove_items))

# create filter item id list
userID_rating_cnt =  Counter(rating_data['userId'].values.tolist())


remove_users = set()
for user_id, cnt in userID_rating_cnt.items():
    if cnt < user_filter_threshold:
        remove_users.add(user_id)

print("\n# of total users", len(userID_rating_cnt))
print("# of users to remove", len(remove_users))

# create new r_matrix (filter + re-index) and update mapping
# 1 filtering
r_matrix = rating_data[['userId', 'itemId', 'rating']].values
print("\nBefore filtering, r_matrix.shape", r_matrix.shape)

filtered_r_matrix = []
for i in range(r_matrix.shape[0]):
    rating_triplet = r_matrix[i, :]
    user_id, item_id, rating = rating_triplet
    user_id, item_id, rating = int(user_id), int(item_id), rating
    if item_id in remove_items:
        continue
    if user_id in remove_users:
        continue
    else:
        filtered_r_matrix.append([user_id, item_id, rating])

filtered_r_matrix = np.stack(filtered_r_matrix)

print("After filtering, filtered_r_matrix.shape", filtered_r_matrix.shape)

# 2. reindexing
users = []
uMap = dict()
uCounts = {}
items = []
iMap = dict()
iCounts = {}
for i in range(len(filtered_r_matrix)):
    u = int(filtered_r_matrix[i][0])
    v = int(filtered_r_matrix[i][1])

    # For each unique user, find its encoded id and count
    if u not in uMap:
        uMap[u] = len(users)
        users.append(u)
        uCounts[u] = 1
    else:
        uCounts[u] += 1
    # For each unique item, find its encoded id and count
    if v not in iMap:
        iMap[v] = len(items)
        items.append(v)
        iCounts[v] = 1
    else:
        iCounts[v] += 1

# log filtered 
print("After filtering, # of users", len(users))
print("After filtering, # of items", len(items))

assert len(users) == max(uMap.values()) - min(uMap.values()) + 1
assert len(items) == max(iMap.values()) - min(iMap.values()) + 1

output_r_matrix = [] # do not transform into numpy array, otherwise will change dtype of id
for i in range(len(filtered_r_matrix)):
    u = int(filtered_r_matrix[i][0])
    v = int(filtered_r_matrix[i][1])
    rating = filtered_r_matrix[i][2]

    output_r_matrix.append([uMap[u], iMap[v], rating])

# TODO: get most popular item from output_r_matrix
binary_r_matrix = np.zeros(shape=[len(users), len(items)])

topk=10
for data in output_r_matrix:
    userid, itemid, rating = data
    if rating>args.boundary_rating:
        binary_r_matrix[int(userid),int(itemid)] = 1.0
    elif rating<args.boundary_rating:
        binary_r_matrix[int(userid),int(itemid)] = -1.0
    else:
        raise NotImplementedError
boundary_user_id = int(len(users) * 0.8)
avg_rating = np.mean(binary_r_matrix[:boundary_user_id], axis=0)
# avg_rating = np.mean(binary_r_matrix[:], axis=0)
topk_items = np.argsort(avg_rating)[-topk:].tolist()
topk_items.reverse()
print("most popular (top-k) items:", topk_items)

# write rating matrix to file
def write_file(fname, data):
    print("Save data to %s" % fname)
    with open(fname, "w") as csvFile:
        if len(data) > 0:
            fw = csv.writer(csvFile, delimiter = "\t")
            for i in tqdm(range(len(data))):
                fw.writerow(data[i])

if not os.path.exists(output_path):
    os.makedirs(output_path)

# print output matrix shape
print("output r_matrix.shape", np.array(output_r_matrix).shape)

output_rating_file = os.path.join(output_path, 'ratings.txt')
write_file(output_rating_file, output_r_matrix)

# process userid to binary sensitive groups (gender)
# NOTE: uMap original user id to new user id

user_info_file = os.path.join(data_path, 'users.dat')
# format: UserID::Gender::Age::Occupation::Zip-code
user_data = pd.read_csv(user_info_file, 
    sep='::',
    names=['userId', 'gender', 'age', 'occupation', 'zipcode'],
    encoding='latin-1',
    engine='python',
)

print("user_data: ")
print(user_data)

print("-"*50 + "Handling sensitive group information" + "-"*50)

# step 1: userID to gender: M->0, F->1
# create user sensitive group mapping (e.g., {... 870: 'M', 871: 'M', 872: 'F', ...})
userID_gender_map = user_data[['userId', 'gender']].set_index('userId').to_dict()['gender']
print("before handling, user gender group distribution: ", Counter(list(userID_gender_map.values())))

# update gender mapping
gender_map = {
    'M': 0,
    'F': 1,
}
# updated_userID_gender_map = {int(uMap[user_id]): gender_map[gender] for user_id, gender in userID_gender_map.items()}
updated_userID_gender_map = {}
for user_id, gender in userID_gender_map.items():
    if user_id in uMap:
        updated_userID_gender_map[int(uMap[user_id])] = gender_map[gender]
print("After handling (update user id), user gender group distribution: ", Counter(list(updated_userID_gender_map.values())))
print("{'M': 0, 'F': 1}")

output_user_gender_group_file = os.path.join(output_path, "user_gender_group_map.json")
print("save user group info to %s" % output_user_gender_group_file)
with open(output_user_gender_group_file, 'w') as fw:
    json.dump(updated_userID_gender_map, fw, indent=2)


# step 2: process userid to binary sensitive groups (age)
userID_age_map = user_data[['userId', 'age']].set_index('userId').to_dict()['age']
print("before handling, user age group distribution: ", Counter(list(userID_age_map.values())))
# age_map = {
#     "0": {1, 18, 25},
#     "1": {35, 45, 50, 56},
# }
age_map = {
    "0": {1, 18, 25, 35},
    "1": {45, 50, 56},
}
updated_userID_age_map = {}
for user_id, age in userID_age_map.items():
    if user_id in uMap:
        updated_userID = int(uMap[user_id])
        if age in age_map["0"]:
            updated_userID_age_map[updated_userID] = 0
        elif age in age_map["1"]:
            updated_userID_age_map[updated_userID] = 1
print("After handling (update user id), user age group distribution: ", Counter(list(updated_userID_age_map.values())))
print("{'<35': 0, '>=35': 1}")

output_user_age_group_file = os.path.join(output_path, "user_age_group_map.json")
print("save user group info to %s" % output_user_age_group_file)
with open(output_user_age_group_file, 'w') as fw:
    json.dump(updated_userID_age_map, fw, indent=2)