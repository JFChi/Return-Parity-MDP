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
                    default=5, 
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
args = parser.parse_args()


# parameters go here
data_path = args.data_path
is_filter = True # filter the item with frequency less than XX
item_filter_threshold = args.item_filter_threshold
user_filter_threshold = args.user_filter_threshold
output_path = args.out_dir

# load rating data
rating_file = os.path.join(data_path, 'BX-Book-Ratings.csv')

rating_data = pd.read_csv(rating_file, 
    sep=';',
    names=['userId', 'itemId', 'rating'],
    dtype={'userId':np.int32,'itemId':np.str,'rating':np.float64},
    encoding='latin-1',
    header=0,
)

# for index, row in rating_data.iterrows():
#     print(index, row.userId, row.itemId, row.rating)
#     print(type(index), type(row.userId), type(row.itemId), type(row.rating))
#     break

print("Rating data: ")
print(rating_data)

# load user data
user_info_file = os.path.join(data_path, 'BX-Users.csv')
user_data = pd.read_csv(user_info_file, 
    sep=";",
    names=['userId', 'location', 'age'],
    encoding='latin-1',
    header=0,
)

print("User data: ")
print(user_data)

# find user where age is null
userID_with_null_age_set = set(user_data[user_data['age'].isna()]['userId'].unique())
print("\n# of user with NULL age: ", len(userID_with_null_age_set))


# create filter item id list
itemID_rating_cnt =  Counter(rating_data['itemId'].values.tolist())

remove_items = set()
for item_id, cnt in itemID_rating_cnt.items():
    if cnt < item_filter_threshold:
        remove_items.add(item_id)

print("\n# of total item: ", len(itemID_rating_cnt))
print("# of item to remove: ", len(remove_items))
print("# of item to remaining: ", len(itemID_rating_cnt)- len(remove_items))

# create filter item id list
userID_rating_cnt =  Counter(rating_data['userId'].values.tolist())
remove_users = set()
for user_id, cnt in userID_rating_cnt.items():
    if cnt < user_filter_threshold:
        remove_users.add(user_id)
    # filter user without age attribute
    if user_id in userID_with_null_age_set:
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
    user_id, item_id, rating = int(user_id), str(item_id), float(rating)
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
    v = str(filtered_r_matrix[i][1])

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
    v = str(filtered_r_matrix[i][1])
    rating = float(filtered_r_matrix[i][2])
    # make the rating into two catgories
    if rating > args.boundary_rating:
        rating = 1.0
    else:
        rating = 0.0

    output_r_matrix.append([uMap[u], iMap[v], rating])

# get most popular item from output_r_matrix
binary_r_matrix = np.zeros(shape=[len(users), len(items)])

topk=10
for data in output_r_matrix:
    userid, itemid, rating = data
    if rating>args.boundary_rating:
        binary_r_matrix[int(userid),int(itemid)] = 1.0
    elif rating<=args.boundary_rating:
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
userID_age_map = user_data[['userId', 'age']].set_index('userId').to_dict()['age']

updated_userID_age_map = {}
for user_id, age in userID_age_map.items():
    if not np.isnan(age) and user_id in uMap:
        updated_userID = int(uMap[user_id])
        updated_userID_age_map[updated_userID] = age
print("User age group distribution: ", dict(sorted(Counter(list(updated_userID_age_map.values())).items())))

age_threshold = 35
updated_binary_userID_age_map = {}
for user_id, age in updated_userID_age_map.items():
    if age < age_threshold:
        updated_binary_userID_age_map[user_id] = 0
    else:
        updated_binary_userID_age_map[user_id] = 1

print(len(updated_userID_age_map))


print("After handling (update user id), user age group distribution: ", Counter(list(updated_binary_userID_age_map.values())))
print(f"'<{age_threshold}': 0, '>={age_threshold}': 1")

output_user_age_group_file = os.path.join(output_path, "user_age_group_map.json")
print("save user group info to %s" % output_user_age_group_file)
with open(output_user_age_group_file, 'w') as fw:
    json.dump(updated_binary_userID_age_map, fw, indent=2)