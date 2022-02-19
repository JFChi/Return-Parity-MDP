import argparse
import numpy as np
import random 
import os 
import json

parser = argparse.ArgumentParser(description='preprocess ml 100k')
parser.add_argument("--data_path", 
                    type=str,
                    default='data/processed/ml-1m/', 
                    help="",
)
parser.add_argument("--out_dir", 
                    type=str,
                    default='data/processed/ml-1m/', 
                    help="",
)
parser.add_argument("--input_json_file", 
                    type=str,
                    default='user_gender_group_map.json', 
                    help="",
)
parser.add_argument("--seed", 
                    type=int,
                    default=0, 
                    help="seed",
)
parser.add_argument("--g0_train_ratio", 
                    type=float,
                    default=1.0, 
                    help="ratio to keep in g0",
)
parser.add_argument("--g1_train_ratio", 
                    type=float,
                    default=1.0, 
                    help="ratio to keep in g1",
)

args = parser.parse_args()

# set seed
random.seed(args.seed)
np.random.seed(args.seed)

# get user sensitive group information ##
group_info_json_file = os.path.join(args.data_path, args.input_json_file)
with open(group_info_json_file, 'r') as fr:
    userID_to_group = json.load(fr)
userID_to_group =  {int(userid): g for userid, g in userID_to_group.items()}

boundary_user_id = int(len(userID_to_group) * 0.8)

g0_userID_train_set = set()
g1_userID_train_set = set()

for userid, g in userID_to_group.items():
    if g == 0 and userid < boundary_user_id:
        g0_userID_train_set.add(userid)
    elif g == 1 and userid < boundary_user_id:
        g1_userID_train_set.add(userid)

assert len(g0_userID_train_set)+ len(g1_userID_train_set) == boundary_user_id

assert args.g0_train_ratio >= 0.0 and args.g0_train_ratio <= 1.0
assert args.g0_train_ratio >= 0.0 and args.g0_train_ratio <= 1.0

print("args.g0_train_ratio: ", args.g0_train_ratio)
print("args.g1_train_ratio: ", args.g1_train_ratio)

# generate training set (user id) for both groups
num_train_g0 = int( len(g0_userID_train_set) * args.g0_train_ratio )
num_train_g1 = int( len(g1_userID_train_set) * args.g1_train_ratio )
 
print("Before downsampling, num_train_g0, num_train_g1: ", 
    len(g0_userID_train_set),
    len(g1_userID_train_set),
)
print("After downsampling num_train_g0, num_train_g1: ", num_train_g0, num_train_g1)
print("# of train now: ", num_train_g0+num_train_g1)

sampled_g0_userID_train_set = random.sample(g0_userID_train_set, num_train_g0)
sampled_g1_userID_train_set = random.sample(g1_userID_train_set, num_train_g1)

# assert train set should have both group data
assert len(sampled_g0_userID_train_set) != 0
assert len(sampled_g1_userID_train_set) != 0
 
sampled_train_set = {}
sampled_train_set.update({userid: 0 for userid in sampled_g0_userID_train_set})
sampled_train_set.update({userid: 1 for userid in sampled_g1_userID_train_set})

assert len(sampled_train_set) == num_train_g0 + num_train_g1

# sort train set based on userid (key)
sampled_train_set = dict(sorted(sampled_train_set.items()))

output_json_file = f"downsampled_trainset_{os. path. splitext(args.input_json_file)[0]}_g0_{args.g0_train_ratio:.2f}_g1_{args.g1_train_ratio:.2f}_seed_{args.seed}.json"
output_json_file = os.path.join(args.out_dir, output_json_file)
print("save user group info to %s" % output_json_file)
with open(output_json_file, 'w') as fw:
    json.dump(sampled_train_set, fw, indent=2)


