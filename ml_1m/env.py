from utils import mf_with_bias, pickle_load, get_envobjects
import numpy as np
import os


class Env():
    def __init__(self, config, data_dir=None, user_num=-1, item_num=-1, r_matrix=-1, average_rate=-1, user_to_rele_num=-1):
        self.config = config
        self.episode_length = int(self.config['META']['EPISODE_LENGTH'])
        self.alpha = float(self.config['ENV']['ALPHA'])
        self.boundary_rating = float(self.config['ENV']['BOUNDARY_RATING'])
        self.action_dim = int(self.config['META']['ACTION_DIM'])
        self.emb_size = self.action_dim
        ratingfile = self.config['ENV']['RATING_FILE']
        self.max_rating = float(self.config['ENV']['MAX_RATING'])
        self.min_rating = float(self.config['ENV']['MIN_RATING'])
        self.a = 2.0 / (float(self.max_rating) - float(self.min_rating))
        self.b = - (float(self.max_rating) + float(self.min_rating)) / (float(self.max_rating) - float(self.min_rating))

        if user_num!=-1:
            self.user_num = user_num
            self.item_num = item_num
            self.r_matrix = r_matrix
            self.average_rate = average_rate
            self.user_to_rele_num = user_to_rele_num
        else:
            assert data_dir is not None
            self.env_object_path = os.path.join(data_dir,'env_objects')
            if not os.path.exists(self.env_object_path):
                get_envobjects(ratingfile=ratingfile, data_path=data_dir, boundary_rating=self.boundary_rating, emb_dim=int(self.config['META']['ACTION_DIM']))
            objects = pickle_load(self.env_object_path)
            self.r_matrix = objects['r_matrix']
            self.user_num = objects['user_num']
            self.item_num = objects['item_num']
            self.user_to_rele_num = objects['rela_num']
            self.average_rate = None

            self.item_embedding_file_path = os.path.join(data_dir, "item_embedding_dim_%d"%self.action_dim)
            if not os.path.exists(self.item_embedding_file_path):
                mf_with_bias(
                    rating_file=ratingfile, 
                    env_object_path=self.env_object_path, 
                    config=self.config,
                    data_path=data_dir,
                )
            self.item_embedding = np.loadtxt(self.item_embedding_file_path, dtype=float, delimiter='\t')

        self.boundary_user_id = int(self.user_num*0.8)
        self.test_user_num = self.user_num - self.boundary_user_id

    def get_init_data(self):
         return self.user_num, self.item_num, self.r_matrix, self.average_rate, self.user_to_rele_num

    def reset(self, user_id):
        self.user_id = user_id
        self.step_count = 0
        self.con_neg_count = 0
        self.con_pos_count = 0
        self.con_zero_count = 0
        self.con_not_neg_count = 0
        self.con_not_pos_count = 0
        self.all_neg_count = 0
        self.all_pos_count = 0

        self.history_items = []
        self.history_rewards = []
        self.history_stats = []

    def step(self, item_id):
        reward = 0.0
        done = False
        r = self.r_matrix[self.user_id, item_id]
        # clip reward to [min_rating+1, max_rating]
        r = np.clip(r, a_min=self.min_rating+1.0, a_max=self.max_rating)
        reward = self.a * r + self.b

        self.step_count += 1
        sr = self.con_pos_count - self.con_neg_count
        if reward < 0:
            self.con_neg_count += 1
            self.all_neg_count += 1
            self.con_not_pos_count += 1
            self.con_pos_count = 0
            self.con_not_neg_count = 0
            self.con_zero_count = 0
        elif reward > 0:
            self.con_pos_count += 1
            self.all_pos_count += 1
            self.con_not_neg_count += 1
            self.con_neg_count = 0
            self.con_not_pos_count = 0
            self.con_zero_count = 0
        else:
            self.con_not_neg_count += 1
            self.con_not_pos_count += 1
            self.con_zero_count += 1
            self.con_pos_count = 0
            self.con_neg_count = 0

        if self.step_count == self.episode_length or len(self.history_items) == self.item_num:
            done = True

        reward += self.alpha * sr
        # clip reward to avoid ont-hot overflow
        reward = np.clip(reward, a_min=-2+1e-5, a_max=2)


        self.history_items.append(item_id)
        self.history_rewards.append(reward) # normalized reward
        self.history_stats.append(self.get_statistic())

        obs = [ 
            np.array(self.history_items),
            np.array(self.history_rewards),
            np.array(self.history_stats),
        ]

        return obs, reward, done

    def get_rating(self, item_id):
        return self.r_matrix[self.user_id, item_id]

    def get_relevant_item_num(self):
        return self.user_to_rele_num[self.user_id]

    def get_statistic(self):
        all_neg_count = self.all_neg_count
        all_pos_count = self.all_pos_count
        step_count = self.step_count
        con_neg_count = self.con_neg_count
        con_pos_count = self.con_pos_count
        all_count = len(self.history_items)
        zero_count = all_count - all_neg_count - all_pos_count
        con_zero_count = self.con_zero_count
        con_not_neg_count = self.con_not_neg_count
        con_not_pos_count = self.con_not_pos_count
        result = [all_neg_count, all_pos_count, zero_count, step_count, con_neg_count, con_pos_count, con_zero_count, con_not_neg_count, con_not_pos_count]
        return [item * 1.0 / self.episode_length for item in result]

