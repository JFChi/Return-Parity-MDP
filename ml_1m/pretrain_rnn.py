import configparser
import argparse
import random
import numpy as np
import os
import json

import torch

from models import PretrainRNNModel
from env import Env


def update_params(optim, loss, retain_graph=False):
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)
    optim.step()

class PRE_TRAIN():
    def __init__(self, config, args):

        self.config = config
        self.log_step = int(self.config['META']['LOG_STEP'])
        self.pre_training_steps = int(self.config['PRE_TRAIN']['PRE_TRAINING_STEP'])

        self.action_dim = int(self.config['META']['ACTION_DIM'])
        self.reward_dim = int(self.config['META']['REWARD_DIM'])
        self.statistic_dim = int(self.config['META']['STATISTIC_DIM'])
        self.learning_rate = float(self.config['PRE_TRAIN']['LEARNING_RATE'])
        self.l2_factor = float(self.config['PRE_TRAIN']['L2_FACTOR'])
        # self.sampled_user = int(self.config['PRE_TRAIN']['SAMPLE_USER'])
        self.sample_episode = int(self.config['PRE_TRAIN']['SAMPLE_EPISODE'])
        self.pre_train_truncated_length = int(self.config['PRE_TRAIN']['PRE_TRAINING_RNN_TRUNCATED_LENGTH'])
        self.pre_train_seq_length = int(self.config['PRE_TRAIN']['PRE_TRAINING_SEQ_LENGTH'])
        self.pre_train_mask_length = int(self.config['PRE_TRAIN']['PRE_TRAINING_MASK_LENGTH'])
        self.boundary_rating = float(self.config['ENV']['BOUNDARY_RATING'])
        self.episode_length = int(self.config['META']['EPISODE_LENGTH'])

        self.rnn_input_dim = self.action_dim + self.reward_dim + self.statistic_dim
        self.rnn_output_dim = self.rnn_input_dim
        self.sensitive_group = args.sensitive_group

        # set seed
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)


        # env
        self.data_dir = args.data_dir
        self.forward_env = Env(self.config, self.data_dir)
        self.boundary_user_id = self.forward_env.boundary_user_id
        self.test_user_num = self.forward_env.test_user_num
        self.user_num, self.item_num, self.r_matrix, self.average_rate, self.user_to_rele_num = self.forward_env.get_init_data()

        self.env = [
            Env(
                config=self.config,
                user_num=self.user_num, 
                item_num=self.item_num, 
                r_matrix=self.r_matrix, 
                average_rate=self.average_rate, 
                user_to_rele_num=self.user_to_rele_num
            ) for _ in range(self.user_num)
        ] # clear env batch as large as possible

        # add rnn_file_path and define rnn models
        
        self.device = torch.device(
                    "cuda" if args.cuda and torch.cuda.is_available() else "cpu")


        # get train set user id by group
        self.g0_userID_train_set = set()
        self.g1_userID_train_set = set()
        train_group_info_json_file = os.path.join(self.data_dir, args.train_json_file)
        assert self.sensitive_group in train_group_info_json_file
        with open(train_group_info_json_file, 'r') as fr:
            train_userID_to_group = json.load(fr)
        train_userID_to_group = {int(userid): g for userid, g in train_userID_to_group.items()}

        for userid, g in train_userID_to_group.items():
            if g == 0:
                self.g0_userID_train_set.add(userid)
            elif g == 1:
                self.g1_userID_train_set.add(userid)
            else:
                raise NotImplementedError

        self.userID_train_set = self.g0_userID_train_set | self.g1_userID_train_set
        assert len(train_userID_to_group) == len(self.userID_train_set)
        print(f"# of training set: {len(self.userID_train_set)}")
        print(f"# of training set for g0 and g1: {len(self.g0_userID_train_set)}, {len(self.g1_userID_train_set)}")

        self.sampled_user_g0 = len(self.g0_userID_train_set)
        self.sampled_user_g1 = len(self.g1_userID_train_set)

        # assert env list has enough space for both users
        assert len(self.env) >= self.sample_episode * self.sampled_user_g0
        assert len(self.env) >= self.sample_episode * self.sampled_user_g1

        # define models and optimizier
        self.pretrainRNNModel_g0 = PretrainRNNModel(
            action_embeddings=self.forward_env.item_embedding,
            reward_dim=self.reward_dim,
            rnn_input_dim=self.rnn_input_dim, 
            rnn_output_dim=self.rnn_output_dim,
            statistic_dim=self.statistic_dim, 
            item_num=self.item_num,
            device=self.device,
        ).to(self.device)

        self.pretrainRNNModel_g1 = PretrainRNNModel(
            action_embeddings=self.forward_env.item_embedding,
            reward_dim=self.reward_dim,
            rnn_input_dim=self.rnn_input_dim, 
            rnn_output_dim=self.rnn_output_dim,
            statistic_dim=self.statistic_dim, 
            item_num=self.item_num,
            device=self.device,
        ).to(self.device)

        print(f"rnnCellModel arch: {self.pretrainRNNModel_g0}")
        
        
        self.optim_g0 = torch.optim.Adam(
            [param for param in self.pretrainRNNModel_g0.parameters() if param.requires_grad == True],
            lr=self.learning_rate,
        )
        self.optim_g1 = torch.optim.Adam(
            [param for param in self.pretrainRNNModel_g1.parameters() if param.requires_grad == True],
            lr=self.learning_rate,
        )

        self.best_eval_loss_g0 = float('inf')
        self.best_eval_loss_g1 = float('inf')
        self.rnn_g0_file_path = os.path.join(self.data_dir, "%s_pretrained_rnn_model_g0.pth" % self.sensitive_group)
        self.rnn_g1_file_path = os.path.join(self.data_dir, "%s_pretrained_rnn_model_g1.pth" % self.sensitive_group)

        print(f"rnn_g0_file_path: {self.rnn_g0_file_path}")
        print(f"rnn_g1_file_path: {self.rnn_g1_file_path}")
        
        # assertion
        assert self.forward_env.episode_length == self.pre_train_truncated_length \
            == self.pre_train_mask_length == self.pre_train_seq_length
        

    def train(self, group_idx):

        assert group_idx in ["g0", "g1"]

        # add group-wise variables
        train_group_userID_set = self.g0_userID_train_set if group_idx == "g0" \
            else  self.g1_userID_train_set
        sampled_user = self.sampled_user_g0 if group_idx == "g0" \
            else  self.sampled_user_g1
        pretrainRNNModel = self.pretrainRNNModel_g0 if group_idx == "g0" \
            else self.pretrainRNNModel_g1
        pretrainRNNModel.train()
        optim = self.optim_g0 if group_idx == "g0" \
            else self.optim_g1
        batch_size = self.sample_episode * sampled_user


        ### begin of vector env sampling ###
        # reset envs
        for i in range(sampled_user):
            user_id = random.sample(train_group_userID_set, k=1)[0]
            # user_id = random.randint(0, self.boundary_user_id - 1) #TODO: change here 
            for j in range(self.sample_episode):
                self.env[i * self.sample_episode + j].reset(user_id)

        # action, reward, user stats
        ars = [[],[],[]]
        # only consider first all items
        action_value_list = np.array([range(self.item_num) for i in range(batch_size)])
        [random.shuffle(action_value_list[i]) for i in range(batch_size)]
        action_value_list = action_value_list[:, :self.pre_train_seq_length]

        for i in range(self.pre_train_seq_length):
            sampled_action = action_value_list[:, i]
            ars[0].append([]) # ars[0].shape = (self.pre_train_seq_length, batch_size)
            ars[1].append([]) # ars[1].shape = (self.pre_train_seq_length, batch_size)
            ars[2].append([]) # ars[2].shape = (self.pre_train_seq_length, batch_size, user_stats_dim)
            for j in range(batch_size):
                obs, reward, _ = self.env[j].step(sampled_action[j])

                history_items, history_rewards, history_stats = obs
                # assert sampled_action[j] == history_items[-1]
                # assert reward == history_rewards[-1]
                # assert np.sum( np.array(self.env[j].get_statistic()) - history_stats[-1] ) < 1e-6
                ars[0][-1].append(history_items[-1])
                ars[1][-1].append(history_rewards[-1])
                ars[2][-1].append(history_stats[-1])
        ### end of vector env sampling ###


        # set parameters before training
        ground_truth = torch.zeros(batch_size, self.item_num, dtype=torch.float32).to(self.device)
        mask_value = torch.zeros(batch_size, self.item_num, dtype=torch.float32).to(self.device)
        # pre_rnn_state_list = [self.sess.run(self.initial_states)] (h0, c0)
        h0, c0 = torch.zeros(batch_size, self.rnn_input_dim).to(self.device),\
                torch.zeros(batch_size, self.rnn_input_dim).to(self.device)

    
        # begin training 
        running_loss = 0.0 # for logging 
        for i in range(self.pre_train_seq_length):
            actions = ars[0][i]
            rewards = np.array(ars[1][i])

            # (1) calculate reward ground truth, 
            # iterate over length of batch size
            for index in range(len(ground_truth)): 
                ground_truth[index][actions[index]] = rewards[index]
            
            # (2) mask 
            # iterate over length of batch size
            for index in range(len(mask_value)):
                mask_value[index][actions[index]] = 1.0
            if i >= self.pre_train_mask_length:
                raise NotImplementedError

            # (3) train rnn 
            if i < self.pre_train_truncated_length:
                pre_actions = torch.LongTensor(np.array(ars[0][:i+1])).to(self.device)
                pre_rewards = torch.FloatTensor(np.array(ars[1][:i+1])).to(self.device)
                pre_statistic = torch.FloatTensor(np.array(ars[2][:i+1])).to(self.device)

                input_states = (pre_actions, pre_rewards, pre_statistic)
                rnn_initial_states = (h0, c0) 

                pn_outputs, l2_norm = pretrainRNNModel(input_states, rnn_initial_states)

                assert len(pn_outputs) == i+1

                loss = (1.0/torch.sum(mask_value)) * torch.sum(torch.square(mask_value * (pn_outputs[i] - ground_truth)))
                loss = torch.pow(loss, 0.5)
                overall_loss = loss + self.l2_factor * l2_norm

                update_params(optim=optim, loss=overall_loss)

                running_loss += overall_loss.item()

                # print(mask_value.shape, ground_truth.shape, pn_outputs[i].shape)
                # print("loss", loss)
                # print("l2_norm", l2_norm)
                # print("overall_loss", overall_loss)

        running_loss = running_loss / self.pre_train_seq_length
        return running_loss

    def eval(self, group_idx):

        assert group_idx in ["g0", "g1"]

        eval_group_userID_set = self.g0_userID_train_set if group_idx == "g0" \
            else  self.g1_userID_train_set
        sampled_user = self.sampled_user_g0 if group_idx == "g0" \
            else  self.sampled_user_g1
        pretrainRNNModel = self.pretrainRNNModel_g0 if group_idx == "g0" \
            else self.pretrainRNNModel_g1
        pretrainRNNModel.eval()
        batch_size = self.sample_episode * sampled_user

        ### begin of vector env sampling ###
        # reset envs
        for i in range(sampled_user):
            user_id = random.sample(eval_group_userID_set, k=1)[0]
            # user_id = random.randint(0, self.boundary_user_id - 1)
            for j in range(self.sample_episode):
                self.env[i * self.sample_episode + j].reset(user_id)

        # action, reward, user stats
        ars = [[],[],[]]
        # only consider first all items
        action_value_list = np.array([range(self.item_num) for i in range(batch_size)])
        [random.shuffle(action_value_list[i]) for i in range(batch_size)]
        action_value_list = action_value_list[:, :self.pre_train_seq_length]

        for i in range(self.pre_train_seq_length):
            sampled_action = action_value_list[:, i]
            ars[0].append([]) # ars[0].shape = (self.pre_train_seq_length, batch_size)
            ars[1].append([]) # ars[1].shape = (self.pre_train_seq_length, batch_size)
            ars[2].append([]) # ars[2].shape = (self.pre_train_seq_length, batch_size, user_stats_dim)
            for j in range(batch_size):
                obs, reward, _ = self.env[j].step(sampled_action[j])

                history_items, history_rewards, history_stats = obs
                # assert sampled_action[j] == history_items[-1]
                # assert reward == history_rewards[-1]
                # assert np.sum( np.array(self.env[j].get_statistic()) - history_stats[-1] ) < 1e-6
                ars[0][-1].append(history_items[-1])
                ars[1][-1].append(history_rewards[-1])
                ars[2][-1].append(history_stats[-1])
        ### end of vector env sampling ###

        # set parameters before training
        ground_truth = torch.zeros(batch_size, self.item_num, dtype=torch.float32).to(self.device)
        mask_value = torch.zeros(batch_size, self.item_num, dtype=torch.float32).to(self.device)
        # pre_rnn_state_list = [self.sess.run(self.initial_states)] (h0, c0)
        h0, c0 = torch.zeros(batch_size, self.rnn_input_dim).to(self.device),\
                torch.zeros(batch_size, self.rnn_input_dim).to(self.device)

        # begin evaluate 
        with torch.no_grad():
            running_loss = 0.0 # for logging 
            for i in range(self.pre_train_seq_length):
                actions = ars[0][i]
                rewards = np.array(ars[1][i])

                # (1) calculate reward ground truth, 
                # iterate over length of batch size
                for index in range(len(ground_truth)): 
                    ground_truth[index][actions[index]] = rewards[index]
                
                # (2) mask 
                # iterate over length of batch size
                for index in range(len(mask_value)):
                    mask_value[index][actions[index]] = 1.0
                if i >= self.pre_train_mask_length:
                    raise NotImplementedError

                # (3) forward rnn 
                if i < self.pre_train_truncated_length:
                    pre_actions = torch.LongTensor(np.array(ars[0][:i+1])).to(self.device)
                    pre_rewards = torch.FloatTensor(np.array(ars[1][:i+1])).to(self.device)
                    pre_statistic = torch.FloatTensor(np.array(ars[2][:i+1])).to(self.device)

                    input_states = (pre_actions, pre_rewards, pre_statistic)
                    rnn_initial_states = (h0, c0) 


                    pn_outputs, l2_norm = pretrainRNNModel(input_states, rnn_initial_states)

                    # self.loss = [tf.pow((1.0 / tf.reduce_sum(self.mask)) * tf.reduce_sum(tf.square(self.mask * (self.pn_outputs[i] \
                    #     - self.expected_pn_outputs))), 0.5) for i in range(self.pre_train_truncated_length)]

                    assert len(pn_outputs) == i+1
                    

                    loss = (1.0/torch.sum(mask_value)) * torch.sum(torch.square(mask_value * (pn_outputs[i] - ground_truth)))
                    loss = torch.pow(loss, 0.5)
                    overall_loss = loss + self.l2_factor * l2_norm

                    running_loss += overall_loss.item()

                else:
                    raise NotImplementedError
            
            running_loss = running_loss / self.pre_train_seq_length

        return running_loss

        

    def run(self):
        print('start pre-training')
        # evaluate before pretraining
        eval_loss_g0 = self.eval(group_idx='g0')
        eval_loss_g1 = self.eval(group_idx='g1')      
        print(f'Before training -- eval -- g0 loss: {eval_loss_g0:.4f}, g1 loss: {eval_loss_g1:.4f}')

        for i in range(self.pre_training_steps):
            train_loss_g0 = self.train(group_idx='g0')
            train_loss_g1 = self.train(group_idx='g1')
            print(f'--train step {i}-- loss g0: {train_loss_g0:.4f}, loss g1: {train_loss_g1:.4f}')
            if i % 3 == 0 and i != 0:
                eval_loss_g0 = self.eval(group_idx='g0')
                eval_loss_g1 = self.eval(group_idx='g1')     
                print(f'-- eval -- g0 loss: {eval_loss_g0:.4f}, g1 loss: {eval_loss_g1:.4f}')

                # save rnn g0 weights
                if eval_loss_g0 < self.best_eval_loss_g0:
                    self.best_eval_loss_g0 = eval_loss_g0
                    print(f"Get best eval loss g0: {self.best_eval_loss_g0:.4f}")
                    print(f"Save rnn model to: {self.rnn_g0_file_path}")
                    torch.save(self.pretrainRNNModel_g0.rnn.state_dict(), self.rnn_g0_file_path)
                # save rnn g1 weights
                if eval_loss_g1 < self.best_eval_loss_g1:
                    self.best_eval_loss_g1 = eval_loss_g1
                    print(f"Get best eval loss g1: {self.best_eval_loss_g1:.4f}")
                    print(f"Save rnn model to: {self.rnn_g1_file_path}")
                    torch.save(self.pretrainRNNModel_g1.rnn.state_dict(), self.rnn_g1_file_path)
        print('end rnn pre-training')

def main():
    parser = argparse.ArgumentParser("Complementary arguments for pre-train")
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--config_file", type=str, help="config", default="config/dqn_config")
    parser.add_argument("--data_dir", help="(processed) data directory", type=str, default="data/processed/")
    parser.add_argument("--train_json_file", help="downsampled userID to user sensitive group", type=str, default="")
    parser.add_argument("--sensitive_group", help="sensitive group", type=str, default="gender")
    parser.add_argument("--log_dir", help="logging directory", type=str, default="logs")
    parser.add_argument("--log_fn", help="abs path name used to save the log file.", type=str, default="ml_1m_pretrain")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read_file(open(args.config_file))

    pre_train = PRE_TRAIN(config, args)
    pre_train.run()
    

if __name__ == '__main__':
    main()