
import numpy as np
import random
import os
import gc
import math
import json
import itertools
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from env import Env
from models import MLP
from utils import get_logger, EpochLogger


class ReplayBuffer():
    def __init__(self, config):
        self.config = config
        self.buffer_size = int(self.config['DQN']['BUFFER_SIZE'])
        self.batch_size = int( int(self.config['DQN']['UPDATE_BATCH']) / 2 )
        self.action_dim = int(self.config['META']['ACTION_DIM'])
        self.statistic_dim = int(self.config['META']['STATISTIC_DIM'])
        self.reward_dim = int(self.config['META']['REWARD_DIM'])
        self.state_dim = self.action_dim + self.reward_dim + 2 * self.statistic_dim

        # s1 a1 r1 s2
        self.buffer = [np.zeros(shape=[self.buffer_size, self.state_dim]),
                       np.zeros(shape=[self.buffer_size]),
                       np.zeros(shape=[self.buffer_size]),
                       np.zeros(shape=[self.buffer_size, self.state_dim])]
        self.index = -1
        self.full = False

    def get_size(self):
        return self.index+1

    # one (s, a, r, s_) pair per time
    def add(self, sample):
        self.index = (self.index+1)%self.buffer_size
        if not self.full and self.index == (self.buffer_size - 1):
            self.full = True
        for i in range(4):
            self.buffer[i][self.index] = sample[i]

    def sample(self):
        max_seed = self.buffer_size
        if not self.full:
            max_seed = self.index+1
        seeds = list(range(0, max_seed))
        if not self.full and self.index<(self.batch_size-1):
            seeds = [num % (self.index+1) for num in range(0, self.batch_size)]
        random.shuffle(seeds)
        seeds = seeds[:self.batch_size]
        result = []
        for i in range(4):
            result.append(self.buffer[i][seeds])
        return result

def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data * (1.0 - tau) + s.data * tau)

def update_params(optim, loss, retain_graph=False):
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)
    optim.step()


class DQN():
    def __init__(self, config, args):
        self.config = config
        self.args = args

        torch.backends.cudnn.benchmark = False
        torch.cuda.synchronize()

        self.buffer_g0 = ReplayBuffer(self.config)
        self.buffer_g1 = ReplayBuffer(self.config)

        self.max_training_step = int(config['META']['MAX_TRAINING_STEP'])
        self.log_step = int(self.config['META']['LOG_STEP'])
        
        self.action_dim = int(self.config['META']['ACTION_DIM'])
        self.reward_dim = int(self.config['META']['REWARD_DIM'])
        self.statistic_dim = int(self.config['META']['STATISTIC_DIM'])
        self.discount_factor = float(self.config['META']['DISCOUNT_FACTOR'])
        self.episode_length = int(self.config['META']['EPISODE_LENGTH'])

        self.learning_rate = float(self.config['DQN']['LEARNING_RATE'])
        self.l2_factor = float(self.config['DQN']['L2_FACTOR'])
        self.tau = float(self.config['DQN']['TAU'])
        self.max_epsilon = float(self.config['DQN']['MAX_EPSILON'])
        self.min_epsilon = float(self.config['DQN']['MIN_EPSILON'])
        self.decay_step = int(self.config['DQN']['DECAY_STEP'])
        self.sample_batch = int(self.config['DQN']['SAMPLE_BATCH'])
        self.update_batch = int(self.config['DQN']['UPDATE_BATCH'])
        self.update_times = int(self.config['DQN']['UPDATE_TIMES'])
        self.sample_times = int(self.config['DQN']['SAMPLE_TIMES'])
        self.hidden_units = int(self.config['DQN']['HIDDEN_UNITS_1'])
        self.boundary_rating = float(self.config['ENV']['BOUNDARY_RATING'])

        self.epsilon = self.max_epsilon
        self.training_steps = 0 # +1 at each training epoch
        self.update_steps = 0 # +1 at each update step
        self.data_dir = args.data_dir
        self.sensitive_group = args.sensitive_group

        # add logging 
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        args.log_fn = "%s_%d" % (args.log_fn, args.seed)
        args.log_fn = os.path.join(args.log_dir, args.log_fn)
        self.logger = get_logger(args.log_fn)
        self.log_fn = args.log_fn
        self.epoch_logger = EpochLogger()

        # set seed
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        # TODO: save model for evaluation
        
        # env
        self.forward_env = Env(self.config, self.data_dir)
        self.logger.info(f"Loading env object from {self.forward_env.env_object_path}")
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
            ) for _ in range(max(self.sample_batch, self.update_batch))
        ]        

        self.device = torch.device(
                    "cuda" if args.cuda and torch.cuda.is_available() else "cpu")


        ## get user train/test sensitive group information ##
        assert self.sensitive_group in args.group_json_file
        group_info_json_file = os.path.join(self.data_dir, args.group_json_file)
        with open(group_info_json_file, 'r') as fr:
            userID_to_group = json.load(fr)
        userID_to_group =  {int(userid): g for userid, g in userID_to_group.items()}

        # get test set user id by group
        self.g0_userID_test_set = set()
        self.g1_userID_test_set = set()
        for userid, g in userID_to_group.items():
            if g == 0 and userid >= self.forward_env.boundary_user_id:
                self.g0_userID_test_set.add(userid)
            elif g == 1 and userid >= self.forward_env.boundary_user_id:
                self.g1_userID_test_set.add(userid)
            elif g == 0 and userid < self.forward_env.boundary_user_id:
                pass
            elif g == 1 and userid < self.forward_env.boundary_user_id:
                pass
            else:
                raise NotImplementedError 

        assert len(userID_to_group) == self.user_num
        assert self.test_user_num == len(self.g0_userID_test_set)+ \
            len(self.g1_userID_test_set)

        # get train set user id by group
        assert self.sensitive_group in args.train_json_file
        self.g0_userID_train_set = set()
        self.g1_userID_train_set = set()
        train_group_info_json_file = os.path.join(self.data_dir, args.train_json_file)
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
        self.logger.info(f"# of training set: {len(self.userID_train_set)}")
        self.logger.info(f"# of training set for g0 and g1: {len(self.g0_userID_train_set)}, {len(self.g1_userID_train_set)}")

        # action embedding
        self.action_embeddings = nn.Embedding.from_pretrained(
            torch.FloatTensor(self.forward_env.item_embedding)
        ).to(self.device).eval()
        self.action_embeddings.requires_grad = False
        self.logger.info(f"Loading pretrained item embedding from {self.forward_env.item_embedding_file_path}")

        # rnn
        # self.rnn_file_path = os.path.join(self.data_dir, "pretrained_rnn_model.pth")
        self.rnn_g0_file_path = os.path.join(self.data_dir, "%s_pretrained_rnn_model_g0.pth" % self.sensitive_group)
        self.rnn_g1_file_path = os.path.join(self.data_dir, "%s_pretrained_rnn_model_g1.pth" % self.sensitive_group)
        self.rnn_input_dim = self.action_dim + self.reward_dim + self.statistic_dim
        self.rnn_output_dim = self.rnn_input_dim
        self.state_dim = self.rnn_input_dim + self.statistic_dim

        self.rnn_g0 = nn.LSTMCell(self.rnn_input_dim, self.rnn_output_dim).to(self.device).eval()
        self.rnn_g1 = nn.LSTMCell(self.rnn_input_dim, self.rnn_output_dim).to(self.device).eval()

        self.rnn_g0.load_state_dict(torch.load(self.rnn_g0_file_path))
        self.rnn_g1.load_state_dict(torch.load(self.rnn_g1_file_path))
        self.logger.info(f"loading RNN g0 model from: {self.rnn_g0_file_path}")
        self.logger.info(f"loading RNN g1 model from: {self.rnn_g1_file_path}")
        self.logger.info(f"RNN model arch: {self.rnn_g0}")

        # add group wise mlp state feature extractors and add it to qnet_optim
        self.g0_feat_ext = MLP(
            input_dim=self.state_dim, 
            output_dim=self.state_dim, 
            units=[self.hidden_units],
        ).to(self.device)
        self.g1_feat_ext = MLP(
            input_dim=self.state_dim, 
            output_dim=self.state_dim, 
            units=[self.hidden_units],
        ).to(self.device)

        # q-net
        self.qnet = MLP(
            input_dim=self.state_dim, 
            output_dim=self.item_num, 
            units=[self.hidden_units],
        ).to(self.device)
        self.target_qnet = MLP(
            input_dim=self.state_dim, 
            output_dim=self.item_num, 
            units=[self.hidden_units],
        ).to(self.device).eval()

        self.logger.info(f"feat ext arch: {self.g0_feat_ext}")
        self.logger.info(f"qnet arch: {self.qnet}")

        # Copy parameters of the learning network to the target network.
        self.target_qnet.load_state_dict(self.qnet.state_dict())

        # define optim
        qnet_optim_params = [self.qnet.parameters(), self.g0_feat_ext.parameters(), self.g1_feat_ext.parameters()]
        self.qnet_optim = torch.optim.Adam(itertools.chain(*qnet_optim_params), lr=self.learning_rate)

        # add wass-specific parameters
        self.wass_update_interval = args.wass_update_interval
        self.wass_batch_size = args.wass_batch_size
        # wass_optim_params = [self.g0_feat_ext.parameters(), self.g1_feat_ext.parameters()]
        self.wass_optim_g0 = torch.optim.Adam(self.g0_feat_ext.parameters(), lr=self.learning_rate)
        self.wass_optim_g1 = torch.optim.Adam(self.g1_feat_ext.parameters(), lr=self.learning_rate)

        self.wass_critic = nn.Linear(
            in_features=self.state_dim, 
            out_features=1,
        ).to(self.device)
        self.wass_critic_optim = torch.optim.Adam(self.wass_critic.parameters(), lr=self.learning_rate) # for distribution alignment
        self.logger.info(f"wass critic arch: {self.wass_critic}")
        self.logger.info(f"wass_update_interval: {self.wass_update_interval}")

    def run(self):
        '''train + eval
        '''
        for i in range(0, self.max_training_step):
            # evaluate
            if i % self.log_step == 0:
                # evaluate by groups
                g0_reward_arr = self.evaluate(group_idx="g0")
                g1_reward_arr = self.evaluate(group_idx="g1")
                overall_reward_arr = np.concatenate((g0_reward_arr, g1_reward_arr), axis=0)

                test_g0_avg_return = np.mean(np.sum(g0_reward_arr, axis=1))
                test_g1_avg_return = np.mean(np.sum(g1_reward_arr, axis=1))
                test_avg_return = np.mean(np.sum(overall_reward_arr, axis=1))

                ## logging: used for wass alignment
                self.curr_test_g0_avg_return = test_g0_avg_return
                self.curr_test_g1_avg_return = test_g1_avg_return 


                self.logger.info("-"*100)
                self.logger.info(f"test g0 avg return: {test_g0_avg_return:.5f}, "
                    f"test g1 avg return: {test_g1_avg_return:.5f}, "
                    f"test avg return: {test_avg_return:.5f} "
                )
                self.logger.info(f"test g0 avg reward per step: {np.mean(g0_reward_arr):.5f}, "
                    f"test g1 avg reward per step: {np.mean(g1_reward_arr):.5f}, "
                    f"test avg reward per step: {np.mean(overall_reward_arr):.5f}"
                )
                self.logger.info(f"return disparity {np.abs(test_g0_avg_return - test_g1_avg_return):.5f}")
                self.logger.info("-"*100)

                self.epoch_logger.store(
                    steps=i, # training_step
                    avg_ret_g0=test_g0_avg_return,
                    avg_ret_g1=test_g1_avg_return,
                    avg_tot_ret=test_avg_return,
                    avg_ret_gap=np.abs(test_g0_avg_return - test_g1_avg_return),
                )
            # train
            self.train()
        
        # save after training
        self.save_stats()

    def train(self):
        '''update + sample + epsilon decay
        '''
        for _ in range(self.sample_times):
            # sample by group, each sampled batch size / 2
            self.sample(group_idx="g0") 
            self.sample(group_idx="g1")
        running_loss = 0.0
        for i in range(self.update_times):
            loss = self.update()
            running_loss += loss
            # self.logger.info(f"In training step {self.training_steps}, batch {i}, loss = {loss: .4f}")
        running_loss = running_loss / self.update_times
        self.training_steps += 1
        if self.training_steps<=self.decay_step:
            self.epsilon -= ((self.max_epsilon-self.min_epsilon)/self.decay_step)
        self.logger.info(f'Finish training step {self.training_steps}, loss = {running_loss:.4f}, epsilon = {self.epsilon:.6f}')
    
    def update(self):
        '''Update model
        '''

        self.update_steps += 1

        # update target q network: soft assign parameters in q-net to target q-net
        soft_update(self.target_qnet, self.qnet, self.tau)

        # update q now !
        states_g0, actions_g0, rewards_g0, next_states_g0 = self.buffer_g0.sample()
        states_g1, actions_g1, rewards_g1, next_states_g1 = self.buffer_g1.sample()

        # to pytorch tensor first 
        states_g0 = torch.FloatTensor(states_g0).to(self.device)
        actions_g0 = torch.LongTensor(actions_g0).to(self.device)
        rewards_g0 = torch.FloatTensor(rewards_g0).to(self.device)
        next_states_g0 = torch.FloatTensor(next_states_g0).to(self.device)

        states_g1 = torch.FloatTensor(states_g1).to(self.device)
        actions_g1 = torch.LongTensor(actions_g1).to(self.device)
        rewards_g1 = torch.FloatTensor(rewards_g1).to(self.device)
        next_states_g1 = torch.FloatTensor(next_states_g1).to(self.device)

        # combine group to train
        # transform states to state feats
        state_feats_g0, state_feats_g1 = self.g0_feat_ext(states_g0), self.g1_feat_ext(states_g1)
        state_feats = torch.cat((state_feats_g0, state_feats_g1), dim=0)
        actions = torch.cat((actions_g0, actions_g1), dim=0)
        rewards = torch.cat((rewards_g0, rewards_g1), dim=0)
        # states = torch.cat((states_g0, states_g1), dim=0)
        # next_states = torch.cat((next_states_g0, next_states_g1), dim=0)

        # get current q value from the current states
        q_value = self.qnet(state_feats).gather(dim=1, index=actions.unsqueeze(1))

        # get target q value
        with torch.no_grad():
            next_state_feats_g0, next_state_feats_g1 = self.g0_feat_ext(next_states_g0), self.g1_feat_ext(next_states_g1)
            next_state_feats = torch.cat((next_state_feats_g0, next_state_feats_g1), dim=0)
            max_q, _ = torch.max(self.target_qnet(next_state_feats), dim=1)
            target_q_value = (rewards + self.discount_factor * max_q).unsqueeze(1)

        
        # l2 regularization for q-net
        l2_loss = self.get_l2_norm(self.qnet)
        loss = F.mse_loss(q_value, target_q_value) + self.l2_factor * l2_loss
        update_params(self.qnet_optim, loss)

       # update wass loss after updating other components
        # in each wass update step size (in update function), first do the sampling by groups (under no grad env), evaluate their return,
        # after collect the states, pass them to the feat ext and train
        if self.update_steps % self.wass_update_interval == 0:
    
            states_g0_for_alignment = self.sample_for_alignment(group_idx='g0')
            states_g1_for_alignment = self.sample_for_alignment(group_idx='g1')

            states_g0_for_alignment = torch.FloatTensor(states_g0_for_alignment).to(self.device)
            states_g1_for_alignment = torch.FloatTensor(states_g1_for_alignment).to(self.device)

            # only update wass critic first
            wass_score_g0, wass_score_g1, gradient_penalty = self.calc_wass_loss(
                states_g0=states_g0_for_alignment,
                states_g1=states_g1_for_alignment,
            )
            ## Loss weight for gradient penalty
            lambda_gp = 10
            wass_critic_loss =  wass_score_g1 - wass_score_g0 + lambda_gp * gradient_penalty
            update_params(self.wass_critic_optim, wass_critic_loss)

            # only update one of the feature extractors
            wass_score_g0, wass_score_g1 = self.calc_wass_loss(
                states_g0=states_g0_for_alignment,
                states_g1=states_g1_for_alignment, 
                grad_penalty=False
            ) 
            wass_feat_ext_loss =  wass_score_g0 - wass_score_g1

            if self.curr_test_g0_avg_return > self.curr_test_g1_avg_return:
                # g0 is the privileged group now
                update_params(self.wass_optim_g0, wass_feat_ext_loss)
            else:
                # g1 is the privileged group now
                update_params(self.wass_optim_g1, wass_feat_ext_loss)

            if self.update_steps % 20 == 0:
                self.logger.info(f"In update step {self.update_steps}, wass score diff: {wass_feat_ext_loss.item():.4f}")    
            
        return loss.item()
        
    def calc_wass_loss(self, states_g0, states_g1, grad_penalty=True):

        state_g0_feats = self.g0_feat_ext(states_g0)
        state_g1_feats = self.g1_feat_ext(states_g1)

        wass_score_g0, wass_score_g1 = self.wass_critic(state_g0_feats).mean(), \
            self.wass_critic(state_g1_feats).mean()

        if grad_penalty:
            # compute gradient penalty
            alpha = torch.rand(state_g0_feats.size(0), 1).to(self.device)
            # Get random interpolation between real and fake samples
            interpolates = (alpha * state_g0_feats + ((1 - alpha) * state_g1_feats)).requires_grad_(True)
            interpolates_score = self.wass_critic(interpolates)
            # Get gradient w.r.t. interpolates
            gradients = torch.autograd.grad(outputs=interpolates_score,
                                            inputs=interpolates,
                                            grad_outputs=torch.ones(interpolates_score.size()).to(self.device),
                                            create_graph=True, 
                                            retain_graph=True)[0]
            gradients = gradients.view(gradients.size(0), -1)
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            return wass_score_g0, wass_score_g1, gradient_penalty
        else:
            return wass_score_g0, wass_score_g1

    def sample(self, group_idx):
        '''sampling
        '''
        assert group_idx in ["g0", "g1"]

        # group-wise variable
        group_sample_batch = int(self.sample_batch / 2)
        userID_train_set = self.g0_userID_train_set if group_idx == "g0" else self.g1_userID_train_set
        rnn_model = self.rnn_g0 if group_idx == "g0" else self.rnn_g1
        buffer = self.buffer_g0 if group_idx == "g0" else self.buffer_g1
        feat_ext = self.g0_feat_ext if group_idx == "g0" else self.g1_feat_ext


        for i in range(group_sample_batch):
            # sample group-wise user_id 
            user_id = random.sample(userID_train_set, k=1)[0]
            assert user_id < self.boundary_user_id
            self.env[i].reset(user_id)

        ars = self._get_init_ars(group_sample_batch)
        # ars[0].shape = (seq_length, group_sample_batch)
        # ars[1].shape = (seq_length, group_sample_batch)
        # ars[2].shape = (seq_length, group_sample_batch, user_stats_dim)

        # rnn_state
        h0, c0 = torch.zeros(group_sample_batch, self.rnn_input_dim).to(self.device), \
                torch.zeros(group_sample_batch, self.rnn_input_dim).to(self.device)
        rnn_state = (h0, c0)
        # if item already used, set it to 1e30
        sampled_action_mask = torch.zeros(group_sample_batch, self.item_num, dtype=torch.float32).to(self.device)
        sample_action_sets = [[] for _ in range(group_sample_batch)]
        for i in range(group_sample_batch):
            item_used = ars[0][0][i] # 1st 0: action, 2nd 0: first action, i: batch id
            sampled_action_mask[i][item_used] = 1e30
            sample_action_sets[i].append(item_used)

        step_count_list = [0 for i in range(group_sample_batch)]
        stop_count = [0 for i in range(group_sample_batch)]
        state_list = []

        step_count = 0
        while True:
            rnn_state, tmp_state = self.get_states_from_obs(
                actions=ars[0][step_count], 
                rewards=ars[1][step_count], 
                statistic=ars[2][step_count], 
                rnn_state=rnn_state,
                rnn_model=rnn_model,
            )
            state_list.append(tmp_state.cpu().numpy())

            tmp_state_feat = feat_ext(tmp_state)
            argmax_action = torch.argmax(self.qnet(tmp_state_feat) - sampled_action_mask, dim=-1)

            ars[0].append([])
            ars[1].append([])
            ars[2].append([])
            step_count += 1
            for j in range(group_sample_batch):
                tmp_action = argmax_action[j] # choose action from policy 
                if random.random()<self.epsilon: # or choose action randomly 
                    tmp_action = self.get_random_action(sample_action_sets[j])
                if tmp_action >= self.item_num:
                    raise ValueError("action index is out of range")
                sampled_action_mask[j][tmp_action] = 1e30
                sample_action_sets[j].append(tmp_action)
                obs, reward, done = self.env[j].step(tmp_action)
                history_items, history_rewards, history_stats = obs

                ars[0][-1].append(history_items[-1])
                ars[1][-1].append(history_rewards[-1])
                ars[2][-1].append(history_stats[-1])

                # set the current env as done 
                if done and stop_count[j] != 1:
                    step_count_list[j] = step_count
                    stop_count[j] = 1
            # test whether all envs are finished
            if np.sum(stop_count) == group_sample_batch:
                break

        for i in range(len(state_list)-1):
            for j in range(group_sample_batch):
                state = state_list[i][j]
                next_state = state_list[i+1][j]
                action = ars[0][i+1][j]
                reward = ars[1][i+1][j]
                buffer.add([state, action, reward, next_state])

        del sampled_action_mask
        del ars
        gc.collect()

    def sample_for_alignment(self, group_idx):
        '''sampling for distriution alignment
        '''
        assert group_idx in ["g0", "g1"]

        group_sample_batch = self.wass_batch_size
        userID_train_set = self.g0_userID_train_set if group_idx == "g0" else self.g1_userID_train_set
        rnn_model = self.rnn_g0 if group_idx == "g0" else self.rnn_g1
        feat_ext = self.g0_feat_ext if group_idx == "g0" else self.g1_feat_ext

        reward_list = []

        for i in range(group_sample_batch):
            # sample group-wise user_id 
            user_id = random.sample(userID_train_set, k=1)[0]
            assert user_id < self.boundary_user_id
            self.env[i].reset(user_id)

        ars = self._get_init_ars(group_sample_batch)
        # ars[0].shape = (seq_length, group_sample_batch)
        # ars[1].shape = (seq_length, group_sample_batch)
        # ars[2].shape = (seq_length, group_sample_batch, user_stats_dim)

        # rnn_state
        h0, c0 = torch.zeros(group_sample_batch, self.rnn_input_dim).to(self.device), \
                torch.zeros(group_sample_batch, self.rnn_input_dim).to(self.device)
        rnn_state = (h0, c0)
        # if item already used, set it to 1e30
        sampled_action_mask = torch.zeros(group_sample_batch, self.item_num, dtype=torch.float32).to(self.device)
        sample_action_sets = [[] for _ in range(group_sample_batch)]
        for i in range(group_sample_batch):
            item_used = ars[0][0][i] # 1st 0: action, 2nd 0: first action, i: batch id
            sampled_action_mask[i][item_used] = 1e30
            sample_action_sets[i].append(item_used)

        step_count_list = [0 for i in range(group_sample_batch)]
        stop_count = [0 for i in range(group_sample_batch)]
        state_list = []

        step_count = 0
        while True:
            rnn_state, tmp_state = self.get_states_from_obs(
                actions=ars[0][step_count], 
                rewards=ars[1][step_count], 
                statistic=ars[2][step_count], 
                rnn_state=rnn_state,
                rnn_model=rnn_model,
            )
            state_list.append(tmp_state.cpu().numpy())

            tmp_state_feat = feat_ext(tmp_state)
            argmax_action = torch.argmax(self.qnet(tmp_state_feat) - sampled_action_mask, dim=-1)

            ars[0].append([])
            ars[1].append([])
            ars[2].append([])
            step_count += 1
            for j in range(group_sample_batch):
                tmp_action = argmax_action[j] # choose action from policy
                if tmp_action >= self.item_num:
                    raise ValueError("action index is out of range")
                sampled_action_mask[j][tmp_action] = 1e30
                sample_action_sets[j].append(tmp_action)
                obs, reward, done = self.env[j].step(tmp_action)
                history_items, history_rewards, history_stats = obs

                ars[0][-1].append(history_items[-1])
                ars[1][-1].append(history_rewards[-1])
                ars[2][-1].append(history_stats[-1])

                # set the current env as done 
                if done and stop_count[j] != 1:
                    step_count_list[j] = step_count
                    stop_count[j] = 1
            # test whether all envs are finished
            if np.sum(stop_count) == group_sample_batch:
                break

        # for j in range(group_sample_batch):
        #     rewards = [ars[1][k][j] for k in range(0, self.episode_length)]
        #     reward_list.append(rewards)
        # reward_arr = np.array(reward_list)

        visitations = np.array(state_list).reshape(-1, self.state_dim) # exclude the first state

        return visitations
        

    def get_states_from_obs(self, actions, rewards, statistic, rnn_state, rnn_model):

        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        statistic = torch.FloatTensor(statistic).to(self.device)

        with torch.no_grad():
            action_embeds = self.action_embeddings(actions)
            # action_embeds.shape = (batch_size, action_dim)
            one_hot_rewards = F.one_hot(
                torch.floor(self.reward_dim * (2.0-rewards) / 4.0).to(torch.int64),
                num_classes=self.reward_dim,
            )
            # one_hot_rewards.shape = (batch_size, reward_dim)
            ars = torch.cat((action_embeds, one_hot_rewards, statistic), dim=-1)

            # input to rnn
            next_rnn_state = rnn_model(ars, rnn_state)
            state = torch.cat((next_rnn_state[0], statistic), dim=-1)
            return next_rnn_state, state

    def evaluate(self, group_idx):

        assert group_idx in ["g0", "g1"]
        eval_group_userid_set = self.g0_userID_test_set if group_idx == "g0" \
            else self.g1_userID_test_set
        rnn_model = self.rnn_g0 if group_idx == "g0" else self.rnn_g1
        feat_ext = self.g0_feat_ext if group_idx == "g0" else self.g1_feat_ext
        
        # different groups use different RNN
        reward_list = []
        group_eval_batch_size = len(eval_group_userid_set)
        assert group_eval_batch_size <= self.sample_batch

        for i, user_id in enumerate(eval_group_userid_set):
            self.env[i].reset(user_id)
            
        ars = self._get_init_ars(group_eval_batch_size)
        # ars[0].shape = (seq_length, group_eval_batch_size)
        # ars[1].shape = (seq_length, group_eval_batch_size)
        # ars[2].shape = (seq_length, group_eval_batch_size, user_stats_dim)

        # rnn_state
        h0, c0 = torch.zeros(group_eval_batch_size, self.rnn_input_dim).to(self.device), \
                torch.zeros(group_eval_batch_size, self.rnn_input_dim).to(self.device)
        rnn_state = (h0, c0)
        # if item already used, set it to 1e30
        sampled_action_mask = torch.zeros(group_eval_batch_size, self.item_num, dtype=torch.float32).to(self.device)
        for i in range(group_eval_batch_size):
            item_used = ars[0][0][i] # 1st 0: action, 2nd 0: first action, i: batch id
            sampled_action_mask[i][item_used] = 1e30

        step_count_list = [0 for i in range(group_eval_batch_size)]
        stop_count = [0 for i in range(group_eval_batch_size)]
        state_list = []

        step_count = 0
        while True:
            rnn_state, tmp_state = self.get_states_from_obs(
            actions=ars[0][step_count], 
            rewards=ars[1][step_count], 
            statistic=ars[2][step_count], 
            rnn_state=rnn_state,
            rnn_model=rnn_model,
            )
            state_list.append(tmp_state.cpu().numpy())


            # pass through feat_ext first
            tmp_state_feat = feat_ext(tmp_state)
            q_values = self.qnet(tmp_state_feat)
            argmax_action = torch.argmax(q_values - sampled_action_mask, dim=-1)

            ars[0].append([])
            ars[1].append([])
            ars[2].append([])
            step_count += 1
            for j in range(group_eval_batch_size):
                tmp_action = argmax_action[j]
                if tmp_action >= self.item_num:
                    raise ValueError("action index is out of range")
                sampled_action_mask[j][tmp_action] = 1e30
                obs, reward, done = self.env[j].step(tmp_action)
                history_items, history_rewards, history_stats = obs

                ars[0][-1].append(history_items[-1])
                ars[1][-1].append(history_rewards[-1])
                ars[2][-1].append(history_stats[-1])

                # set the current env as done 
                if done and stop_count[j] != 1:
                    step_count_list[j] = step_count
                    stop_count[j] = 1
            # test whether all envs are finished
            if np.sum(stop_count) == group_eval_batch_size:
                break
        
        for j in range(group_eval_batch_size):
            rewards = [ars[1][k][j] for k in range(0, self.episode_length)]
            reward_list.append(rewards)

        del sampled_action_mask
        del ars
        gc.collect()

        reward_arr = np.array(reward_list)
        return reward_arr

    def _get_init_ars(self, seq_num=-1):
        ''' get most popular item and do the sampling at the first step
        '''
        ars = [[[]], [[]], [[]]]
        if seq_num<=0:
            raise ValueError('_get_init_ars needs seq_num > 0')
        for i in range(seq_num):
            init_item_id = int(self.config['ENV']['POP1_ID'])
            obs, reward, _ = self.env[i].step(init_item_id)
            history_items, history_rewards, history_stats = obs

            # assert init_item_id == history_items[-1]
            # assert reward == history_rewards[-1]
            # assert np.sum( np.array(self.env[i].get_statistic()) - history_stats[-1] ) < 1e-6

            ars[0][0].append(history_items[-1])
            ars[1][0].append(history_rewards[-1])
            ars[2][0].append(history_stats[-1])

        return ars

    def get_random_action(self, used_action_list):
        sampled_action = random.sample(set(range(self.item_num)) - set(used_action_list), k=1)[0]
        return sampled_action

    def get_l2_norm(self, net):
        # add l2 regularization to net
        l2_norm = torch.tensor(0.0).to(self.device)
        for name, param in net.named_parameters():
            l2_norm += torch.norm(param)
        return l2_norm

    def save_stats(self):
        filename = self.log_fn + ".pkl"
        with open(filename, 'wb') as fb:
            pickle.dump(self.epoch_logger, fb)
