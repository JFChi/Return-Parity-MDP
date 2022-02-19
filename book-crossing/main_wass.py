
import configparser
import argparse
import random
import numpy as np
import os

import torch

from dqn_wass import DQN

def main():
    parser = argparse.ArgumentParser("Complementary arguments for pre-train")
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--no_pretrain_rnn', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--config_file", type=str, help="config", default="config/dqn_config")
    parser.add_argument("--data_dir", help="(processed) data directory", type=str, default="data/processed/")
    parser.add_argument("--group_json_file", help="userID to user sensitive group", type=str, default="user_gender_group_map.json")
    parser.add_argument("--train_json_file", help="downsampled userID to user sensitive group", type=str, default="")
    parser.add_argument("--sensitive_group", help="sensitive group name", type=str, default="gender")
    parser.add_argument("--log_dir", help="logging directory", type=str, default="logs")
    parser.add_argument("--log_fn", help="abs path name used to save the log file.", type=str, default="ml_1m_dqn_v4")
    # wass args
    parser.add_argument('--wass_update_interval', type=int, default=1)
    parser.add_argument('--wass_batch_size', type=int, default=100)
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read_file(open(args.config_file))

    dqn = DQN(config, args)
    dqn.run()
    

    

if __name__ == '__main__':
    main()