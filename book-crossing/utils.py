#coding: utf-8

from scipy.sparse import coo_matrix
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)
tf.disable_v2_behavior()
import configparser
import numpy as np
import pickle
import time
import os
import logging
import sys

def pickle_save(object, file_path):
    f = open(file_path, 'wb')
    pickle.dump(object, f)

def pickle_load(file_path):
    f = open(file_path, 'rb')
    return pickle.load(f)

class EpochLogger:
	"""
	A variant of Logger tailored for tracking average values over epochs/steps.
	Code modified from https://spinningup.openai.com/en/latest/_modules/spinup/utils/logx.html#EpochLogger
	Typical use case: there is some quantity which is calculated many times
	throughout an epoch, and at the end of the epoch, you would like to 
	report the average / std / min / max value of that quantity.
	"""

	def __init__(self):
		self.epoch_dict = dict()

	def store(self, **kwargs):
		"""
		Save something into the epoch_logger's current state.
		Provide an arbitrary number of keyword arguments with numerical 
		values.
		"""
		for k,v in kwargs.items():
			if not(k in self.epoch_dict.keys()):
				self.epoch_dict[k] = []
			self.epoch_dict[k].append(v)

def get_logger(filename):
	# Logging configuration: set the basic configuration of the logging system
	log_formatter = logging.Formatter(fmt='%(asctime)s [%(processName)s, %(process)s] [%(levelname)-5.5s]  %(message)s',
									  datefmt='%m-%d %H:%M')
	logger = logging.getLogger()
	logger.setLevel(logging.DEBUG)
	# File logger
	file_handler = logging.FileHandler("{}.log".format(filename))
	file_handler.setFormatter(log_formatter)
	file_handler.setLevel(logging.DEBUG)
	logger.addHandler(file_handler)
	# Stderr logger
	std_handler = logging.StreamHandler(sys.stdout)
	std_handler.setFormatter(log_formatter)
	std_handler.setLevel(logging.DEBUG)
	logger.addHandler(std_handler)
	return logger

def mf_with_bias(rating_file, env_object_path, config, data_path, lr=1e-2, l2_factor=1e-2, max_step=1000, train_rate=0.95, max_stop_count=5):
    # config = configparser.ConfigParser()
    # config.read_file(open('../config'))
    # env_object_path = '../data/run_time/%s_env_objects' % rating_file
    print("Training item embedding ...")

    if not os.path.exists(env_object_path):
        exit('error')
    objects = pickle_load(env_object_path)
    user_num = objects['user_num']
    item_num = objects['item_num']

    print('user number: %d' % user_num)
    print('item number: %d' % item_num)
    boundary_user_id = int(user_num*0.8)
    emb_size = int(config['META']['ACTION_DIM'])

    rating = np.loadtxt(dtype=float, fname=os.path.join(data_path, rating_file), delimiter='\t')
    data = np.array(list(filter(lambda x: x[0] < boundary_user_id, rating)))
    np.random.shuffle(data)

    t = int(len(data)*train_rate)
    dtrain = data[:t]
    dtest = data[t:]

    user_embeddings = tf.Variable(tf.truncated_normal([user_num, emb_size], mean=0, stddev=0.01))
    item_embeddings = tf.Variable(tf.truncated_normal([item_num, emb_size], mean=0, stddev=0.01))
    item_bias = tf.Variable(tf.zeros([item_num, 1], tf.float32))

    user_ids = tf.placeholder(tf.int32, shape=[None])
    item_ids = tf.placeholder(tf.int32, shape=[None])
    ys = tf.placeholder(tf.float32, shape=[None])

    user_embs = tf.nn.embedding_lookup(user_embeddings, user_ids)
    item_embs = tf.nn.embedding_lookup(item_embeddings, item_ids)
    ibias_embs = tf.nn.embedding_lookup(item_bias, item_ids)
    dot_e = user_embs * item_embs

    ys_pre = tf.reduce_sum(dot_e, 1)+tf.reduce_sum(ibias_embs,1)

    target_loss = tf.reduce_mean(0.5 * tf.square(ys - ys_pre))
    loss = target_loss + l2_factor * (tf.reduce_mean(tf.square(user_embs) + tf.square(item_embs)))

    train_step = tf.train.AdamOptimizer(lr).minimize(loss)

    rmse = tf.sqrt(tf.reduce_mean(tf.square(ys - ys_pre)))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        np.random.shuffle(dtrain)
        print('training svd...')
        rmse_train, loss_v, target_loss_v = sess.run([rmse, loss, target_loss], feed_dict={user_ids: dtrain[:, 0], item_ids: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
        rmse_test = sess.run(rmse, feed_dict={user_ids: dtest[:, 0], item_ids: dtest[:, 1], ys: np.float32(dtest[:, 2])})
        print('----round%2d: rmse_train: %f, rmse_test: %f, loss: %f, target_loss: %f' % (0, rmse_train, rmse_test, loss_v, target_loss_v))
        pre_rmse_test = 100.0
        stop_count = 0
        for i in range(max_step):
            feed_dict = {user_ids: dtrain[:, 0],
                         item_ids: dtrain[:, 1],
                         ys: np.float32(dtrain[:, 2])}
            sess.run(train_step, feed_dict)
            np.random.shuffle(dtrain)
            rmse_train, loss_v, target_loss_v = sess.run([rmse, loss, target_loss], feed_dict={user_ids: dtrain[:, 0], item_ids: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
            rmse_test = sess.run(rmse, feed_dict={user_ids: dtest[:, 0], item_ids: dtest[:, 1], ys: np.float32(dtest[:, 2])})
            print('----round%2d: rmse_train: %f, rmse_test: %f, loss: %f, target_loss: %f' % (i + 1, rmse_train, rmse_test, loss_v, target_loss_v))
            if rmse_test>pre_rmse_test:
                stop_count += 1
                if stop_count==max_stop_count:
                    item_embeddings_value = sess.run(item_embeddings)
                    item_bias_value = sess.run(item_bias)
                    # np.savetxt('../data/run_time/' + rating_file + '_item_embedding_dim%d' % emb_size, delimiter='\t', X=item_embeddings_value)
                    # np.savetxt('../data/run_time/' + rating_file + '_item_bias_dim%d' % emb_size, delimiter='\t', X=item_bias_value)
                    np.savetxt(os.path.join(data_path, "item_embedding_dim_%d"%emb_size), delimiter='\t', X=item_embeddings_value)
                    np.savetxt(os.path.join(data_path, "item_bias_dim_%d"%emb_size), delimiter='\t', X=item_bias_value)
                    print('done with full stop count for training item embedding')
                    return
            pre_rmse_test = rmse_test

        item_embeddings_value = sess.run(item_embeddings)
        item_bias_value = sess.run(item_bias)

        # np.savetxt('../data/run_time/'+rating_file+'_item_embedding_dim%d'%emb_size, delimiter='\t', X=item_embeddings_value)
        # np.savetxt('../data/run_time/'+rating_file+'_item_bias_dim%d'%emb_size, delimiter='\t', X=item_bias_value)
        np.savetxt(os.path.join(data_path, "item_embedding_dim_%d"%emb_size), delimiter='\t', X=item_embeddings_value)
        np.savetxt(os.path.join(data_path, "item_bias_dim_%d"%emb_size), delimiter='\t', X=item_bias_value)

        print('done with full training step for training item embedding')

def get_envobjects(ratingfile, data_path, boundary_rating=None, emb_dim=8, lr=1e-2, l2_factor=1e-2, max_step=2000, train_rate=0.95, max_stop_count=5):
    # rating_file_path = '../data/rating/' + ratingfile
    rating_file_path = os.path.join(data_path, ratingfile)
    rating = np.loadtxt(fname=rating_file_path, delimiter='\t')

    user_set = set()
    item_set = set()
    for i, j, k in rating:
        user_set.add(int(i))
        item_set.add(int(j))

    user_num = len(user_set)
    item_num = len(item_set)
    emb_size = emb_dim

    data = np.array(rating)
    np.random.shuffle(data)

    t = int(len(data) * train_rate)
    dtrain = data[:t]
    dtest = data[t:]

    user_embeddings = tf.Variable(tf.truncated_normal([user_num, emb_size], mean=0, stddev=0.01))
    item_embeddings = tf.Variable(tf.truncated_normal([item_num, emb_size], mean=0, stddev=0.01))
    item_bias = tf.Variable(tf.zeros([item_num, 1], tf.float32))

    user_ids = tf.placeholder(tf.int32, shape=[None])
    item_ids = tf.placeholder(tf.int32, shape=[None])
    ys = tf.placeholder(tf.float32, shape=[None])

    user_embs = tf.nn.embedding_lookup(user_embeddings, user_ids)
    item_embs = tf.nn.embedding_lookup(item_embeddings, item_ids)
    ibias_embs = tf.nn.embedding_lookup(item_bias, item_ids)
    dot_e = user_embs * item_embs

    ys_pre = tf.reduce_sum(dot_e, 1) + tf.squeeze(ibias_embs)

    target_loss = tf.reduce_mean(0.5 * tf.square(ys - ys_pre))
    loss = target_loss + l2_factor * (tf.reduce_mean(tf.square(user_embs) + tf.square(item_embs)))

    train_step = tf.train.AdamOptimizer(lr).minimize(loss)

    rmse = tf.sqrt(tf.reduce_mean(tf.square(ys - ys_pre)))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        np.random.shuffle(dtrain)
        rmse_train, loss_v, target_loss_v = sess.run([rmse, loss, target_loss],
                                                     feed_dict={user_ids: dtrain[:, 0], item_ids: dtrain[:, 1],
                                                                ys: np.float32(dtrain[:, 2])})
        rmse_test = sess.run(rmse,
                             feed_dict={user_ids: dtest[:, 0], item_ids: dtest[:, 1], ys: np.float32(dtest[:, 2])})
        print('----round%2d: rmse_train: %f, rmse_test: %f, loss: %f, target_loss: %f' % (
        0, rmse_train, rmse_test, loss_v, target_loss_v))
        pre_rmse_test = 100.0
        stop_count = 0
        stop_count_flag = False
        for i in range(max_step):
            feed_dict = {user_ids: dtrain[:, 0],
                         item_ids: dtrain[:, 1],
                         ys: np.float32(dtrain[:, 2])}
            sess.run(train_step, feed_dict)
            np.random.shuffle(dtrain)
            rmse_train, loss_v, target_loss_v = sess.run([rmse, loss, target_loss],
                                                         feed_dict={user_ids: dtrain[:, 0], item_ids: dtrain[:, 1],
                                                                    ys: np.float32(dtrain[:, 2])})
            rmse_test = sess.run(rmse, feed_dict={user_ids: dtest[:, 0], item_ids: dtest[:, 1],
                                                  ys: np.float32(dtest[:, 2])})
            print('----round%2d: rmse_train: %f, rmse_test: %f, loss: %f, target_loss: %f' % (
            i + 1, rmse_train, rmse_test, loss_v, target_loss_v))
            if rmse_test > pre_rmse_test:
                stop_count += 1
                if stop_count == max_stop_count:
                    stop_count_flag = True
                    break
            pre_rmse_test = rmse_test

        user_embeddings_value, item_embeddings_value = sess.run([user_embeddings, item_embeddings])
        mf_rating = np.dot(user_embeddings_value, item_embeddings_value.T)
        rela_num = np.sum(np.where(mf_rating>boundary_rating,1,0),axis=1)
        print('Done with full stop count' if stop_count_flag else 'Done with full training step')
        env_objects_path = os.path.join(data_path, 'env_objects')
        pickle_save({'r_matrix': mf_rating, 'user_num': user_num, 'item_num': item_num,'rela_num':rela_num}, env_objects_path)
        print(f"save env object to {env_objects_path}")