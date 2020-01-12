import tensorflow as tf
import numpy as np
class AC:
    @staticmethod
    def weight_variable(name, shape, c_names):
        return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(),
                               collections=c_names, dtype=tf.float32)
    @staticmethod
    def bias_variable(name, shape, c_names):
        return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(),
                               collections=c_names, dtype=tf.float32)
    @staticmethod
    def actor_build_network(name, observation, n_features, n_actions):
        c_names = ['actor_params', tf.GraphKeys.GLOBAL_VARIABLES]
        first_fc_actor = [n_features, 256]  #[n_features, 256]
        second_fc = [256, 256]              #[256, 128]
        third_fc_actor = [256, n_actions]   #[128, n_actions]

        with tf.variable_scope(name) as scope:
            x = tf.cast(observation, dtype=tf.float32)
            with tf.variable_scope('l1'):
                w_fc1_actor = AC.weight_variable('_w_fc1', first_fc_actor, c_names)
                b_fc1_actor = AC.bias_variable('_b_fc1', [first_fc_actor[1]], c_names)
                h_fc1_actor = tf.nn.relu(tf.matmul(x, w_fc1_actor) + b_fc1_actor)

            with tf.variable_scope('l2'):
                w_fc2_actor = AC.weight_variable('_w_fc2', second_fc, c_names)
                b_fc2_actor = AC.bias_variable('_b_fc2', [second_fc[1]], c_names)
                h_fc2_actor = tf.nn.relu(tf.matmul(h_fc1_actor, w_fc2_actor) + b_fc2_actor)

            with tf.variable_scope('l3'):
                w_fc3_actor = AC.weight_variable('_w_fc3', third_fc_actor, c_names)
                b_fc3_actor = AC.bias_variable('_b_fc3', [third_fc_actor[1]], c_names)
                output_actor = tf.nn.softmax(tf.matmul(h_fc2_actor, w_fc3_actor) + b_fc3_actor)   # ( , 14)
                output_actor = tf.cast(output_actor, dtype=tf.float16)
        return output_actor


    @staticmethod
    def critic_build_network(name, observation, n_features):
        c_names = ['critic_params', tf.GraphKeys.GLOBAL_VARIABLES]
        with tf.variable_scope(name) as scope:
            first_fc_critic = [n_features, 256]
            second_fc = [256, 256]
            third_fc_critic = [256, 1]
            x = tf.cast(observation, dtype=tf.float32)

            with tf.variable_scope('l1'):
                w_fc1_critic = AC.weight_variable('_w_fc1', first_fc_critic, c_names)
                b_fc1_critic = AC.bias_variable('_b_fc1', [first_fc_critic[1]], c_names)
                h_fc1_critic = tf.nn.relu(tf.matmul(x, w_fc1_critic) + b_fc1_critic)

            with tf.variable_scope('l2'):
                w_fc2_critic = AC.weight_variable('_w_fc2', second_fc, c_names)
                b_fc2_critic = AC.bias_variable('_b_fc2', [second_fc[1]], c_names)
                h_fc2_critic = tf.nn.relu(tf.matmul(h_fc1_critic, w_fc2_critic) + b_fc2_critic)

            with tf.variable_scope('l3'):
                w_fc3_critic = AC.weight_variable('_w_fc3', third_fc_critic, c_names)
                b_fc3_critic = AC.bias_variable('_b_fc3', [third_fc_critic[1]], c_names)
                output_critic = tf.matmul(h_fc2_critic, w_fc3_critic) + b_fc3_critic
                output_critic = tf.cast(output_critic, dtype=tf.float16)

        return output_critic