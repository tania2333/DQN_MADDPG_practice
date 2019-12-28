import tensorflow as tf
import numpy as np
class DDPG:
    @staticmethod
    def weight_variable(name, shape, c_names):
        return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(),
                               collections=c_names, dtype=tf.float32)
    @staticmethod
    def bias_variable(name, shape, c_names):
        return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(),
                               collections=c_names, dtype=tf.float32)
    @staticmethod
    def actor_build_network(name, observation, n_features, n_actions, training_step, decay_period, test_flag):
        c_names = ['actor_params', tf.GraphKeys.GLOBAL_VARIABLES]
        first_fc_actor = [n_features, 256]  #[n_features, 256]
        second_fc = [256, 256]              #[256, 128]
        third_fc_actor = [256, n_actions]   #[128, n_actions]

        with tf.variable_scope(name) as scope:
            x = tf.cast(observation, dtype=tf.float32)
            with tf.variable_scope('l1'):
                w_fc1_actor = DDPG.weight_variable('_w_fc1', first_fc_actor, c_names)
                b_fc1_actor = DDPG.bias_variable('_b_fc1', [first_fc_actor[1]], c_names)
                h_fc1_actor = tf.nn.relu(tf.matmul(x, w_fc1_actor) + b_fc1_actor)

            with tf.variable_scope('l2'):
                w_fc2_actor = DDPG.weight_variable('_w_fc2', second_fc, c_names)
                b_fc2_actor = DDPG.bias_variable('_b_fc2', [second_fc[1]], c_names)
                h_fc2_actor = tf.nn.relu(tf.matmul(h_fc1_actor, w_fc2_actor) + b_fc2_actor)

            with tf.variable_scope('l3'):
                w_fc3_actor = DDPG.weight_variable('_w_fc3', third_fc_actor, c_names)
                b_fc3_actor = DDPG.bias_variable('_b_fc3', [third_fc_actor[1]], c_names)
                logits = tf.matmul(h_fc2_actor, w_fc3_actor) + b_fc3_actor
                if(test_flag is False):
                    logits = DDPG.add_noise(logits, training_step, decay_period)
                output_actor = tf.nn.softmax(logits)   # ( , 14)
                output_actor = tf.cast(output_actor, dtype=tf.float16)

        return output_actor
    @staticmethod
    def add_noise(logits, training_step, decay_period, anneal_ratio_max=1, anneal_ratio_min=0.001):
        noise = tf.random_uniform(tf.shape(logits))
        logits_with_noise = logits - tf.log(-tf.log(noise))
        anneal_ratio = anneal_ratio_max - (anneal_ratio_max - anneal_ratio_min) * min(1.0, training_step / decay_period)
        return logits_with_noise / anneal_ratio

    @staticmethod
    def critic_build_network(name, observation, n_features, n_actions, action):
        c_names = ['critic_params', tf.GraphKeys.GLOBAL_VARIABLES]
        with tf.variable_scope(name) as scope:
            first_fc_critic = [n_features + n_actions * 1, 256]
            second_fc = [256, 128]
            third_fc_critic = [128, 1]
            x = tf.concat([observation, action], axis=-1)
            x = tf.cast(x, dtype=tf.float32)

            with tf.variable_scope('l1'):
                w_fc1_critic = DDPG.weight_variable('_w_fc1', first_fc_critic, c_names)
                b_fc1_critic = DDPG.bias_variable('_b_fc1', [first_fc_critic[1]], c_names)
                h_fc1_critic = tf.nn.relu(tf.matmul(x, w_fc1_critic) + b_fc1_critic)

            with tf.variable_scope('l2'):
                w_fc2_critic = DDPG.weight_variable('_w_fc2', second_fc, c_names)
                b_fc2_critic = DDPG.bias_variable('_b_fc2', [second_fc[1]], c_names)
                h_fc2_critic = tf.nn.relu(tf.matmul(h_fc1_critic, w_fc2_critic) + b_fc2_critic)

            with tf.variable_scope('l3'):
                w_fc3_critic = DDPG.weight_variable('_w_fc3', third_fc_critic, c_names)
                b_fc3_critic = DDPG.bias_variable('_b_fc3', [third_fc_critic[1]], c_names)
                output_critic = tf.matmul(h_fc2_critic, w_fc3_critic) + b_fc3_critic
                output_critic = tf.cast(output_critic, dtype=tf.float16)

        return output_critic