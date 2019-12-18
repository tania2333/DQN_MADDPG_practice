import tensorflow as tf

class MADDPG:
    @staticmethod
    def weight_variable(name, shape, c_names):
        return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(),
                               collections=c_names)
    @staticmethod
    def bias_variable(name, shape, c_names):
        return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(),
                               collections=c_names)
    @staticmethod
    def actor_build_network(name, observation, n_features, n_actions):
        c_names = ['actor_params', tf.GraphKeys.GLOBAL_VARIABLES]
        first_fc_actor = [n_features, 256]
        second_fc = [256, 128]
        third_fc_actor = [128, n_actions]

        with tf.variable_scope(name):
            x = observation
            with tf.variable_scope('l1'):
                w_fc1_actor = MADDPG.weight_variable('_w_fc1', first_fc_actor, c_names)
                b_fc1_actor = MADDPG.bias_variable('_b_fc1', [first_fc_actor[1]], c_names)
                h_fc1_actor = tf.nn.relu(tf.matmul(x, w_fc1_actor) + b_fc1_actor)

            with tf.variable_scope('l2'):
                w_fc2_actor = MADDPG.weight_variable('_w_fc2', second_fc, c_names)
                b_fc2_actor = MADDPG.bias_variable('_b_fc2', [second_fc[1]], c_names)
                h_fc2_actor = tf.nn.relu(tf.matmul(h_fc1_actor, w_fc2_actor) + b_fc2_actor)

            with tf.variable_scope('l3'):
                w_fc3_actor = MADDPG.weight_variable('_w_fc3', third_fc_actor, c_names)
                b_fc3_actor = MADDPG.bias_variable('_b_fc3', [third_fc_actor[1]], c_names)
                output_actor = tf.nn.tanh(tf.matmul(h_fc2_actor, w_fc3_actor) + b_fc3_actor)

        return output_actor

    @staticmethod
    def critic_build_network(name, observation, n_features, n_agents, n_actions, own_action, other_action):
        c_names = ['critic_params', tf.GraphKeys.GLOBAL_VARIABLES]
        with tf.variable_scope(name) as scope:
            first_fc_critic = [n_features + n_actions * n_agents, 256]
            second_fc = [256, 128]
            third_fc_critic = [128, 1]
            action = tf.concat([own_action, other_action], axis=1)
            x = tf.concat([observation, action], axis=-1)

            with tf.variable_scope('l1'):
                w_fc1_critic = MADDPG.weight_variable('_w_fc1', first_fc_critic, c_names)
                b_fc1_critic = MADDPG.bias_variable('_b_fc1', [first_fc_critic[1]], c_names)
                h_fc1_critic = tf.nn.relu(tf.matmul(x, w_fc1_critic) + b_fc1_critic)

            with tf.variable_scope('l2'):
                w_fc2_critic = MADDPG.weight_variable('_w_fc2', second_fc, c_names)
                b_fc2_critic = MADDPG.bias_variable('_b_fc2', [second_fc[1]], c_names)
                h_fc2_critic = tf.nn.relu(tf.matmul(h_fc1_critic, w_fc2_critic) + b_fc2_critic)

            with tf.variable_scope('l3'):
                w_fc3_critic = MADDPG.weight_variable('_w_fc3', third_fc_critic, c_names)
                b_fc3_critic = MADDPG.bias_variable('_b_fc3', [third_fc_critic[1]], c_names)
                output_critic = tf.matmul(h_fc2_critic, w_fc3_critic) + b_fc3_critic

        return output_critic