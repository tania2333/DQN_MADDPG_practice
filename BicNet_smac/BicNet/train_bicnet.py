"""
Implementation of DDPG - Deep Deterministic Policy Gradient https://github.com/pemami4911/deep-rl
Modified by Coac for BiCNet implementation https://github.com/Coac/CommNet-BiCnet
"""

import numpy as np
import tensorflow as tf

from BicNet.bicnet import BiCNet as CommNet
# from smac.BicNet.comm_net import CommNet as CommNet
from BicNet.guessing_sum_env import *
from BicNet.replay_buffer import ReplayBuffer

# ===========================
#   Actor and Critic DNNs
# ===========================

class ActorNetwork(object):
    def __init__(self, sess, learning_rate, tau, batch_size, num_agents, vector_obs_len, output_len, hidden_vector_len):
        self.sess = sess
        # self.s_dim = state_dim
        # self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size
        self.num_agents = num_agents
        self.obs_len = vector_obs_len
        self.output_len = output_len
        self.hidden_vector_len = hidden_vector_len

        self.inputs, self.out = self.create_actor_network("actor_network")  #in: (,2,1) out(,2,1)/ (?,3,1)
        self.network_params = tf.trainable_variables()   # print('param',len(self.network_params))   8

        self.target_inputs, self.target_out = self.create_actor_network("target_actor_network")
        self.target_network_params = tf.trainable_variables()[
                                     len(self.network_params):]

        with tf.name_scope("actor_update_target_network_params"):
            self.update_target_network_params = \
                [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                      tf.multiply(self.target_network_params[i], 1. - self.tau))
                 for i in range(len(self.target_network_params))]

        self.action_gradient = tf.placeholder(tf.float32, (self.num_agents, None, self.num_agents, self.output_len), name="action_gradient")# print('action_grad',self.action_gradient) 2,n,2,1 /3,?,3,1

        with tf.name_scope("actor_gradients"):
            grads = []
            for i in range(self.num_agents):
                for j in range(self.num_agents):
                    grads.append(tf.gradients(self.out[:, j], self.network_params, -self.action_gradient[j][:, i]))# (y,x,w) y对x中每个元素求导，之后w加权
            # print("actor_grads",grads) #                = 4 [grad1,2,3,4]
            grads = np.array(grads)
            # print(grads.shape)    (4, 8)
            self.unnormalized_actor_gradients = [tf.reduce_sum(list(grads[:, i]), axis=0) for i in range(len(self.network_params))]   # len_net_param=8
            self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))  # unn(1,8)  len(actor_gradients)=8

        self.optimize = tf.train.AdamOptimizer(self.learning_rate)
        self.optimize = self.optimize.apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_actor_network(self, name):
        inputs = tf.placeholder(tf.float32, shape=(None, self.num_agents, self.obs_len), name="actor_inputs")
        # print('input',inputs)
        out = CommNet.actor_build_network(name, inputs, self.num_agents, self.hidden_vector_len, self.output_len)   #(, 2,1)
        return inputs, out

    def train(self, inputs, action_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: action_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """

    def __init__(self, sess, learning_rate, tau, num_actor_vars, num_agents, vector_obs_len, output_len, hidden_vector_len):
        self.sess = sess
        self.learning_rate = learning_rate
        self.tau = tau
        self.num_agents = num_agents
        self.obs_len = vector_obs_len
        self.output_len = output_len
        self.hidden_vector_len = hidden_vector_len

        self.inputs, self.action, self.out = self.create_critic_network("critic_network")
        self.network_params = tf.trainable_variables()[num_actor_vars:]

        self.target_inputs, self.target_action, self.target_out = self.create_critic_network("target_critic_network")
        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        with tf.name_scope("critic_update_target_network_params"):
            self.update_target_network_params = \
                [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau)
                                                      + tf.multiply(self.target_network_params[i], 1. - self.tau))
                 for i in range(len(self.target_network_params))]

        self.predicted_q_value = tf.placeholder(tf.float32, (None, self.num_agents, self.output_len), name="predicted_q_value")

        M = tf.to_float(tf.shape(self.out)[0])  # num of minibatches
        # Li = (Yi - Qi)^2
        # L = Sum(Li)

        grads = []
        for i in range(self.num_agents):
            grads.append(tf.gradients(self.out[:, i], self.network_params, -(self.predicted_q_value[:, i] - self.out[:, i])))
        grads = np.array(grads)
        # print(grads.shape)    #(2, 8)
        self.unnormalized_critic_gradients = [tf.reduce_sum(list(grads[:, i]), axis=0) for i in  #按列求和 axis=0
                                             range(len(self.network_params))]  # len_net_param=8
        self.critic_gradients = list(map(lambda x: tf.div(x, M),  #对于unn中的每一个单位x，都要计算一次div
                                        self.unnormalized_critic_gradients))  # unn(1,8)  len(critic_gradients)=8
        self.optimize = tf.train.AdamOptimizer(self.learning_rate)
        self.optimize = self.optimize.apply_gradients(zip(self.critic_gradients, self.network_params))

        # self.action_grads = tf.gradients(self.out, self.action, name="action_grads")
        self.action_grads = [tf.gradients(self.out[:, i], self.action) for i in range(self.num_agents)] # out.shape:batchs,2,1; action: n,2,1;
        # action_grads = [[grad1],[grad2]]  grad1.shape=n ,2, 1
        self.action_grads = tf.stack(tf.squeeze(self.action_grads, 1))   #tf.squeeze: 默认删除所有为1的维度   (2, ?, 2, 1)

    def create_critic_network(self, name):
        inputs = tf.placeholder(tf.float32, shape=(None, self.num_agents, self.obs_len), name="critic_inputs")
        action = tf.placeholder(tf.float32, shape=(None, self.num_agents, self.output_len), name="critic_action")

        out = CommNet.critic_build_network(name, inputs, action, self.num_agents, self.hidden_vector_len, self.output_len)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        # return self.sess.run([self.out, self.optimize, self.loss], feed_dict={
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)


