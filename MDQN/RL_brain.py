"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import numpy as np
import tensorflow as tf
import os
import pickle
from network import Q_Net
np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            n_agents,
            sess,
            num_training,
            learning_rate=0.01,
            reward_decay=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            save_model_freq=100,
            max_epsilon=1,
            min_epsilon=0,
            load_model=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.n_agents = n_agents
        self.sess = sess
        self.num_training = num_training
        self.lr = learning_rate
        self.gamma = reward_decay
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.save_model_freq = save_model_freq
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon = self.max_epsilon
        self.load_model = load_model

        # total learning step
        self.learn_step_counter = 0
        self.episode_rew_agent = 0
        self.episode_rew_all = 0
        self.episode = 0

        self.s, self.q_eval = self.create_Q_network("Q_network")
        self.s_, self.q_next = self.create_Q_network("target_Q_network")

        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.q_target = tf.placeholder(tf.float32, [None, self.n_agents, self.n_actions], name='Q_target')
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))

        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr, epsilon=1e-02).minimize(self.loss)

        self.cost_his = []
        if(self.load_model):
            saver = tf.train.Saver(max_to_keep=100000000)
            model_load_steps = 620000
            model_file_load = os.path.join("models/", str(model_load_steps) + "_" + "model_segment_training/", "8m")
            saver.restore(self.sess, model_file_load)
            print("model trained for %s steps have been loaded"%(model_load_steps))
        else:
            self.sess, self.saver, self.summary_placeholders, self.update_ops, self.summary_op, self.summary_writer, self.summary_vars = self.init_sess()

        # 将网络计算的初始化工作完成
    def init_sess(self):
        # Summary for tensorboard
        summary_placeholders, update_ops, summary_op, summary_vars = self.setup_summary()
        fileWritePath = os.path.join("logs/")
        summary_writer = tf.summary.FileWriter(fileWritePath, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        # Load the file if the saved file exists

        saver = tf.train.Saver(max_to_keep=100000000)

        return self.sess, saver, summary_placeholders, update_ops, summary_op, summary_writer, summary_vars

    def create_Q_network(self, name):
        inputs = tf.placeholder(tf.float32, shape=(None, self.n_agents, self.n_features), name="inputs")
        out = Q_Net.build_network(name, inputs, self.n_agents, 256, self.n_actions)
        return inputs, out

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        # observation = observation[np.newaxis, :]
        if(self.load_model == False):
            if np.random.uniform() < self.epsilon:
                # forward feed the observation and get q value for every actions
                action_list = []
                for i in range(self.n_agents):
                    action_list.append(np.random.randint(0, self.n_actions))
            else:
                actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
                action_list = np.argmax(actions_value, axis=2).tolist()
                action_list = action_list[0]
        else:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action_list = (np.argmax(actions_value, axis=2).tolist())
            action_list = action_list[0]

        return action_list

    def predict(self, inputs):
        return self.sess.run(self.q_eval, feed_dict={
            self.s: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.q_next, feed_dict={
            self.s_: inputs
        })
    def learn(self, s, q_target):

        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: s,
                                                self.q_target: q_target})

        self.cost_his.append(self.cost)

        self.learn_step_counter += 1

        self.plotting()

        # Decreasing epsilon
        if self.epsilon > self.min_epsilon:
            self.epsilon -= self.max_epsilon/self.num_training
        else:
            self.epsilon = self.min_epsilon


        if (self.learn_step_counter % self.save_model_freq == 0):
            model_file_save = os.path.join("models/", str(self.learn_step_counter) + "_" + "model_segment_training/", "8m")
            dirname = os.path.dirname(model_file_save)
            if any(dirname):
                os.makedirs(dirname, exist_ok=True)
            self.saver.save(self.sess, model_file_save)
            print("Model trained for %s times is saved"%self.learn_step_counter)

    def setup_summary(self):
        cost = tf.Variable(0.)
        eps_rew_agent = tf.Variable(0.)
        eps_rew_all = tf.Variable(0.)

        tf.summary.scalar("cost", cost)
        tf.summary.scalar("eps_rew_agent", eps_rew_agent)
        tf.summary.scalar("eps_rew_all", eps_rew_all)
        summary_vars = [cost, eps_rew_agent, eps_rew_all]

        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]

        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()

        return summary_placeholders, update_ops, summary_op, summary_vars

    def plotting(self):
        tensorboard_info = [self.cost, self.episode_rew_agent, self.episode_rew_all]
        vars_plot = []
        for i in range(len(tensorboard_info)):
            vars_plot.append(self.sess.run(self.update_ops[i], feed_dict={self.summary_placeholders[i]: float(tensorboard_info[i])}))

        summary_1 = tf.Summary(value=[tf.Summary.Value(tag="cost", simple_value=vars_plot[0])])
        summary_2 = tf.Summary(value=[tf.Summary.Value(tag="eps_rew_agent", simple_value=vars_plot[1])])
        summary_3 = tf.Summary(value=[tf.Summary.Value(tag="eps_rew_all", simple_value=vars_plot[2])])

        self.summary_writer.add_summary(summary_1, self.learn_step_counter)
        self.summary_writer.add_summary(summary_2, self.episode)
        self.summary_writer.add_summary(summary_3, self.episode)

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    def get_episode_reward(self, eps_r_agent, eps_r_all, episode):
        self.episode_rew_agent = eps_r_agent
        self.episode_rew_all = eps_r_all
        self.episode = episode
