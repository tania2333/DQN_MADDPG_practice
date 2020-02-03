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
np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            sess,
            agent_id,
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
        self.sess = sess
        self.agent_id = agent_id
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

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.cost_his = []
        if(self.load_model):
            saver = tf.train.Saver(max_to_keep=100000000)
            model_load_steps = 420000
            model_file_load = os.path.join("models/", "agent_No_" + str(self.agent_id) + "/",
                                           str(model_load_steps) + "_" + "model_segment_training/", "8m")
            saver.restore(self.sess, model_file_load)
            print("model trained for %s steps of agent %s have been loaded"%(model_load_steps, self.agent_id))
        else:
            self.sess, self.saver, self.summary_placeholders, self.update_ops, self.summary_op, self.summary_writer, self.summary_vars = self.init_sess()

        # 将网络计算的初始化工作完成
    def init_sess(self):
        # Summary for tensorboard
        summary_placeholders, update_ops, summary_op, summary_vars = self.setup_summary()
        fileWritePath = os.path.join("logs/", "agent_No_" + str(self.agent_id) + "/")
        summary_writer = tf.summary.FileWriter(fileWritePath, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        # Load the file if the saved file exists

        saver = tf.train.Saver(max_to_keep=100000000)

        return self.sess, saver, summary_placeholders, update_ops, summary_op, summary_writer, summary_vars

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables                                512*512的网络结构
            c_names, n_l1, n_l2, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 256, 256, \
                tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer()
                # tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, n_l2], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, n_l2], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

            # third layer. collections is used later when assign to target net
            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [n_l2, self.n_actions], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l2, w3) + b3

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))

        with tf.variable_scope('train'):
            # self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
            self._train_op = tf.train.AdamOptimizer(self.lr, epsilon=1e-02).minimize(self.loss)
        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, n_l2], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, n_l2], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2)) + b2

            # third layer. collections is used later when assign to target net
            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [n_l2, self.n_actions], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l2, w3) + b3

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))  #往水平方向平铺，所以是一行数

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]
        if(self.load_model == False):
            if np.random.uniform() < self.epsilon:
                # forward feed the observation and get q value for every actions
                action = np.random.randint(0, self.n_actions)
            else:
                actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
                action = np.argmax(actions_value)
        else:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        return action

    def learn(self):
        # check to replace target parameters
        if(self.memory_counter < self.batch_size):
            return
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            # print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]     #batch个行，第n_features + 1列的数，那正好是reward

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]

        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]

        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
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
            model_file_save = os.path.join("models/", "agent_No_"+str(self.agent_id)+"/", str(self.learn_step_counter) + "_" + "model_segment_training/", "8m")
            dirname = os.path.dirname(model_file_save)
            if any(dirname):
                os.makedirs(dirname, exist_ok=True)
            self.saver.save(self.sess, model_file_save)
            print("Model trained for %s times is saved"%self.learn_step_counter)

            # save data of replay buffer
            obj = self.memory
            filename = 'buffer_agent'+str(self.agent_id)+'.txt'
            file = open(filename, 'wb')
            pickle.dump(obj, file)
            file.close()

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
