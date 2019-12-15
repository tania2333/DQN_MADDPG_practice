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
import tensorflow.contrib as tc
from replay_buffer import ReplayBuffer
np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class MADDPG:
    def __init__(
            self,
            n_actions,
            n_agents,
            nb_other_aciton,
            n_features,
            sess,
            agent_id,
            num_training,
            learning_rate_actor=1e-4,
            learning_rate_critic=1e-3,
            reward_decay=0.99,
            replace_target_iter=100,
            memory_size=500,
            batch_size=32,
            save_model_freq=100,
            load_model=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.n_agents = n_agents
        self.sess = sess
        self.agent_id = agent_id
        self.num_training = num_training
        self.lr_actor = learning_rate_actor
        self.lr_critic = learning_rate_critic
        self.gamma = reward_decay
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.save_model_freq = save_model_freq
        self.load_model = load_model

        # total learning step
        self.learn_step_counter = 0
        self.episode_rew_agent = 0
        self.episode_rew_all = 0
        self.episode = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = ReplayBuffer(self.memory_size)
        # consist of [target_net, evaluate_net]

        state_input = tf.placeholder(shape=[None, n_features], dtype=tf.float32)
        action_input = tf.placeholder(shape=[None, n_actions], dtype=tf.float32)
        state_input_next = tf.placeholder(shape=[None, n_features], dtype=tf.float32)
        action_input_next = tf.placeholder(shape=[None, n_actions], dtype=tf.float32)
        other_action_input = tf.placeholder(shape=[None, nb_other_aciton], dtype=tf.float32)
        other_action_input_next = tf.placeholder(shape=[None, nb_other_aciton], dtype=tf.float32)

        reward = tf.placeholder(shape=[None, 1], dtype=tf.float32)


        first_fc_actor = [n_features, 256]
        first_fc_critic = [n_features + n_actions, 256]
        second_fc = [256, 128]
        third_fc_actor = [128, n_actions]
        third_fc_critic = [128, 1]

        def weight_variable(name, shape):
            return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

        def bias_variable(name, shape):
            return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

        def actor_network(name):
            # Actor Network
            with tf.variable_scope(name):
                w_fc1_actor = weight_variable('_w_fc1', first_fc_actor)
                b_fc1_actor = bias_variable('_b_fc1', [first_fc_actor[1]])

                w_fc2_actor = weight_variable('_w_fc2', second_fc)
                b_fc2_actor = bias_variable('_b_fc2', [second_fc[1]])

                w_fc3_actor = weight_variable('_w_fc3', third_fc_actor)
                b_fc3_actor = bias_variable('_b_fc3', [third_fc_actor[1]])

            x = state_input
            h_fc1_actor = tf.nn.relu(tf.matmul(x, w_fc1_actor) + b_fc1_actor)
            h_fc2_actor = tf.nn.relu(tf.matmul(h_fc1_actor, w_fc2_actor) + b_fc2_actor)

            output_actor = tf.nn.tanh(tf.matmul(h_fc2_actor, w_fc3_actor) + b_fc3_actor)
            return output_actor

        def actor_target_network(name):
            # Actor Network
            with tf.variable_scope(name):
                w_fc1_actor = weight_variable('_w_fc1', first_fc_actor)
                b_fc1_actor = bias_variable('_b_fc1', [first_fc_actor[1]])

                w_fc2_actor = weight_variable('_w_fc2', second_fc)
                b_fc2_actor = bias_variable('_b_fc2', [second_fc[1]])

                w_fc3_actor = weight_variable('_w_fc3', third_fc_actor)
                b_fc3_actor = bias_variable('_b_fc3', [third_fc_actor[1]])

            x = state_input_next
            h_fc1_actor = tf.nn.relu(tf.matmul(x, w_fc1_actor) + b_fc1_actor)
            h_fc2_actor = tf.nn.relu(tf.matmul(h_fc1_actor, w_fc2_actor) + b_fc2_actor)

            output_actor = tf.nn.tanh(tf.matmul(h_fc2_actor, w_fc3_actor) + b_fc3_actor)
            return output_actor


        def critic_network(name, action_input):
            with tf.variable_scope(name):
                w_fc1_critic = weight_variable('_w_fc1', first_fc_critic)
                b_fc1_critic = bias_variable('_b_fc1', [first_fc_critic[1]])

                w_fc2_critic = weight_variable('_w_fc2', second_fc)
                b_fc2_critic = bias_variable('_b_fc2', [second_fc[1]])

                w_fc3_critic = weight_variable('_w_fc3', third_fc_critic)
                b_fc3_critic = bias_variable('_b_fc3', [third_fc_critic[1]])

            x = tf.concat([state_input, action_input], axis=-1)
            # Critic Network
            h_fc1_critic = tf.nn.relu(tf.matmul(x, w_fc1_critic) + b_fc1_critic)
            h_fc2_critic = tf.nn.relu(tf.matmul(h_fc1_critic, w_fc2_critic) + b_fc2_critic)

            output_critic = tf.matmul(h_fc2_critic, w_fc3_critic) + b_fc3_critic
            return output_critic

        def critic_target_network(name, action_input_next):
            with tf.variable_scope(name):
                w_fc1_critic = weight_variable('_w_fc1', first_fc_critic)
                b_fc1_critic = bias_variable('_b_fc1', [first_fc_critic[1]])

                w_fc2_critic = weight_variable('_w_fc2', second_fc)
                b_fc2_critic = bias_variable('_b_fc2', [second_fc[1]])

                w_fc3_critic = weight_variable('_w_fc3', third_fc_critic)
                b_fc3_critic = bias_variable('_b_fc3', [third_fc_critic[1]])

            x = tf.concat([state_input_next, action_input_next], axis=-1)
            # Critic Network
            h_fc1_critic = tf.nn.relu(tf.matmul(x, w_fc1_critic) + b_fc1_critic)
            h_fc2_critic = tf.nn.relu(tf.matmul(h_fc1_critic, w_fc2_critic) + b_fc2_critic)

            output_critic = tf.matmul(h_fc2_critic, w_fc3_critic) + b_fc3_critic
            return output_critic


        self.action_output = actor_network("agent_actor")
        self.critic_output = critic_network('agent_critic',
                                            action_input=tf.concat([action_input, other_action_input], axis=1))  # 把所有动作都输入到critic网络中来
        self.action_target_output = actor_target_network("agent_target_actor")
        self.critic_target_output = critic_target_network("agent_target_critic",
                                            action_input_next=tf.concat([action_input_next, other_action_input_next], axis=1))  # 把所有动作都输入到critic网络中来

        self.state_input = state_input
        self.action_input = action_input

        self.state_input_next = state_input_next
        self.action_input_next = action_input_next
        self.other_action_input = other_action_input
        self.other_action_input_next = other_action_input_next
        self.reward = reward

        self.actor_optimizer = tf.train.AdamOptimizer(self.lr_actor)
        self.critic_optimizer = tf.train.AdamOptimizer(self.lr_critic)

        # 最大化Q值
        self.actor_loss = -tf.reduce_mean(  # error2: actor_loss没有考虑动作对策略网络参数的导数,但可以理解为Q值的最大化
            critic_network('agent_critic', action_input=tf.concat([self.action_output, other_action_input], axis=1)))
        self.actor_train = self.actor_optimizer.minimize(self.actor_loss)

        self.target_Q = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.critic_loss = tf.reduce_mean(tf.square(self.target_Q - self.critic_output))
        self.critic_train = self.critic_optimizer.minimize(self.critic_loss)

        if (self.load_model):
            saver = tf.train.Saver(max_to_keep=100000000)
            model_load_steps = 250000
            model_file_load = os.path.join("models/", "agent_No_" + str(self.agent_id) + "/",
                                           str(model_load_steps) + "_" + "model_segment_training/", "8m")
            saver.restore(self.sess, model_file_load)
            print("model trained for %s steps of agent %s have been loaded" % (model_load_steps, self.agent_id))
        else:
            self.sess, self.saver, self.summary_placeholders, self.update_ops, self.summary_op, self.summary_writer, self.summary_vars, self.actor_target_init, self.actor_target_update, \
            self.critic_target_init, self.critic_target_update = self.init_sess()

    def train_actor(self, state, other_action):
        _, self.actor_cost = self.sess.run(self.actor_train,self.actor_loss, {self.state_input: state, self.other_action_input: other_action})

    def train_critic(self, state, action, other_action, target):
        _, self.critic_cost = self.sess.run(self.critic_train,self.critic_loss,
                 {self.state_input: state, self.action_input: action, self.other_action_input: other_action,
                  self.target_Q: target})

    def action(self, state):
        return self.sess.run(self.action_output, {self.state_input: state})

    def action_target(self, state):
        return self.sess.run(self.action_target_output, {self.state_input_next: state})

    def Q(self, state, action, other_action):
        return self.sess.run(self.critic_output,
                        {self.state_input: state, self.action_input: action, self.other_action_input: other_action})

    def Q_target(self, state, action, other_action):
        return self.sess.run(self.critic_target_output,
                        {self.state_input_next: state, self.action_input_next: action, self.other_action_input_next: other_action})
        # 将网络计算的初始化工作完成
    def init_sess(self):
        # Summary for tensorboard
        summary_placeholders, update_ops, summary_op, summary_vars = self.setup_summary()
        fileWritePath = os.path.join("logs/", "agent_No_" + str(self.agent_id) + "/")
        summary_writer = tf.summary.FileWriter(fileWritePath, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        actor_target_init, actor_target_update = self.create_init_update('agent_actor', 'agent_target_actor')
        critic_target_init, critic_target_update = self.create_init_update('agent_critic','agent_target_critic')
        self.sess.run([actor_target_init, critic_target_init])

        # Load the file if the saved file exists

        saver = tf.train.Saver(max_to_keep=100000000)

        return self.sess, saver, summary_placeholders, update_ops, summary_op, summary_writer, summary_vars, actor_target_init, actor_target_update, \
               critic_target_init, critic_target_update

    def store_transition(self, s_set, a_set, r, s_next_set, done):
        s_list = []
        a_list = []
        s_next_list = []
        s_list.append(s_set[self.agent_id])
        a_list.append(a_set[self.agent_id])
        s_next_list.append(s_next_set[self.agent_id])

        if(self.agent_id != self.n_agents - 1):
            for i in range(self.agent_id+1, self.n_agents):
                s_list.append(s_set[i])
                a_list.append(a_set[i])
                s_next_list.append(s_next_set[i])
        if (self.agent_id != 0):
            for i in range(self.agent_id):
                s_list.append(s_set[i])
                a_list.append(a_set[i])
                s_next_list.append(s_next_set[i])

        self.memory.add(np.vstack(s_list), np.vstack(a_list), r, np.vstack(s_next_list), done)



    def train_agent(self, RL_set):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run([self.actor_target_update, self.critic_target_update])

        total_obs_batch, total_act_batch, rew_batch, total_next_obs_batch, done_mask = self.memory.sample(32)  # batch_size默认为32

        act_batch = total_act_batch[:, 0, :]  # 0 代表当前智能体的动作

        other_act_batch = []
        for i in range(1, self.n_agents):
            other_act_batch.append(total_act_batch[:, i, :])


        other_act_batch = np.hstack(other_act_batch)  # 其他智能体的动作，可以将其放到critic网络进行训练

        obs_batch = total_obs_batch[:, 0, :]  # 当前智能体局部观察
        next_obs_batch = total_next_obs_batch[:, 0, :]

        other_actors = []
        if (self.agent_id != self.n_agents - 1):
            for i in range(self.agent_id + 1, self.n_agents):
                other_actors.append(RL_set[i])
        if (self.agent_id != 0):
            for i in range(self.agent_id):
                other_actors.append(RL_set[i])

        next_other_action = []
        for i in range(1, self.n_agents):
            next_other_action.append(other_actors[i-1].action_target(total_next_obs_batch[:, i, :]))
        next_other_action = np.hstack(next_other_action)
        # 获取下一个情况下另外两个agent的行动

        target = rew_batch.reshape(-1, 1) + self.gamma * self.Q_target(state=next_obs_batch,
                                                                         action=self.action(next_obs_batch),
                                                                         other_action=next_other_action)
        self.train_actor(state=obs_batch, other_action=other_act_batch)
        self.train_critic(state=obs_batch, action=act_batch, other_action=other_act_batch, target=target)

        self.learn_step_counter += 1

        self.plotting()

    def create_init_update(self, oneline_name, target_name, tau=0.99):
        online_var = [i for i in tf.trainable_variables() if oneline_name in i.name]
        target_var = [i for i in tf.trainable_variables() if target_name in i.name]

        target_init = [tf.assign(target, online) for online, target in zip(online_var, target_var)]
        target_update = [tf.assign(target, (1 - tau) * online + tau * target) for online, target in zip(online_var, target_var)]

        return target_init, target_update

    def setup_summary(self):
        actor_cost = tf.Variable(0.)
        critic_cost = tf.Variable(0.)
        eps_rew_agent = tf.Variable(0.)
        eps_rew_all = tf.Variable(0.)

        tf.summary.scalar("actor_cost", actor_cost)
        tf.summary.scalar("critic_cost", critic_cost)
        tf.summary.scalar("eps_rew_agent", eps_rew_agent)
        tf.summary.scalar("eps_rew_all", eps_rew_all)
        summary_vars = [actor_cost, critic_cost, eps_rew_agent, eps_rew_all]

        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]

        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()

        return summary_placeholders, update_ops, summary_op, summary_vars

    def plotting(self):
        tensorboard_info = [self.actor_cost, self.critic_cost, self.episode_rew_agent, self.episode_rew_all]
        vars_plot = []
        for i in range(len(tensorboard_info)):
            vars_plot.append(self.sess.run(self.update_ops[i], feed_dict={self.summary_placeholders[i]: float(tensorboard_info[i])}))

        summary_1 = tf.Summary(value=[tf.Summary.Value(tag="actor_cost", simple_value=vars_plot[0])])
        summary_2 = tf.Summary(value=[tf.Summary.Value(tag="critic_cost", simple_value=vars_plot[1])])
        summary_3 = tf.Summary(value=[tf.Summary.Value(tag="eps_rew_agent", simple_value=vars_plot[2])])
        summary_4 = tf.Summary(value=[tf.Summary.Value(tag="eps_rew_all", simple_value=vars_plot[3])])

        self.summary_writer.add_summary(summary_1, self.learn_step_counter)
        self.summary_writer.add_summary(summary_2, self.learn_step_counter)
        self.summary_writer.add_summary(summary_3, self.episode)
        self.summary_writer.add_summary(summary_4, self.episode)

    def get_episode_reward(self, eps_r_agent, eps_r_all, episode):
        self.episode_rew_agent = eps_r_agent
        self.episode_rew_all = eps_r_all
        self.episode = episode