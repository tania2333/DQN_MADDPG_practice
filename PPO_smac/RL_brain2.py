import tensorflow as tf
import numpy as np
from actor_critic import AC
from replay_buffer import ReplayBuffer
import os

class ActorNetwork(object):
    def __init__(self, sess, learning_rate, n_features, n_actions, agent_id):
        self.sess = sess
        self.learning_rate = learning_rate
        self.n_features = n_features
        self.n_actions = n_actions
        self.agent_id = agent_id
        self.training_step = 0

        self.inputs, self.out = self.create_actor_network("actor_network")  #in: (,2,1) out(,2,1)/ (?,3,1)

        self.action_actor = tf.placeholder(tf.float16, shape=[None, self.n_actions])
        self.advantage_actor = tf.placeholder(tf.float16, shape=[None])

        self.action_prob = tf.reduce_sum(tf.multiply(self.action_actor, self.out))

        with tf.variable_scope('loss'):
            self.cross_entropy = tf.multiply(tf.log(self.action_prob + 1e-10), self.advantage_actor)
            self.Loss_actor = - tf.reduce_sum(self.cross_entropy)

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(
                self.Loss_actor)  # minimize(-exp_v) = maximize(exp_v)

        self.saver = tf.train.Saver(max_to_keep=100000000)

    def create_actor_network(self, name):
        inputs = tf.placeholder(tf.float16, shape=(None, self.n_features), name="actor_inputs")
        out = AC.actor_build_network(name, inputs, self.n_features, self.n_actions)  # (, 2,1)
        return inputs, out

    def train(self, inputs, action_actor, advantage_actor):
        self.training_step += 1
        self.sess.run(self.train_op, feed_dict={
            self.inputs: inputs,
            self.action_actor: action_actor,
            self.advantage_actor: advantage_actor
        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

    def save_model(self, training_steps):
        model_file_save = os.path.join("models/", "agent_No_" + str(self.agent_id) + "/",
                                       str(training_steps) + "_" + "model_segment_training_actor/", "8m")
        dirname = os.path.dirname(model_file_save)
        if any(dirname):
            os.makedirs(dirname, exist_ok=True)
        self.saver.save(self.sess, model_file_save)
        print("Model trained for %s times is saved" % training_steps)

    def load_model(self, model_load_steps):
        saver = tf.train.Saver()
        model_file_load = os.path.join("models/", "agent_No_" + str(self.agent_id) + "/",
                                       str(model_load_steps) + "_" + "model_segment_training_actor/", "8m")
        saver.restore(self.sess, model_file_load)
        print("model trained for %s steps of agent %s have been loaded" % (model_load_steps, self.agent_id))

class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """

    def __init__(self, sess, learning_rate, n_features, output_len, n_actions, agent_id):
        self.sess = sess
        self.learning_rate = learning_rate
        self.n_features = n_features
        self.output_len = output_len
        self.n_actions = n_actions
        self.agent_id = agent_id
        self.learn_step_counter = 0

        self.inputs, self.out = self.create_critic_network("critic_network")

        self.summary_placeholders, self.update_ops, self.summary_op, self.summary_vars, self.summary_writer = self.setup_summary()

        self.predicted_q_value = tf.placeholder(tf.float16, [None], name="predicted_q_value")

        self.critic_loss = tf.reduce_mean(tf.squared_difference(self.predicted_q_value, self.out))
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.critic_loss)

        self.saver = tf.train.Saver(max_to_keep=100000000)

    def create_critic_network(self, name):
        inputs = tf.placeholder(tf.float16, shape=(None, self.n_features), name="critic_inputs")
        out = AC.critic_build_network(name, inputs, self.n_features)
        return inputs, out

    def train(self, inputs, predicted_q_value):
        self.learn_step_counter += 1

        return self.sess.run([self.out, self.critic_loss, self.optimize], feed_dict={
            self.inputs: inputs,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
        })

    def setup_summary(self):
        fileWritePath = os.path.join("logs/", "agent_No_" + str(self.agent_id) + "/")
        summary_writer = tf.summary.FileWriter(fileWritePath, self.sess.graph)
        critic_cost = tf.Variable(0., dtype=tf.float16)
        eps_rew_agent = tf.Variable(0., dtype=tf.float16)
        eps_rew_all = tf.Variable(0., dtype=tf.float16)

        tf.summary.scalar("critic_cost", critic_cost)
        tf.summary.scalar("eps_rew_agent", eps_rew_agent)
        tf.summary.scalar("eps_rew_all", eps_rew_all)
        summary_vars = [critic_cost, eps_rew_agent, eps_rew_all]

        summary_placeholders = [tf.placeholder(tf.float16) for _ in range(len(summary_vars))]

        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()

        return summary_placeholders, update_ops, summary_op, summary_vars, summary_writer

    def plotting(self):
        tensorboard_info = [self.critic_cost, self.episode_rew_agent, self.episode_rew_all]
        vars_plot = []
        for i in range(len(tensorboard_info)):
            vars_plot.append(self.sess.run(self.update_ops[i], feed_dict={self.summary_placeholders[i]: float(tensorboard_info[i])}))

        summary_1 = tf.Summary(value=[tf.Summary.Value(tag="critic_cost", simple_value=vars_plot[0])])
        summary_2 = tf.Summary(value=[tf.Summary.Value(tag="eps_rew_agent", simple_value=vars_plot[1])])
        summary_3 = tf.Summary(value=[tf.Summary.Value(tag="eps_rew_all", simple_value=vars_plot[2])])

        self.summary_writer.add_summary(summary_1, self.learn_step_counter)
        self.summary_writer.add_summary(summary_2, self.episode)
        self.summary_writer.add_summary(summary_3, self.episode)

    def get_episode_reward(self, eps_r_agent, eps_r_all, episode):
        self.episode_rew_agent = eps_r_agent
        self.episode_rew_all = eps_r_all
        self.episode = episode

    def get_critic_loss(self, critic_cost):
        self.critic_cost = critic_cost
        self.plotting()

    def save_model(self, training_steps):
        model_file_save = os.path.join("models/", "agent_No_" + str(self.agent_id) + "/",
                                       str(training_steps) + "_" + "model_segment_training_critic/", "8m")
        dirname = os.path.dirname(model_file_save)
        if any(dirname):
            os.makedirs(dirname, exist_ok=True)
        self.saver.save(self.sess, model_file_save)
        print("Model trained for %s times is saved" % training_steps)

    def load_model(self, model_load_steps):
        saver = tf.train.Saver()
        model_file_load = os.path.join("models/", "agent_No_" + str(self.agent_id) + "/",
                                       str(model_load_steps) + "_" + "model_segment_training_critic/", "8m")
        saver.restore(self.sess, model_file_load)
        print("model trained for %s steps of agent %s have been loaded" % (model_load_steps, self.agent_id))



