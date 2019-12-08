"""
Implementation of DDPG - Deep Deterministic Policy Gradient https://github.com/pemami4911/deep-rl
Modified by Coac for BiCNet implementation https://github.com/Coac/CommNet-BiCnet
"""
import argparse
import pprint as pp
from datetime import datetime

import numpy as np
import tensorflow as tf

from BicNet.ref_bicnet import BiCNet as CommNet
from BicNet.guessing_sum_env import *
from BicNet.replay_buffer import ReplayBuffer

HIDDEN_VECTOR_LEN = 1
NUM_AGENTS = 9  #2
VECTOR_OBS_LEN = 4096 #1024 #1
OUTPUT_LEN = 1


# ===========================
#   Actor and Critic DNNs
# ===========================

class ActorNetwork(object):
    def __init__(self, sess, learning_rate, tau, batch_size, NUM_AGENTS,VECTOR_OBS_LEN,OUTPUT_LEN):
        self.sess = sess
        # self.s_dim = state_dim
        # self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

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

        self.action_gradient = tf.placeholder(tf.float32, (NUM_AGENTS, None, NUM_AGENTS, OUTPUT_LEN), name="action_gradient")# print('action_grad',self.action_gradient) 2,n,2,1 /3,?,3,1

        with tf.name_scope("actor_gradients"):
            grads = []
            for i in range(NUM_AGENTS):
                for j in range(NUM_AGENTS):
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
        inputs = tf.placeholder(tf.float32, shape=(None, NUM_AGENTS, VECTOR_OBS_LEN), name="actor_inputs")
        # print('input',inputs)
        out = CommNet.actor_build_network(name, inputs)   #(, 2,1)
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

    def __init__(self, sess, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        # self.s_dim = state_dim
        # self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        self.inputs, self.action, self.out = self.create_critic_network("critic_network")
        self.network_params = tf.trainable_variables()[num_actor_vars:]

        self.target_inputs, self.target_action, self.target_out = self.create_critic_network("target_critic_network")
        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        with tf.name_scope("critic_update_target_network_params"):
            self.update_target_network_params = \
                [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau)
                                                      + tf.multiply(self.target_network_params[i], 1. - self.tau))
                 for i in range(len(self.target_network_params))]

        self.predicted_q_value = tf.placeholder(tf.float32, (None, NUM_AGENTS, 1), name="predicted_q_value")

        M = tf.to_float(tf.shape(self.out)[0])  # num of minibatches
        # Li = (Yi - Qi)^2
        # L = Sum(Li)

        grads = []
        for i in range(NUM_AGENTS):
            grads.append(tf.gradients(self.out[:, i], self.network_params, -(self.predicted_q_value[:, i] - self.out[:, i])))
        grads = np.array(grads)
        # print(grads.shape)    #(2, 8)
        self.unnormalized_critic_gradients = [tf.reduce_sum(list(grads[:, i]), axis=0) for i in
                                             range(len(self.network_params))]  # len_net_param=8
        self.critic_gradients = list(map(lambda x: tf.div(x, M),
                                        self.unnormalized_critic_gradients))  # unn(1,8)  len(critic_gradients)=8
        self.optimize = tf.train.AdamOptimizer(self.learning_rate)
        self.optimize = self.optimize.apply_gradients(zip(self.critic_gradients, self.network_params))

        # self.action_grads = tf.gradients(self.out, self.action, name="action_grads")
        self.action_grads = [tf.gradients(self.out[:, i], self.action) for i in range(NUM_AGENTS)] # out.shape:batchs,2,1; action: n,2,1;
        # action_grads = [[grad1],[grad2]]  grad1.shape=n ,2, 1
        self.action_grads = tf.stack(tf.squeeze(self.action_grads, 1))   #tf.squeeze: 默认删除所有为1的维度   (2, ?, 2, 1)

    def create_critic_network(self, name):
        inputs = tf.placeholder(tf.float32, shape=(None, NUM_AGENTS, VECTOR_OBS_LEN), name="critic_inputs")
        action = tf.placeholder(tf.float32, shape=(None, NUM_AGENTS, OUTPUT_LEN), name="critic_action")

        out = CommNet.critic_build_network(name, inputs, action)
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


# ===========================
#   Tensorflow Summary Ops
# ===========================

def build_summaries():
    episode_reward = tf.Variable(0., name="episode_reward")
    tf.summary.scalar("Reward", episode_reward)  # 用来显示标量信息，在tensorboard上可查看
    episode_ave_max_q = tf.Variable(0., name="episode_ave_max_q")
    tf.summary.scalar("Qmax Value", episode_ave_max_q)
    # loss = tf.Variable(0., name="critic_loss")
    # tf.summary.scalar("Critic_loss", loss)

    # summary_vars = [episode_reward, episode_ave_max_q, loss]
    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all() #merge_all 可以将所有summary全部保存到磁盘，以便tensorboard显示

    return summary_ops, summary_vars


# ===========================
#   Agent Training
# ===========================

def train(sess, env, args, actor, critic):
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(args['summary_dir'] +  " actor_lr" + str(args['actor_lr']) + " critic_lr" + str(args["critic_lr"]), sess.graph)

    actor.update_target_network()
    critic.update_target_network()

    replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))

    for i in range(int(args['max_episodes'])):
        state = env.reset() #(2,1)

        ep_reward = 0
        ep_ave_max_q = 0

        for j in range(int(args['max_episode_len'])):
            action = actor.predict([state])[0]    #(2,1)
            # print('action',action)     #为什么输出的这个action在-1到1之间，网络最后一层全连接层哪部分设定的？？？？？？

            state2, reward, done, info = env.step(action)  # reward->(2,1)  done = true

            # state2 = np.row_stack((state2, state2))
            replay_buffer.add(state, action, reward, done, state2)
            if replay_buffer.size() > int(args['minibatch_size']):
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(int(args['minibatch_size']))
                # TODO
                # Calculate targets
                # target_q = tf.zeros((1))
                # print('r',r_batch)  #(1024, 2, 1)
                target_q = r_batch + 0.99 * critic.predict_target(s2_batch, actor.predict_target(s2_batch))

                # Update the critic given the targets
                # predicted_q_value, _, loss = critic.train(s_batch, a_batch,
                predicted_q_value, _= critic.train(s_batch, a_batch,
                                                          np.reshape(target_q, (int(args['minibatch_size']), NUM_AGENTS, 1)))
                ep_ave_max_q += np.amax(predicted_q_value)
                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)       # a_outs和a_batch是完全相同的
                grads = critic.action_gradients(s_batch, a_outs)   # delta Q对a的导数
                actor.train(s_batch, grads)                         #这里会计算a对θ的导数和最后的梯度

                actor.update_target_network()
                critic.update_target_network()

                replay_buffer.clear()

                # Log
                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: np.mean(r_batch),
                    summary_vars[1]: ep_ave_max_q / float(j + 1),
                    # summary_vars[2]: loss
                })

                writer.add_summary(summary_str, i)
                writer.flush()

                print('| Reward: {:.4f} | Episode: {:d} | Qmax: {:.4f}'.format(np.mean(r_batch),
                                                                               i, (ep_ave_max_q / float(j + 1))))

            state = state2
            ep_reward += reward

            if done:
                break


def main(args=None):
    args = parse_arg(args or None)

    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        env = GuessingSumEnv(NUM_AGENTS)
        env.seed(0)

        np.random.seed(int(args['random_seed']))      # 与np.random.randn随机数产生相关
        tf.set_random_seed(int(args['random_seed']))  # 为了使所有op产生的随机序列在会话之间是可重复的，比如本次执行中的sess1和2中随机数产生序列是一样的
        env.seed(int(args['random_seed']))

        state_dim = (NUM_AGENTS, VECTOR_OBS_LEN)   # （2,1）
        action_dim = (NUM_AGENTS, OUTPUT_LEN)      # （2,1）

        actor = ActorNetwork(sess, state_dim, action_dim,
                             float(args['actor_lr']), float(args['tau']),
                             int(args['minibatch_size']),NUM_AGENTS,VECTOR_OBS_LEN,OUTPUT_LEN)

        critic = CriticNetwork(sess, state_dim, action_dim,
                               float(args['critic_lr']), float(args['tau']),
                               float(args['gamma']),
                               actor.get_num_trainable_vars())

        train(sess, env, args, actor, critic)


def parse_arg(args=None):
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.1)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.1)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--tau', help='soft target update parameter', default=0.001)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=1024)

    # run parameters
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=9999999999999)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1000)
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info',
                        default="summaries/" + datetime.now().strftime('%d-%m-%y %H%M'))

    if args is not None:
        args = vars(parser.parse_args(args))
    else:
        args = vars(parser.parse_args())

    pp.pprint(args)

    return args


if __name__ == '__main__':
    main()
