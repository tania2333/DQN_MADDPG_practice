import numpy as np
import tensorflow as tf
from BicNet.guessing_sum_env import *

# TODO use the parameters of train_ddpg
HIDDEN_VECTOR_LEN = 1
NUM_AGENTS = 9
VECTOR_OBS_LEN = 4096 #1024
OUTPUT_LEN = 1


class BiCNet:
    @staticmethod
    def base_build_network(observation):
        encoded = BiCNet.shared_dense_layer("encoder", observation, HIDDEN_VECTOR_LEN)   #(batches, 2)

        hidden_agents = tf.unstack(encoded, NUM_AGENTS, 1)  #将每一列作为一个新的元素，list  [n,1 ; n,1]
        lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_VECTOR_LEN, forget_bias=1.0)  #LSTM单元中的神经元数量，即输出神经元数量;默认的激活函数是tanh  , name="lstm_fw_cell"
        lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_VECTOR_LEN, forget_bias=1.0)  #, name="lstm_bw_cell"
        outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, hidden_agents, dtype=tf.float32) #用于前向和后向传播的RNNCell; hidden_agents:输入为list,list中元素为Tensor,每个Tensor的shape为[batch_size, input_size].
        # outputs,  _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, observation, dtype=tf.float32)
        # with tf.variable_scope("bidirectional_rnn", reuse=tf.AUTO_REUSE):
        #     tf.summary.histogram("lstm_fw_cell/kernel", tf.get_variable("fw/lstm_fw_cell/kernel"))
        #     tf.summary.histogram("lstm_bw_cell/kernel", tf.get_variable("bw/lstm_bw_cell/kernel"))
        # outputs list.len=2  [n,2; n,2]
        outputs = tf.stack(outputs, 1)  #shape(,2,2)
        # outputs = tf.concat(outputs, 2)
        return outputs

    @staticmethod
    def actor_build_network(name, observation):
        with tf.variable_scope(name):
            # print('obs',observation.shape)
            outputs = BiCNet.base_build_network(observation)
            outputs = BiCNet.shared_dense_layer("output_layer", outputs, OUTPUT_LEN)
            return tf.clip_by_value(outputs, -1, 1)#outputs


    @staticmethod # 理解此处的建模方法是本代码实现的关键
    def shared_dense_layer(name, observation, output_len):
        H = []
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            for j in range(NUM_AGENTS):
                agent_obs = observation[:, j]   #（batches, 1) or 2
                agent_encoded = tf.layers.dense(agent_obs, output_len, name="dense")  # always (batches, 1)  out_len:输出的维度大小，outputs会改变inputs的最后一维为output_len
                tf.summary.histogram(name + "/dense/kernel", tf.get_variable("dense/kernel"))
                H.append(agent_encoded)
            H = tf.stack(H, 1)  #来自同一组的放在一列进行拼接  (batches, 2)
        return H

    @staticmethod
    def critic_build_network(name, observation, action):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            outputs = BiCNet.base_build_network(tf.concat([observation, action], 2))  #对第三维进行拼接： b,num,1-> b,num,2
            outputs = BiCNet.shared_dense_layer("output_layer", outputs, 1)
            return outputs

if __name__ == '__main__':
    tf.set_random_seed(42)

    tf.reset_default_graph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        BATCH_SIZE = 10

        observation = tf.placeholder(tf.float32, shape=(None, NUM_AGENTS, VECTOR_OBS_LEN), name="observation")
        actions = tf.placeholder(tf.float32, shape=(None, NUM_AGENTS, OUTPUT_LEN), name="actions")

        actor_out = BiCNet.actor_build_network("actor_network", observation)
        critic_out = BiCNet.critic_build_network("critic_network", observation, actions)

        sess.run(tf.global_variables_initializer())

        feed_dict = {observation: np.random.random_sample((BATCH_SIZE, NUM_AGENTS, VECTOR_OBS_LEN))}
        print(sess.run(actor_out, feed_dict=feed_dict).shape, "==", (BATCH_SIZE, NUM_AGENTS, OUTPUT_LEN), "== (BATCH_SIZE, NUM_AGENTS, OUTPUT_LEN)")

        feed_dict = {observation: np.random.random_sample((BATCH_SIZE, NUM_AGENTS, VECTOR_OBS_LEN)),
                     actions: np.random.random_sample((BATCH_SIZE, NUM_AGENTS, OUTPUT_LEN))}
        print(sess.run(critic_out, feed_dict=feed_dict).shape, "==", (BATCH_SIZE, NUM_AGENTS, 1), "== (BATCH_SIZE, NUM_AGENTS, 1)")

        feed_dict = {observation: np.random.random_sample((1, NUM_AGENTS, VECTOR_OBS_LEN))}
        print(sess.run(actor_out, feed_dict=feed_dict).shape, "==", (1, NUM_AGENTS, OUTPUT_LEN), "== (BATCH_SIZE, NUM_AGENTS, OUTPUT_LEN)")

        feed_dict = {observation: np.random.random_sample((1, NUM_AGENTS, VECTOR_OBS_LEN)),
                     actions: np.random.random_sample((1, NUM_AGENTS, OUTPUT_LEN))}
        print(sess.run(critic_out, feed_dict=feed_dict).shape, "==", (1, NUM_AGENTS, 1), "== (BATCH_SIZE, NUM_AGENTS, 1)")
