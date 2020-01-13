import numpy as np
import tensorflow as tf

class Q_Net:
    @staticmethod
    def base_build_network(observation, num_agents, hidden_vector_len):
        encoded = Q_Net.shared_dense_layer("encoder", observation, num_agents, hidden_vector_len)   #(batches, 2)

        hidden_agents = tf.unstack(encoded, num_agents, 1)  #将每一列作为一个新的元素，list  [n,1 ; n,1]
        lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_vector_len, forget_bias=1.0, reuse=tf.AUTO_REUSE)  #LSTM单元中的神经元数量，即输出神经元数量;默认的激活函数是tanh  , name="lstm_fw_cell"
        lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_vector_len, forget_bias=1.0, reuse=tf.AUTO_REUSE)  #, name="lstm_bw_cell"
        outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, hidden_agents, dtype=tf.float32) #用于前向和后向传播的RNNCell; hidden_agents:输入为list,list中元素为Tensor,每个Tensor的shape为[batch_size, input_size].
        # outputs,  _ = tf.nn.
        # (lstm_fw_cell, lstm_bw_cell, observation, dtype=tf.float32)
        # with tf.variable_scope("bidirectional_rnn", reuse=tf.AUTO_REUSE):
        #     tf.summary.histogram("lstm_fw_cell/kernel", tf.get_variable("fw/lstm_fw_cell/kernel"))
        #     tf.summary.histogram("lstm_bw_cell/kernel", tf.get_variable("bw/lstm_bw_cell/kernel"))
        # outputs list.len=2  [n,2; n,2]
        outputs = tf.stack(outputs, 1)  #shape(,2,2)
        # print("outputs_shape", outputs.shape)  #outputs_shape (?, 9, 2)
        # outputs = tf.concat(outputs, 2)
        return outputs

    @staticmethod  # 理解此处的建模方法是本代码实现的关键
    def shared_dense_layer(name, observation, num_agents, output_len):
        H = []
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            for j in range(num_agents):
                agent_obs = observation[:, j]  # （batches, 1) or 2
                agent_encoded = tf.layers.dense(agent_obs, output_len,
                                                name="dense")  # always (batches, 1)  out_len:输出的维度大小，outputs会改变inputs的最后一维为output_len
                tf.summary.histogram(name + "/dense/kernel", tf.get_variable("dense/kernel"))
                H.append(agent_encoded)
            H = tf.stack(H, 1)  # 来自同一组的放在一列进行拼接  (batches, 2)
        return H

    @staticmethod
    def build_network(name, observation, num_agents, hidden_vector_len,output_len):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            outputs = Q_Net.base_build_network(observation, num_agents, hidden_vector_len)  #对第三维进行拼接： b,num,1-> b,num,2
            outputs = Q_Net.shared_dense_layer("output_layer", outputs, num_agents, output_len)
            return outputs
