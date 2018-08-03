import tensorflow as tf

class LSTM(object):
    def __init__(self, class_num, embedding_size, hidden_neural_size, num_step, hidden_layer_num, dropout = 0.9, is_training = True):

        self._dropout = dropout
    
        self.num_step = num_step
        self.class_num = class_num
        self.hidden_neural_size = hidden_neural_size
        self.embedding_size = embedding_size
        self.hidden_layer_num = hidden_layer_num

        if(self._dropout != 0):
            self._keep_prob = tf.placeholder(tf.float32)

        self._store_param = []

    def __variable_summaries(self,var):
        """attach to a Tensor for tensorboard Visualiation
        """
        mean = tf.reduce_mean(var)
        mean_scalar = tf.summary.scalar('mean',mean)
        stddev_scalar = tf.summary.scalar('stddev',tf.sqrt(tf.reduce_mean(var-mean)))
        max_scalar = tf.summary.scalar('max',tf.reduce_max(var))
        min_scalar = tf.summary.scalar('min',tf.reduce_min(var))
        hist_scalar = tf.summary.histogram('histogram',var)
        return tf.summary.merge_all()

    def __weight_init(self, shape, mean, stddev):
        init = tf.truncated_normal(shape, mean, stddev)
        return tf.Variable(init, dtype = tf.float32, name = 'weight')

    def __bias_init(self, shape, constant):
        init = tf.constant(constant, shape = shape)
        return tf.Variable(init, dtype = tf.float32, name = 'bias')

    def _add_lstm_layer(self, input):
        #Build LSTM network
        with tf.name_scope('bidirectional_lstm_layer'):
            fw_layer = []
            bw_layer = []
            for i in range(self.hidden_layer_num):
                lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_neural_size, forget_bias = 1.0)
                lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_neural_size, forget_bias = 1.0)
                if self._dropout < 1:
                    lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob = self._keep_prob)
                    lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob = self._keep_prob)
                fw_layer.append(lstm_fw_cell)
                bw_layer.append(lstm_bw_cell)

            output, _, _ = tf.contrib.rnn.stack_bidirectional_rnn(fw_layer, bw_layer, input, dtype = tf.float32)
            return output
    
    def _add_liner_layer(self, input, input_feature_num, output_feature_num, mean = 0.0, stddev = 1.0):
        with tf.name_scope('lstm_linear_layer_weight'):
            shape = [input_feature_num, output_feature_num]
            weight = self.__weight_init(shape, mean, stddev)
            self.__variable_summaries(weight)
        with tf.name_scope('lstm_linear_layer_bias'):
            shape = [output_feature_num]
            bias = self.__bias_init(shape, mean)
            self.__variable_summaries(bias)
        output = tf.matmul(input[-1], weight) + bias
        return output