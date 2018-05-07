"""
    Extract the feature with the NetVALD algorithm
    paper: NetVLAD: CNN architecture for weakly supervised place recognition
    author heyubo
    create: 2018/4/20
    last update: 2018/4/23
"""
import tensorflow as tf

class NetVLAD(object):
    def __init__(self, cluster_num):
        """the netvlad class initialization

        @param cluster_num : the number of cluster center
        """
        self._cluser_num = cluster_num

    def get_cluster_num(self):
        return self._cluser_num

    def set_cluster_num(self, cluster_num):
        self._cluser_num = cluster_num

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

    def __variable_init(self, shape, mean, stddev):
        init = tf.truncated_normal(shape, mean, stddev)
        return tf.Variable(init, dtype = tf.float32, name = 'weight')

    def __constant_init(self, shape, constant):
        init = tf.constant(constant, shape = shape)
        return tf.Variable(init, dtype = tf.float32, name = 'bias')

    def _add_vald_layer(self, input, channel_in, name, mean = 0.0, stddev = 0.001):
        """add a netvald layer in the neural network model

        for future development:

        @param input: the input tensor
        @param channel_in : the former layer output feature num
        @param name : the layer scope name
        @param mean : the mean of the weight and bais
        @param weight : the standard deviation
        """
        with tf.variable_scope(name) as scope:
            #initialize the variable
            with tf.name_scope('weight'):
                shape = [1, channel_in, 1, self._cluser_num]
                weight = self.__variable_init(shape, mean, stddev)
                self.__variable_summaries(weight)
            with tf.name_scope('bias'):
                shape = [self._cluser_num]

                bias = self.__variable_init(shape, mean, stddev)
                self.__variable_summaries(bias)
            with tf.name_scope('center'):
                shape = [channel_in, self._cluser_num]
                center = self.__variable_init(shape, mean, stddev)
                self.__variable_summaries(center)

            input_reshape = tf.reshape(input, shape = [-1, (input.get_shape().as_list()[1] * input.get_shape().as_list()[2]), channel_in], name = 'reshape')
            input_norm = tf.nn.l2_normalize(input_reshape, axis = 1)
            descriptor = tf.expand_dims(input_norm, axis = -1, name = 'expand_dim')
            conv_vlad = tf.nn.convolution(descriptor, weight, padding = 'VALID')
            bias = tf.nn.bias_add(conv_vlad, bias)
            a_k = tf.nn.softmax(tf.squeeze(bias, axis = 2), axis = -1, name = 'vlad_softmax')

            V1 = tf.matmul(input_reshape, a_k, transpose_a = True)
            V2 = tf.multiply(tf.reduce_sum(a_k, axis = 1, keepdims = True), center)
            V = tf.subtract(V1, V2)
            norm = tf.nn.l2_normalize(tf.reshape(tf.nn.l2_normalize(V, axis = 1), shape = [-1, channel_in * self._cluser_num]), axis = 1, name = 'output')
            
        return norm