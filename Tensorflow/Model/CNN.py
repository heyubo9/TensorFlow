#coding=utf-8
"""
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

##tensorflow embedding
#from tensorboard.plugins import projector

class CNN():
    store_param = []

    def __init__(self, dropout = 0.9, visualization = False):
        """initialize the cnn class

        @param dropout : the dropout ratio in the fully_connect layer, if 1 ,do not use dropout
        """
        self._dropout = dropout
        self._cnn_visualization = visualization

    def __weight_init(self, shape, mean, stddev):
        init = tf.truncated_normal(shape, mean, stddev)
        return tf.Variable(init, dtype = tf.float32, name = 'weight')

    def __bias_init(self, shape, constant):
        init = tf.constant(constant, shape = shape)
        return tf.Variable(init, dtype = tf.float32, name = 'bias')

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
    
    def _add_conv1d_layer(self, input, layer_index, conv_kernel_length, stride, input_feature_num, output_feature_num, mean = 0.0, stddev = 1.0):
        """add a 1D-convolution layer to the model

        convolution kernel size(shape):[conv_kernel_length, input_feature_num, output_feature_num]
        the weight obeys Gauss distribution.
        Args:
        	input: the input tensor to the layer, size = [batch_num, input_width, channel_in]
        	layer_index: the layer index of the layer
            conv_kernel_length: the length of the convolutional filter
        	stride: the size of stride
        	input_feature_num: the number of the input feature
        	output_feature_num: the number of the output feature
        	mean: the mean of the weight initilization
        	stddev: the standard devision of the weight initilization 
        Return:
            output: a image with operation convolution and max_pooling
        """
        string = 'conv_'+str(layer_index)
        with tf.variable_scope(string) as scope:
            with tf.name_scope('weight'):
                shape = [conv_kernel_length, input_feature_num,output_feature_num]
                weight = self.__weight_init(shape, mean, stddev)
                self.__variable_summaries(weight)
            with tf.name_scope('bias'):
                shape = [output_feature_num]
                bias = self.__bias_init(shape, 0.1)
                self.__variable_summaries(bias)
            conv = tf.nn.conv1d(input, weight, stride, padding = 'SAME')
            sconv = tf.nn.bias_add(conv, bias)
            conv = tf.nn.relu(conv, name = scope.name)

            if self._vis_layer_num >= layer_index and self._cnn_visualization == True:
                self.store_param.append(tf.shape(input))
                self.store_param.append(weight)
                self.store_param.append(bias)
        return conv

    def _add_conv2d_layer(self,input,layer_index,conv_kernel_x,conv_kernel_y,stride,input_feature_num,output_feature_num,mean = 0.0,stddev = 1.0):
        """add a 2D-convolution layer to the model

        convolution kernel size(shape):[conv_kernel_x,conv_kernel_y,input_feature_num,output_feature_num]
        the weight obeys Gauss distribution.
        Args:
        	input: the input tensor to the layer
        	layer_index: the layer index of the layer
        	conv_kernel_x: the x axis size of the convolution kernel
        	conv_kernel_y: the y axis size of the convolution kernel
        	stride: the size of stride
        	input_feature_num: the number of the input feature
        	output_feature_num: the number of the output feature
        	mean: the mean of the weight initilization
        	stddev: the standard devision of the weight initilization 
        Return:
            output: a image with operation convolution and max_pooling
        """
        string = 'conv_'+str(layer_index)
        with tf.variable_scope(string) as scope:
            with tf.name_scope('weight'):
                shape = [conv_kernel_x,conv_kernel_y,input_feature_num,output_feature_num]
                weight = self.__weight_init(shape, mean, stddev)
                self.__variable_summaries(weight)
            with tf.name_scope('bias'):
                shape = [output_feature_num]
                bias = self.__bias_init(shape, 0.1)
                self.__variable_summaries(bias)
            conv = tf.nn.conv2d(input, weight, stride, padding = 'SAME')
            sconv = tf.nn.bias_add(conv, bias)
            conv = tf.nn.relu(conv, name = scope.name)

            if self._vis_layer_num >= layer_index and self._cnn_visualization == True:
                self.store_param.append(tf.shape(input))
                self.store_param.append(weight)
                self.store_param.append(bias)
        return conv

    def _add_pool(self, input, layer_index, ksize, stride, padding = 'SAME'):
        """add a pooling layer to the model for 2D convolutional neural network 
        Args:
        	input: the input tensor to the layer
        	layer_index：the index of the layer
        	ksize: the size of the window for each dimension of the input tensor
        	stride：the stride of the sliding window for each dimension of the input tensor
        	padding：the padding algorithm
        """
        string = 'pool_' + str(layer_index)
        with tf.variable_scope(string) as scope:
            return tf.nn.max_pool(input, ksize, stride, padding = padding,name = 'pool')

    def _add_pool1d(self, input, layer_index, window_shape, stride, padding = 'SAME'):
        string = 'pool_' + str(layer_index)
        with tf.variable_scope(string) as scope:
            return tf.nn.pool(input, window_shape, pooling_type = 'MAX', padding = padding)

    def _add_fclayer(self,input,layer_index,input_feature_num,output_feature_num,mean = 0.0,stddev = 1.0):
        """add a fully connect layer to the model

        the weight obeys Gauss distribution.
        Args：
        	input: the input tensor to the layer
        	layer_index: the layer index of the layer
        	input_feature_num: the number of the input feature
        	output_feature_num: the number of the output feature
        	mean: the mean of the weight initilization
        	stddev: the standard devision of the weight initilization 
        Return
            output: a tensor
        """
        string = 'fully_connect_layer_' + str(layer_index)
        with tf.name_scope(string):
            with tf.name_scope('weight'):
                shape = [input_feature_num,output_feature_num]
                weight = self.__weight_init(shape,mean,stddev)
                self.__variable_summaries(weight)
            with tf.name_scope('bias'):
                shape = [output_feature_num]
                bias = self.__bias_init(shape,0.1)
                self.__variable_summaries(bias)
            flat = tf.reshape(input,[-1,input_feature_num])
            fclayer_result = tf.nn.relu(tf.matmul(flat, weight) + bias)
        return fclayer_result
    
    def _output_layer(self,input,input_feature_num,output_size, mean = 0.0,stddev = 1.0):
        """add a fully connect layer to the model

        the weight obeys Gauss distribution.
        Args:
            input: the input tensor to the layer
            input_feature_num: the number of the input feature
            mean: the mean of the weight initilization
            stddev: the standard devision of the weight initilization 
        Return:
            output: a tensor
        """
        #dropout option
        if self._dropout != 0:
            self._keep_prob = tf.placeholder(tf.float32)
            input = tf.nn.dropout(input, keep_prob = self._keep_prob, name='dropout')

        with tf.name_scope('output_layer'):
            with tf.name_scope('weight'):
                shape = [input_feature_num, output_size]
                weight = self.__weight_init(shape, mean, stddev)
                self.__variable_summaries(weight)
            with tf.name_scope('bias'):
                shape = [output_size]
                bias = self.__bias_init(shape,0.1)
                self.__variable_summaries(bias)
            result = tf.matmul(input, weight) + bias

        return result

    def _add_unpool_layer(self, input, ind, stride = (1, 2, 2, 1), scope='unpool'):
        """Adds a 2D unpooling op.
        https://arxiv.org/abs/1505.04366
        Unpooling layer after max_pool_with_argmax.
        Args:
            pool:        max pooled output tensor
            ind:         argmax indices
            stride:      stride is the same as for the pool
        Return:
            unpool:    unpooling tensor
        """
        with tf.variable_scope(scope):
            input_shape = tf.shape(input)
            output_shape = [input_shape[0], input_shape[1] * stride[1], input_shape[2] * stride[2], input_shape[3]]

            flat_input_size = tf.reduce_prod(input_shape)
            flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

            pool_ = tf.reshape(input, [flat_input_size])
            batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype), 
                                                shape=[input_shape[0], 1, 1, 1])
            b = tf.ones_like(ind) * batch_range
            b1 = tf.reshape(b, [flat_input_size, 1])
            ind_ = tf.reshape(ind, [flat_input_size, 1])
            ind_ = tf.concat([b1, ind_], 1)

            ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
            ret = tf.reshape(ret, output_shape)

            set_input_shape = input.get_shape()
            set_output_shape = [set_input_shape[0], set_input_shape[1] * stride[1], set_input_shape[2] * stride[2], set_input_shape[3]]
            ret.set_shape(set_output_shape)
            return ret

    def _add_deconv2d_layer(self, input, filter, output_shape, stride):
        input = tf.nn.relu(input)
        result = tf.nn.conv2d_transpose(input, filter, output_shape, stride)
        return result

    def _add_deconv1d_layer(self, input, filter, output_shape, stride):
        input = tf.nn.relu(input)
        result = tf.nn.conv1d_transpose(input, filter, output_shape, stride)
        return result
