#coding=utf-8
"""
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.examples.tutorials.mnist.input_data as input_data
#import input_data
import tensorflow as tf
import math
import os

#tensorflow embedding
from tensorboard.plugins import projector

class CNN():
    def __init__(self, dropout = 0.9):
        """initialize the cnn class

        @param dropout : the dropout ratio in the fully_connect layer, if 1 ,do not use dropout
        """
        self._dropout = dropout

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

    def set_log_dir(self,log_dir):
        """set the Tensorboard dictionary
        @param log_dir  Tensorboard dictionary  
        """
        self._log_dir = log_dir

    def __weight_init(self, shape, mean, stddev):
        init = tf.truncated_normal(shape, mean, stddev)
        return tf.Variable(init, dtype = tf.float32, name = 'weight')

    def __bias_init(self, shape, constant):
        init = tf.constant(constant, shape = shape)
        return tf.Variable(init, dtype = tf.float32, name = 'bias')

    def _add_conv_layer(self,input,layer_index,conv_kernel_x,conv_kernel_y,stride,input_feature_num,output_feature_num,mean = 0.0,stddev = 1.0):
        """add a convolution layer to the model

        convolution kernel size(shape):[conv_kernel_x,conv_kernel_y,input_feature_num,output_feature_num]
        the weight obeys Gauss distribution.

        @param input: the input tensor to the layer
        @param layer_index: the layer index of the layer
        @param conv_kernel_x: the x axis size of the convolution kernel
        @param conv_kernel_y: the y axis size of the convolution kernel
        @param stride: the size of stride
        @param input_feature_num: the number of the input feature
        @param output_feature_num: the number of the output feature
        @param mean: the mean of the weight initilization
        @param stddev: the standard devision of the weight initilization 

        @return a image with operation convolution and max_pooling
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
        return conv

    def _add_pool(self, input, layer_index, ksize, stride, padding = 'SAME'):
        """add a pooling layer to the model

        @param input: the input tensor to the layer
        @param layer_index：the index of the layer
        @param ksize: the size of the window for each dimension of the input tensor
        @param stride：the stride of the sliding window for each dimension of the input tensor
        @param padding：the padding algorithm
        """
        string = 'pool_' + str(layer_index)
        with tf.variable_scope(string) as scope:
            pool = tf.nn.max_pool(input, ksize, stride, padding = padding,name = 'pool')
            #norm = tf.nn.lrn(pool,depth_radius = 4,bias = 1.0, alpha = self.__learning_rate, beta = 0.75,name = 'normal')
        
        return pool

    def _add_fclayer(self,input,layer_index,input_feature_num,output_feature_num,mean = 0.0,stddev = 1.0):
        """add a fully connect layer to the model

        the weight obeys Gauss distribution.

        @param input: the input tensor to the layer
        @param layer_index: the layer index of the layer
        @param input_feature_num: the number of the input feature
        @param output_feature_num: the number of the output feature
        @param mean: the mean of the weight initilization
        @param stddev: the standard devision of the weight initilization 

        @return a tensor
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

        @param input: the input tensor to the layer
        @param input_feature_num: the number of the input feature
        @param mean: the mean of the weight initilization
        @param stddev: the standard devision of the weight initilization 

        @return a tensor
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