#coding=utf-8

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
    def __init__(self, learning_rate = 0.001, step = 1000, batch_size = 100, dropout = 0.9):
        """initialize the cnn class

        @param learning_rate : the cnn model's learning_rate
        @param step : the step to iterate the model learning
        @param batch_size : the number of image in each batch
        @param dropout : the dropout ratio in the fully_connect layer, if 1 ,do not use dropout
        """
        self._learning_rate = learning_rate
        self._step = step
        self._batch_size = batch_size
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
            pool = tf.nn.max_pool(input, ksize, stride, padding = padding)
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

class My_CNN(CNN):
    """see the example on the http://www.cnblogs.com/denny402/p/5853538.html
    simplest example to achieve the CNN
    """
    def __init__(self, input_size, output_size, learning_rate = 0.001, step = 1000, batch_size = 100, dropout = 0.9):
        """construction function
        @param input_size the size of input
        @param output_size the size of output
        @param learning_rate the learning_rate to train the model, default 0.0001
        @param step the iterate number to train the model, default 1000(not enough)
        @param batch_size the batch number per step
        @param dropout whether uses dropout
        """
        super(My_CNN,self).__init__(input_size, output_size, learning_rate, step, batch_size, dropout)

        edge_len = int(math.sqrt(input_size))
        assert edge_len * edge_len == input_size
        self.__ximage = tf.reshape(self._x,[-1,edge_len,edge_len,1])
        self.__train_step, self.__accuracy= self.__model()

        #tensorflow embedding
        self.__embedding_var = tf.Variable(tf.zeros([self._batch_size, 28, 28]), name = 'embedding')
        self.__assignment = self.__embedding_var.assign(self.__embedding_var)
        self.__saver = tf.train.Saver()

        self.__sess = tf.Session()
        init = tf.global_variables_initializer()
        self.__sess.run(init)

    def __model(self):
        """model architecture: 
        convolution layer;max pooling;convolution layer;max poolinglfully connect layer;fully connect layer

        @return accuracy accuracy of the model to the sample
        """
        #with tf.name_scope('model'):
        #    conv1 = self._add_conv_layer(self.__ximage,1,3,3,[1,1,1,1],1,1,stddev = 0.1)
        #    conv2 = self._add_conv_layer(conv1,2,3,3,[1,1,1,1],1,32,stddev = 0.1)
        #    norm1 = self._add_pool(conv2, 1, [1,2,2,1], [1,2,2,1])
        #    conv3 = self._add_conv_layer(norm1,3,3,3,[1,1,1,1],32,32,stddev = 0.1)
        #    conv4 = self._add_conv_layer(conv3,4,3,3,[1,1,1,1],32,64,stddev = 0.1)
        #    norm2 = self._add_pool(conv4, 1, [1,2,2,1], [1,2,2,1])
        #    fc1 = self._add_fclayer(norm2,1,7*7*64,1024,stddev = 0.1)
        #    predict = self._output_layer(fc1,1024,stddev = 0.1)
        #    #predict_hist = tf.summary.histogram('prediction', predict)
        with tf.name_scope('model'):
            tf.summary.image('input', self.__ximage, 10)
            conv1 = self._add_conv_layer(self.__ximage,1,5,5,[1,1,1,1],1,32,stddev = 0.1)
            norm1 = self._add_pool(conv1, 1, [1,2,2,1], [1,2,2,1])
            conv2 = self._add_conv_layer(norm1,3,5,5,[1,1,1,1],32,64,stddev = 0.1)
            norm2 = self._add_pool(conv2, 1, [1,2,2,1], [1,2,2,1])
            #image_transpose = tf.transpose(norm2, perm = [3, 1, 2, 0])
            fc1 = self._add_fclayer(norm2,1,7*7*64,1024,stddev = 0.1)
            predict = self._output_layer(fc1,1024,stddev = 0.1)
        with tf.name_scope('loss'):
            cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits = predict, labels = self._accurate_data))
            loss_scalar = tf.summary.scalar('cross_entropy', cross_entropy)
        with tf.name_scope('optimizer'):
            #train_step = tf.train.GradientDescentOptimizer(self._learning_rate).minimize(cross_entropy)
            train_step = tf.train.AdamOptimizer(self._learning_rate).minimize(cross_entropy)
        with tf.name_scope('accuarcy'):
            correct_prediction = tf.equal(tf.argmax(predict, 1), tf.argmax(self._accurate_data, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
            accu_scalar = tf.summary.scalar('accuarcy', accuracy)

        return train_step, accuracy

    def train(self):
        """train the model described above
        """
        #flow = input_data.read_data_sets()
        flow = input_data.read_data_sets("MNIST_data/", one_hot=True)

        #merge the summary and write it to the tensorboard
        merge = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(self._log_dir+'/train/log',self.__sess.graph)
        test_writer = tf.summary.FileWriter(self._log_dir+'/test/log',self.__sess.graph)
        '''
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = self.__embedding_var.name
        embedding.sprite.image_path = self._sprite_image_dir
        embedding.metadata_path = self._label_dir
        embedding.sprite.single_image_dim.extend([28, 28])
        projector.visualize_embeddings(train_writer, config)
        '''
        for i in range(self._step):
            batch_xs, batch_ys = flow.train.next_batch(self._batch_size)
            self.__sess.run(self.__train_step, feed_dict = {self._x : batch_xs, self._accurate_data : batch_ys, self._keep_prob : self._dropout})
            accuarcy = self.__sess.run(self.__accuracy, feed_dict = {self._x : batch_xs, self._accurate_data : batch_ys, self._keep_prob : self._dropout})
            if i % 100 == 99:
                summary = self.__sess.run(merge, feed_dict = {self._x : batch_xs, self._accurate_data : batch_ys, self._keep_prob : 1})
                train_writer.add_summary(summary, i)
                print('round {} accuarcy: {:.6f}'.format(i,accuarcy))
            '''
            if i % 100 == 0:
                self.__saver.save(self.__sess, os.path.join(self._log_dir, 'model.ckpt'), i)
            '''
        #print(self.__sess.run(self.__accuracy,feed_dict={self._x:mnist.test.images,self._accurate_data:mnist.test.labels,self._keep_prob:0.5}))
        def set_sprite_image_dir(self, dir):
            self._sprite_image_dir = dir

        def set_label_dir(self, dir):
            self._label_dir = dir