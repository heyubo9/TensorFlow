#coding=utf-8
from Model import CNN
from Model import VLAD
from Model import LSTM

import tensorflow as tf
#import tensorflow.examples.tutorials.mnist.input_data as input_data
from tensorflow.python.client import device_lib as _device_lib
import input_data
from input_data import read_csv_dataset, read_csv_test

import matplotlib.pyplot as plt
import numpy as np
import math

class nn(CNN.CNN, VLAD.NetVLAD, LSTM.LSTM):
    def __init__(self, input_size, output_size, cluster_num, hidden_neural_size, num_step, hidden_layer_num = 1, embedding_size = 0, cnn_learning_rate = 0.001, rnn_learning_rate = 0.01, cnn_step = 1000, rnn_step = 5000, batch_size = 100, dropout = 0.9, is_training = True, visualization_cnn = False):
        """construction function
        Args:
        	input_size the size of input
        	output_size the size of output
        	cluster_num the kmeans cluster number of the feature image
        	learning_rate the learning_rate to train the model, default 0.0001
        	step the iterate number to train the model, default 1000(not enough)
        	batch_size the batch number per step
        	dropout whether uses dropout
        """
        self._input_size = input_size
        self._output_size = output_size
        self._cnn_learning_rate = cnn_learning_rate
        self._rnn_learning_rate = rnn_learning_rate
        self._cnn_step = cnn_step
        self._rnn_step = rnn_step
        self._batch_size = batch_size
        self._cnn_visualization = visualization_cnn
        self._vis_layer_num = 1
        if embedding_size == 0:
            embedding_size = 64 * cluster_num
        else:
            embedding_size = embedding_size

        CNN.CNN.__init__(self, dropout)
        VLAD.NetVLAD.__init__(self, cluster_num)
        LSTM.LSTM.__init__(self, output_size, embedding_size, hidden_neural_size, num_step, hidden_layer_num, dropout, is_training)
        
        #with tf.name_scope('output'):
        #    self._accurate_data = tf.placeholder(tf.float32,[None,self._output_size],name = 'output')



        self.flow = input_data.read_data_sets()
        #self.flow = input_data.read_data_sets("MNIST_data/", one_hot=True)
        
    def set_log_dir(self,log_dir):
        """set the Tensorboard dictionary
        @Args:
            log_dir  Tensorboard dictionary  
        """
        self._log_dir = log_dir

    def set_cnn_visualization(self, bool = True, layer_num = 1):
        """set the visualize index of the layer
        Args:
            bool: boolean is not visualization
            layer_num: the visualization of convolution layer index
        """
        self._cnn_visualization = bool
        self._vis_layer_num = layer_num

    def get_cnn_visualization(self):
        return self._cnn_visualization

    def close_sess(self):
        self.__sess.close()

    def is_gpu_available(self):
        """
        code from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/platform/test.py
        Returns whether TensorFlow can access a GPU.
        Args:
        cuda_only: limit the search to CUDA gpus.
        Returns:
        True iff a gpu device of the requested kind is available.
        """
        return any((x.device_type == 'GPU') for x in _device_lib.list_local_devices())

    def __model_cnn(self):
        with tf.name_scope('cnn_model'):
            conv1 = self._add_conv_layer(self.__ximage,1,5,5,[1,1,1,1],1,32,stddev = 0.1)
            norm1 = self._add_pool(conv1, 1, [1,2,2,1], [1,2,2,1])
            if self._vis_layer_num == 1:
                #self.store_param.append(max_index)
                pass
            conv2 = self._add_conv_layer(norm1,2,5,5,[1,1,1,1],32,64,stddev = 0.1)
            norm2= self._add_pool(conv2, 2, [1,2,2,1], [1,2,2,1])
            if self._vis_layer_num == 2:
                #self.store_param.append(max_index)
                pass
            vald_output = self._add_vald_layer(norm2, 64, 'vald')
            #fc = self._add_fclayer(vald_output, 1, 1024, 1024, stddev = 0.1)
            predict = self._output_layer(vald_output,64 * self._cluser_num,self._output_size, stddev = 0.1)
        with tf.name_scope('cnn_loss'):
            cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits = predict, labels = self._accurate_data))
            loss_scalar = tf.summary.scalar('cross_entropy', cross_entropy)
        with tf.name_scope('cnn_optimizer'):
            train_step = tf.train.AdamOptimizer(self._cnn_learning_rate).minimize(cross_entropy)
        with tf.name_scope('cnn_accuarcy'):
            correct_prediction = tf.equal(tf.argmax(predict, 1), tf.argmax(self._accurate_data, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
            accu_scalar = tf.summary.scalar('accuarcy', accuracy)

        return train_step, accuracy

    def __model_rnn(self):
        ###TODO
        #add some information to enhance the performance fo the malware flow detection
        with tf.name_scope('input'):
            #rnn input shape: [batch_size, time_sequence_num, embedding_size]
            self._x = tf.placeholder(tf.float32, [None, self.num_step, self.embedding_size], name = 'input')
        with tf.name_scope('output'):
            self._y = tf.placeholder(tf.float32, [None, self.class_num], name = 'output')

        with tf.name_scope('rnn_model'):
            input = tf.unstack(self._x, self.num_step, 1)
            linear_output = self._add_lstm_layer(input)
            lstm_output = tf.nn.softsign(linear_output)
            #double input feature number
            output = self._add_liner_layer(lstm_output, 2 * self.hidden_neural_size, self.class_num)
        with tf.name_scope('rnn_loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = output, labels = self._y))
            loss_scalar = tf.summary.scalar('cross_entropy', loss)
        with tf.name_scope('rnn_optimizer'):
            optimizer = tf.train.AdagradOptimizer(learning_rate = self._rnn_learning_rate).minimize(loss)
        with tf.name_scope('rnn_accuarcy'):
            correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(self._y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            accu_scalar = tf.summary.scalar('accuracy', accuracy)
        return optimizer, accuracy

    def train_cnn(self):
        """train the model described above
        """
        if self.is_gpu_available():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
            self.__sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        else:
            self.__sess = tf.Session()
        
        with tf.name_scope('input'):
            self._x = tf.placeholder(tf.float32,[None,self._input_size],name = 'input')
            edge_len = int(math.sqrt(self._input_size))
            assert edge_len * edge_len == self._input_size
            self.__ximage = tf.reshape(self._x, [-1, edge_len, edge_len, 1])

        with tf.name_scope('output'):
            self._accurate_data =  tf.placeholder(tf.float32, [None, 10], name = 'output')

        with tf.name_scope('cnn') as scope:
            train_step, accuracy= self.__model_cnn()

            init = tf.global_variables_initializer()
            self.__sess.run(init)
            self.__saver = tf.train.Saver(max_to_keep = 1)
            #merge the summary and write it to the tensorboard
            merge = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope))
            train_writer = tf.summary.FileWriter(self._log_dir + '/train', self.__sess.graph)
            #test_writer = tf.summary.FileWriter(self._log_dir + '/test',self.__sess.graph)

        for i in range(self._cnn_step):
            batch_xs, batch_ys = self.flow.train.next_batch(self._batch_size)
            self.__sess.run(train_step, feed_dict = {self._x : batch_xs, self._accurate_data : batch_ys, self._keep_prob : self._dropout})
            accuarcy = self.__sess.run(accuracy, feed_dict = {self._x : batch_xs, self._accurate_data : batch_ys, self._keep_prob : self._dropout})
            summary = self.__sess.run(merge, feed_dict = {self._x : batch_xs, self._accurate_data : batch_ys, self._keep_prob : 1})
            train_writer.add_summary(summary, i + 1)
            if i % 100 == 99:
                print('round {} accuarcy: {:.6f}'.format(i + 1, accuarcy))

        #store_param is [output_shape, weight, bias, max_index]
        #weight's shape: [conv_x,conv_y,input_feature_num,output_feature_num]
        if self._cnn_visualization == True:
            pass

        test_accu = self.__sess.run(accuracy, feed_dict = {self._x : self.flow.validation.images, self._accurate_data : self.flow.validation.labels, self._keep_prob : 1})
        print('validation : {:.6f}'.format(test_accu))
        self.__saver.save(self.__sess, './saver/model')

    def train_rnn(self):
        print('run lstm neural network')

        #self.__sess = tf.Session()

        #self.__saver = tf.train.import_meta_graph('./saver/model.meta')
        #self.__saver.restore(self.__sess, tf.train.latest_checkpoint('./saver/'))

        #graph = tf.get_default_graph()
        #self._x = graph.get_tensor_by_name('input/input:0')
        #input_feature = graph.get_tensor_by_name('cnn/cnn_model/vald/output:0')
        #input_feature = tf.stop_gradient(input_feature)
        if self.is_gpu_available():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
            self.__sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        else:
            self.__sess = tf.Session()

        feature, label = read_csv_dataset('train.csv', self.__sess, self._batch_size, self._rnn_step)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess = self.__sess, coord = coord)

        with tf.name_scope('rnn') as scope:
            train_step, accuracy= self.__model_rnn()

            init = tf.global_variables_initializer()
            self.__sess.run(init)

            #merge the rnn summary and write it to the tensorboard
            merge = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope))
            train_writer = tf.summary.FileWriter(self._log_dir + '/train',self.__sess.graph)
            #test_writer = tf.summary.FileWriter(self._log_dir + '/test',self.__sess.graph)

        try:
            i = 1;
            while not coord.should_stop():
                ##print parameters
                #batch_xs = self.__sess.run(tf.Print(input, [feature], summarize = 600))
                batch_xs, batch_ys = self.__sess.run([feature, label])
                batch_xs = batch_xs.reshape([-1, self.num_step, self.embedding_size])
                _, summary = self.__sess.run([train_step, merge], feed_dict = {self._x : batch_xs, self._y : batch_ys, self._keep_prob : self._dropout})
                train_writer.add_summary(summary, i)
                if i % 100 == 0:
                    accu = self.__sess.run(accuracy, feed_dict = {self._x : batch_xs, self._y : batch_ys, self._keep_prob : 1})
                    print('round {} accuarcy: {:.6f}'.format(i, accu))
                i += 1
                
            print(i)

            #test_accu = self.__sess.run(accuracy, feed_dict = {self._x : self.flow.validation.images, self._accurate_data : self.flow.validation.labels, self.keep_prob : 1})
            #print('validation : {:.6f}'.format(test_accu))
        except tf.errors.OutOfRangeError:
            print(i)
            print('Done Training')
        finally:
            coord.request_stop()
            coord.join(threads)

            ##visualize the weight param
            #graph = tf.get_default_graph()
            #weight = graph.get_tensor_by_name('bidirectional_rnn/fw/basic_lstm_cell/kernel:0')
            #w = self.__sess.run(weight)

            ##running the test
            print('begin valid')
            mm_input, mm_label = read_csv_test('valid_mm.csv', self.num_step)
            mm_input = mm_input.reshape([-1, self.num_step, self.embedding_size])
            benign_input, benign_label = read_csv_test('valid_benign.csv', self.num_step)
            benign_input = benign_input.reshape([-1, self.num_step, self.embedding_size])
            input, label = read_csv_test('valid.csv', self.num_step)
            input = input.reshape([-1, self.num_step, self.embedding_size])

            accu = self.__sess.run(accuracy,  feed_dict = {self._x : input, self._y : label, self._keep_prob : 1})
            mm_accu = self.__sess.run(accuracy,  feed_dict = {self._x : mm_input, self._y : mm_label, self._keep_prob : 1})
            benign_accu = self.__sess.run(accuracy,  feed_dict = {self._x : benign_input, self._y : benign_label, self._keep_prob : 1})
            print('Detection : {:.6f} , false: {:.6f}, accuarcy : {:.6f}'.format(mm_accu, 1-benign_accu, accu))
            
            print('begin test')
            input, label = read_csv_test('test.csv', self.num_step)
            input = input.reshape([-1, self.num_step, self.embedding_size])

            accu = self.__sess.run(accuracy,  feed_dict = {self._x : input, self._y : label, self._keep_prob : 1})
            print('Detection : {:.6f}'.format(accu))
            return 

    def feature_visualization(self, input_feature_num):
        """visualize the CNN feature extraction
        @param input : the input image 
        """ 
        print('feature visualization')
        start = input_feature_num
        end = input_feature_num + 1
        input = self.flow.train.images[start : end]
        #visualize input image
        fig, ax = plt.subplots(figsize = (2, 2))
        ax.imshow(np.reshape(input, (28,28)), cmap = plt.cm.gray)
        plt.show()

        #visualize the first convolution feature 
        graph = tf.get_default_graph()
        self._x = graph.get_tensor_by_name('input/input:0')
        feature = graph.get_tensor_by_name('cnn/cnn_model/conv_1/conv_1:0')
        conv_output = self.__sess.run(feature, feed_dict = {self._x : input})
        conv = self.__sess.run(tf.transpose(conv_output, [3, 0, 1, 2]))
        fig, ax = plt.subplots(ncols = 16, figsize = (16, 1))
        for i in range(16):
            ax[i].imshow(conv[i][0], cmap = plt.cm.gray)
        plt.title('conv')
        plt.show()

        #visualize the first pooling feature
        graph = tf.get_default_graph()
        self._x = graph.get_tensor_by_name('input/input:0')
        feature = graph.get_tensor_by_name('cnn/cnn_model/pool_1/pool:0')
        conv_output = self.__sess.run(feature, feed_dict = {self._x : input})
        conv = self.__sess.run(tf.transpose(conv_output, [3, 0, 1, 2]))
        fig, ax = plt.subplots(ncols = 16, figsize = (16, 1))
        for i in range(16):
            ax[i].imshow(conv[i][0], cmap = plt.cm.gray)
        plt.title('pool')
        plt.show()
        pass

    def deconvolution(self, image_index):
        """deconvolution network to extract the visualize feature
        """
        #model
        graph = tf.get_default_graph()
        start = image_index
        end = image_index + 1
        image = self.flow.train.images[start : end]
        #visualize input image
        fig, ax = plt.subplots(figsize = (2, 2))
        ax.imshow(np.reshape(image, (28,28)), cmap = plt.cm.gray) 
        plt.show()

        input = graph.get_tensor_by_name('input/input:0')
        output = graph.get_tensor_by_name('cnn/cnn_model/pool_{}/pool:0'.format(self._vis_layer_num))
        ###TODO
        #add selection to select the maximum activation feature to visualization the response convolution filter
        #image_feature = self.__sess.run(output, feed_dict = {input : image})
        #max = tf.arg_max(output, 1)
        #max = tf.arg_max(max,2)
        #np.insert

        while self._vis_layer_num > 0:
            max_index = self.store_param.pop()
            bias = self.store_param.pop()
            weight = self.store_param.pop()
            output_shape = self.store_param.pop()
            index = self.__sess.run(max_index, feed_dict = {input : image})
            unpool = self._add_unpool_layer(output, index)
            unbias = tf.subtract(unpool, bias)
            output = self._add_deconv_layer(unpool, weight, output_shape, stride = [1, 1, 1, 1])
            self._vis_layer_num -= 1

        #feature visualization
        output = self.__sess.run(output, feed_dict = {input : image})
        fig, ax = plt.subplots(figsize = (2, 2))
        ax.imshow(np.reshape(output, (28,28)), cmap = plt.cm.gray)
        plt.show()
        pass