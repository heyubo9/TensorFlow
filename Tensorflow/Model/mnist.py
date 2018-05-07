#coding=utf-8
import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf

class MNSIT_Train():
    """Convolution Neural Network class to train the MNIST dataset
using TensorBoard
    """
    def __init__(self,input_size,output_size,learining_rate = 0.001,step = 1000,batch_size = 100):
        """initilize the model parameter
        @param input_size   the input size of the CNN model
        @param output_size  the output size of the CNN model
        @param learning_rate the model learning rate to tarin the model,default = 0.001
        @param step         the step to train the model, default = 1000
        @param batch_size   batch the num of picture in one batch, default = 100
        """
        self.__x = tf.placeholder(tf.float32,[None,input_size])
        with tf.name_scope('weight'):
            self.__weight = tf.truncated_normal([input_size,output_size])
            self.__variable_summaries(self.__weight)
        with tf.name_scope('biases'):
            self.__bias = tf.Variable(tf.zeros([output_size]))
            self.__variable_summaries(self.__bias)
        self.__actual_data = tf.placeholder(tf.float32,[None,output_size])

        self.__learning_rate = learining_rate
        self.__step = 1000
        self.__batch_size = 100

        init = tf.global_variables_initializer()
        self.__sess = tf.Session()
        self.__sess.run(init)
        
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

    def input_data(self):
        """maybe useful for the private dataset
        not used
        """
        pass

    def set_log_dir(self,log_dir):
        """set the Tensorboard dictionary
        @param log_dir  Tensorboard dictionary  
        """
        self.__log_dir = log_dir

    def __model(self):
        """the simplest model to representt the conventional neural network
        contain no hidden layer
        probably not the same as the convolution nerual network
        """
        with tf.name_scope('prediction'):
            predict = tf.nn.softmax(tf.matmul(self.__x, self.__weight) + self.__bias)
            predict_hist = tf.summary.histogram('prediction', predict)
        with tf.name_scope('loss'):
            cross_entropy = tf.reduce_sum(self.__actual_data * tf.log(predict))
            scalar = loss = tf.summary.scalar('cross_entropy', cross_entropy)
        with tf.name_scope('optimizer'):
            train_step = tf.train.GradientDescentOptimizer(self.__learning_rate).minimize(cross_entropy)
        with tf.name_scope('accuarcy'):
            correct_prediction = tf.equal(tf.argmax(predict, 1), tf.argmax(self.__actual_data, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            accu_scalar = tf.summary.scalar('accuarcy', accuracy)

        return train_step, accuracy

    def train(self):
        """train the model
        a lot of things to change
        """
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        train_step, accuracy= self.__model()

        #merge the summary and write it to the tensorboard
        merge = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(self.__log_dir+'/train/log',self.__sess.graph)
        test_writer = tf.summary.FileWriter(self.__log_dir+'/test/log',self.__sess.graph)

        for i in range(self.__step):
            batch_xs,batch_ys = mnist.train.next_batch(self.__batch_size)
            summary,_ = self.__sess.run([merge,train_step],feed_dict={self.__x:batch_xs,self.__actual_data:batch_ys})
            train_writer.add_summary(summary, i)

        print(self.__sess.run(accuracy,feed_dict={self.__x:mnist.test.images,self.__actual_data:mnist.test.labels}))