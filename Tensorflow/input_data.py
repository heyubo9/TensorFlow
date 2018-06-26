# coding = utf-8
"""add the image,label to the dataset
last version: 2018/3/14
"""
import numpy
import os 
import tensorflow as tf
from PIL import Image
import global_var

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import gfile

def dense_to_one_hot(label_dense, num_class):
    """Convert class labels from scalars to one-hot vector"""
    num_label = label_dense.shape[0]
    index_offset = numpy.arange(num_label) * num_class
    label_one_hot = numpy.zeros([num_label, num_class])
    label_one_hot.flat[index_offset + label_dense.ravel()] = 1
    return label_one_hot

class DataSet(object):
    """Dataset class
    """
    def __init__(self, 
                 images,
                 labels,
                 dtype=dtypes.float32,
                 reshape=True):
        """Construct a DataSet.
        `dtype` can be either `uint8` to leave the input as `[0, 255]`, 
        or `float32` to rescale into `[0, 1]`.
        """
        #initilize the image weight and height
        image_weight = int(global_var.get_value('image_weight'))
        image_height = int(global_var.get_value('image_height'))

        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                            dtype)
        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        if reshape:
            assert images.shape[3] == 1
            images = images.reshape(images.shape[0],
                                    image_weight * image_height)
        if dtype == dtypes.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(numpy.float32)
            images = numpy.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle = True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start : self._num_examples]
            labels_rest_part = self._labels[start : self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start : end]
            labels_new_part = self._labels[start : end]
            # concatenate this epoch's rest image and next epoch's first image
            return numpy.concatenate((images_rest_part, images_new_part), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]

def read_data_sets(dtype = dtypes.float32, validation_size = 5000, reshape = True, seed = None):
    """read data
    last version: 2018/3/14
    """
    filepath = global_var.get_value('filepath')
    train_dir = filepath + "dataset\\"
    root = train_dir + "image\\"

    #set the image size
    image_weight = int(global_var.get_value('image_weight'))
    image_height = int(global_var.get_value('image_height'))
    
    #open the image file
    count = 0
    buf = []
    for root, dirs, files in os.walk(root):
        for file in files:
            #open the file and attach it to the train_images array
            with Image.open(root + file) as f:
                buf.append(numpy.array(f))
            count += 1
    #rewrite the read png file program
    train_images = numpy.array(buf)
    train_images = train_images.reshape([count, image_weight, image_height, 1])

    #open the test image
    #count = 0
    #buf = []
    #for root, dirs, files in os.walk(root + "test\\"):
    #    for file in files:
    #        #open the file and attach it to the test_images array
    #        with Image.open(root + file) as f:
    #            buf.append(numpy.array(f))
    #        count += 1
    #test_images = numpy.array(buf)
    #test_images.reshape(count, image_weight, image_height, 1)
    
    #open the label file
    count = 0
    buf = b""
    filename = train_dir + "label"
    with open(filename, "rb") as f:
        buf += f.read()
    train_labels = numpy.frombuffer(buf, dtype = numpy.uint8)
    train_labels = dense_to_one_hot(train_labels, int(global_var.get_value('classnum')))

    ##open the test label file
    #count = 0
    #buf = b""
    #filename = train_dir + "label"
    #with open(filename, "rb") as f:
    #    buf += f.read()
    #test_labels = numpy.frombuffer(buf, dtype = numpy.uint8)

    if not 0 <= validation_size <= len(train_images):
        raise ValueError("Validation size should be between 0 and {}. Received:{}".format(len(train_images), validation_size))

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    option = dict(dtype = dtype, reshape = reshape)

    train = DataSet(train_images, train_labels, **option)
    validation = DataSet(validation_images, validation_labels, **option)
    #test = DataSet(test_images, test_labels, **option)

    return base.Datasets(train = train, validation = validation, test = None)

def read_csv(session, batch_size, num_epochs):
    record_defaults = [[1], [1]]
    folder = []
    filename = global_var.get_value('filepath') + global_var.get_value('filename')
    folder.append(filename)

    filename_queue = tf.train.string_input_producer(folder, shuffle = False, num_epochs = num_epochs)
    reader = tf.TextLineReader(skip_header_lines = 1)
    key, value = reader.read(filename_queue)

    ##need to be update 
    textline = tf.decode_csv(value, record_defaults = record_defaults)
    x, y = tf.train.batch(textline, batch_size = batch_size, capacity = batch_size * 10)
    init = tf.local_variables_initializer()
    session.run(init)
    return x, y