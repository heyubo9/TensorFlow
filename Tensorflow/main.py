#coding=utf-8
"""the experiment

    architecture:use Convolution Neural Network and Long Short Term Memory
    dataset: CIC-IDS-2017
    environment: mysql 5.7, python 3.5, tensorflow 1.7

    author:heyubo
    create: 2018/3/8
    last version: 2018/5/7
"""
from Model import mnist
from Model import CNN
from nerualnetwork import nn
import tensorflow as tf
import pcap_reader
import input_data
import configparser
import global_var
import os
import shutil

def rename():
    filefold = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    filecount = [131676, 108189, 117665, 159616, 294130]
    count = 0
    i = 0
    while i < len(filefold):
        filepath = global_var.get_value('filepath') + filefold[i] + '\\dataset\\image\\'
        index = 0

        while index <= filecount[i]:
            filename = filepath + 'flow-' + str(index) + '-image.png'
            try:
                new_filename = global_var.get_value('filepath') + 'dataset\\image\\flow-' + str(count) + '-image.png'
                shutil.copyfile(filename, new_filename)
                index += 1
                count += 1
            except FileNotFoundError:
                print("%s not found" % filename)
                index += 1
                continue
        
        i += 1

def initialize():
    """initialize the global variable"""
    #initilize the global variable
    global_var.init()
    cf = configparser.ConfigParser()
    cf.read('main.conf')
    global_var.set_value('filename', cf.get('file', 'filename'))
    global_var.set_value('filepath', cf.get('file', 'filepath'))
    global_var.set_value('image_weight', cf.get('image', 'weight'))
    global_var.set_value('image_height', cf.get('image', 'height'))
    global_var.set_value('host', cf.get('mysql','host'))
    global_var.set_value('username', cf.get('mysql','username'))
    global_var.set_value('passwd', cf.get('mysql','passwd'))
    global_var.set_value('database', cf.get('mysql','database'))
    global_var.set_value('classnum', cf.get('network','class_num'))
    global_var.set_value('log_dir', cf.get('network','log_dir'))

def preprocess():
    """pre-process the pcap file and transfer to the image and label file"""
    #split flow
    pcapreader = pcap_reader.pcap_reader()
    pcapreader.flow_split()
    for i in range(pcapreader.get_count()):
        pcapreader.flow_statistic(i)
    del pcapreader

    #traverse the file in the file root ".\dataset\image":
    root = global_var.get_value('filepath') + "flow"
    i = 0
    for fn in os.listdir(root):
        image = pcap_reader.pcap2img(i)
        image.save_img()
        i += 1

    #transfer the file to the label
    label = pcap_reader.netflow2label()
    label.write_file()

def main():
    initialize()
    #preprocess()
    #rename()

    #Neural Network
    log_dir = global_var.get_value('log_dir')
    #if tf.gfile.Exists(log_dir):
    #    tf.gfile.DeleteRecursively(log_dir)
    #tf.gfile.MakeDirs(log_dir)
    #exam = mnist.MNSIT_Train(784,10)
    #exam.set_log_dir('./TensorBoard')
    #exam.train()
    exam = nn(28*28, int(global_var.get_value('classnum')),cluster_num = 128, hidden_neural_size = 128, num_step = 1, step = 5000, learning_rate = 0.002, batch_size = 100)
    exam.set_log_dir(log_dir + '/cnn')
    exam.train_cnn()
    exam.set_log_dir(log_dir + '/rnn')
    exam.train_rnn()

    #input the train dataset
    pass

if __name__ == '__main__':
    main()