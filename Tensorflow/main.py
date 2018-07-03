#coding=utf-8
"""the experiment

    architecture:use Convolution Neural Network and Long Short Term Memory
    dataset: CIC-IDS-2017
    environment: mysql 5.7, python 3.5, tensorflow 1.8

    author:heyubo
    create: 2018/3/8
    last version: 2018/7/2
"""
from Model import mnist
from Model import CNN
from nerualnetwork import nn
import tensorflow as tf
import pcap_reader
import input_data
import configparser
import global_var
from input_data import read_csv
import os
import shutil
from prepossess import split_flow, heartbeat_filter, write_csv_file

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
    
    global_var.set_value('classnum', cf.get('dataset','class_num'))
    
    global_var.set_value('log_dir', cf.get('network','log_dir'))
    global_var.set_value('delta', cf.get('network','delta'))
    global_var.set_value('embedding_size', cf.get('network', 'embedding_size'))
    global_var.set_value('cluster_num', cf.get('network', 'cluster_num'))
    global_var.set_value('batch_size', cf.get('network', 'batch_size'))
    global_var.set_value('rnn_epoch', cf.get('network', 'rnn_epoch'))
    global_var.set_value('cnn_epoch', cf.get('network', 'cnn_epoch'))
    global_var.set_value('hidden_neural_size', cf.get('network', 'hidden_neural_size'))
    global_var.set_value('cnn_learning_rate', cf.get('network', 'cnn_learning_rate'))
    global_var.set_value('rnn_learning_rate', cf.get('network', 'rnn_learning_rate'))
    global_var.set_value('num_step', cf.get('network', 'num_step'))

def trans_img():
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
        image = pcap_reader.pcap2img(i, int(global_var.get_value('delta')))
        i += 1

    #transfer the file to the label
    label = pcap_reader.netflow2label()
    label.write_file()

def trans_csv(filecount):
    for i in range(filecount):
        filename = global_var.get_value('filepath') + 'endpoint\session-{}.pcap'.format(i)
        log_dir = global_var.get_value('log_dir')
        flow = split_flow(filename, 10, 1, 2)
        client_IP = ['192.168.10.{}'.format(i) for i in range(0, 256)]
        clean_flow = heartbeat_filter(flow, client_IP, 3, 0.2)
        write_csv_file(clean_flow, 'benign', 
                       global_var.get_value('filepath') + 'dataset\csv\session-{}.csv'.format(i))

def main():
    initialize()
    ##this work is to transfer the pcap file to the image file and dataset
    #trans_img()
    #rename()

    ##this work is to transfer the pcap file to the csv file
    #trans_csv(6217)

    ##Neural Network
    #if tf.gfile.Exists(log_dir):
    #    tf.gfile.DeleteRecursively(log_dir)
    #tf.gfile.MakeDirs(log_dir)
    exam = nn(int(global_var.get_value('image_weight')) * int(global_var.get_value('image_height')), 
              int(global_var.get_value('classnum')),
              cluster_num = int(global_var.get_value('cluster_num')), 
              hidden_neural_size = int(global_var.get_value('hidden_neural_size')), 
              embedding_size = int(global_var.get_value('embedding_size')), 
              num_step = int(global_var.get_value('num_step')), 
              cnn_step = int(global_var.get_value('cnn_epoch')), 
              rnn_step = int(global_var.get_value('rnn_epoch')), 
              cnn_learning_rate = float(global_var.get_value('cnn_learning_rate')), 
              rnn_learning_rate = float(global_var.get_value('cnn_learning_rate')), 
              batch_size = int(global_var.get_value('batch_size')))
    log_dir = global_var.get_value('log_dir')
    ##this work is to train cnn network and visualize the cnn feature detection
    #exam.set_log_dir(log_dir + '/cnn')
    #exam.set_cnn_visualization()
    #exam.train_cnn()
    #exam.feature_visualization(14)
    #exam.deconvolution(14)

    #this work is to train rnn network and visualize the rnn feature detection
    exam.set_log_dir(log_dir + '/rnn')
    exam.train_rnn()
    exam.close_sess()
    pass

if __name__ == '__main__':
    main()