# -*- coding: utf-8 -*-
"""transform the image, csv file to the image,label"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scapy.all import *
import time
import shutil
import socket
from struct import pack
from PIL import Image
import matplotlib.pyplot as plt
import six

import global_var
from mysql import mysql

class pcap2img:
    """transfer the pcap file to image
    first experiment: Session + application layer
    last version: 2018/3/20

    TODO:
    decide the layer info
    image size
    important information
    """
    _store_name = ""
    _image_raw = ""
    def __init__(self, index):
        """initilize the class
        
        last version: 2018/3/14
        first use thr application layer data to input to the neural network
        maybe change later to fix the accuration
        TODO:
        find someway to represent the flow data

        @param index: the index flow to transfer
        @param store_name: the store filename
        """
        #initilize the image size
        self._image_weight = int(global_var.get_value('image_weight'))
        self._image_height = int(global_var.get_value('image_height'))
        size = self._image_weight * self._image_height

        #initilize the variable
        starttime = 0
        endtime = 0
        payload = b""
        result = b""
        length = 0
        self._index = index
        try:
            #initilize the variable
            filepath = global_var.get_value("filepath")
            self._img_fold_root = filepath + "dataset\\image\\"
            pcapreader = PcapReader(filepath + "flow\\flow-" + str(index) + ".pcap")
            while True:
                packet = pcapreader.read_packet()
                if packet is None or length > 65535:
                    break

                src = socket.inet_aton(packet['IP'].src)
                dst = socket.inet_aton(packet['IP'].dst)
                proto = pack('h',packet['IP'].proto)
                sport = pack('i', socket.htons(packet['IP'].payload.fields['sport']))
                dport = pack('i', socket.htons(packet['IP'].payload.fields['dport']))
                length += len(packet)
                #get the application layer
                if len(packet['IP'].payload.payload) != 0 and len(payload) < 784:
                    payload += packet['IP'].payload.payload.original

                if starttime == 0:
                    starttime = packet.time
                endtime = packet.time

            duration = pack('f', endtime - starttime)
            length = pack('i', length)
            #five tuple plus session length plus start time plus duration plus all layer payload
            result = (proto + src + dst + sport + dport + length + duration + pack('f',starttime) + payload)[0 : size]
    
            #padding the zero byte
            if len(result) < size:
                result = result + bytes(size - len(result))
            self._image_raw = result
        except Scapy_Exception as e:
            print(e)


    def get_store_name(self):
        return self._store_name
    
    def get_image_raw(self):
        return self._image_raw

    def get_file_root(self):
        return self._img_fold_root

    def get_size(self):
        return self._size

    def set_store_name(self, filename):
        self._store_name = filename
        
    def set_size(self,size):
        self._size = size

    def save_img(self):
        """save the bytes array to the .png file
        last version: 2018/3/20
        """
        self.set_store_name(self.get_file_root() + "flow-" + str(self._index) + "-image.png")
        #write file
        image = Image.frombytes('L', (self._image_weight, self._image_height), self.get_image_raw())
        image.save(self._store_name)
        image.close()

class pcap_reader:
    """read the pcap file to the memory
    last version: 2018/3/20
    """
    def __init__(self, count = -1):
        """read the pcap file, split the network flow
        last version: 2018/3/8
        @param count: the packet read, default -1
        """
        try:
            filename = global_var.get_value("filename")
            filepath = global_var.get_value("filepath")

            self._pcapreader = PcapReader(filepath + filename)
            self._count = count
        except Scapy_Exception as e:
            print(e)
    
    def __del__(self):
        """close the pcapreader flag
        last version: 2018/3/20
        """
        self._pcapreader.close()

    def get_count(self):
        return self._count

    def set_count(self, count):
        self._count = count

    def flow_split(self):
        """split the network flow ,depending on quintuple
        last version: 2018/3/8
        """
        filepath = global_var.get_value("filepath")
        index = 0
        list = []
        type = -1
        count = self.get_count()
        #make the dictionary
        shutil.rmtree(filepath + "flow\\")
        os.makedirs(filepath + "flow\\")

        #read the packet
        ###TODO 
        #multithread
        while index < count or count < 0:
            index += 1
            packet = self._pcapreader.read_packet()
            if packet is None:
                break
            if packet.payload.name == 'IP':
                if packet['IP'].payload.name == 'TCP':
                    #put the upstream traffic and downlink traffic into one netflow
                    #last version: 2018/3/12
                    tuple = set([packet['IP'].src, packet['IP'].dst, packet['IP'].payload.name, packet['IP'].payload.fields['sport'], packet['IP'].payload.fields['dport']])
                    if tuple in list:
                        #last version:2018/3/8
                        class_index = list.index(tuple)
                        writer = PcapWriter(filepath + "flow\\flow-" + str(class_index) + ".pcap", append = True)
                        writer.write(packet)
                        writer.flush()
                        writer.close()
                    else:
                        #last version:2018/3/8
                        list.append(tuple)
                        type += 1
                        writer = PcapWriter(filepath + "flow\\flow-" + str(type) + ".pcap", append = False)
                        writer.write(packet)
                        writer.flush()
                        writer.close()
        if count < 0:
            self.set_count(type + 1)

    def flow_statistic(self, index):
        """flow statistic
        last version: 2018/3/12
        @param index: the num of flow  
        """

        filepath = global_var.get_value("filepath")
        #set the specific file
        pcapreader = PcapReader(filepath + "flow\\flow-" + str(index) + ".pcap")

        #initilize the data    
        x = []
        y = []
        char = []
        client_ip = 0
        begintime = 0
        direction = 1

        while True:
            packet = pcapreader.read_packet()
            if packet is None:
                break

            #set the packet's direction
            #last version: 2018/3/12
            if client_ip == 0:
                client_ip = packet['IP'].src
            if client_ip == packet['IP'].src:
                direction = 1
            else:
                direction = -1

            #timestamp statistic data
            if begintime == 0:
                begintime = packet.time
            delay = packet.time - begintime
            x.append(delay)
            y.append(direction * math.log(len(packet), math.e))

            #character count
            i = 0
            while i < len(packet):
                char.append(packet.original[i])
                i += 1

        #show the timestamp, char count, etc
        #timestamp : Line Chart
        #char count : Histogram
        #last update: 2018/3/12

        #draw the timestamp Line Chart
        plt.plot(x, y, marker = '*', label = 'timestamp-payload')
        plt.xlabel('timestamp')
        plt.ylabel('payload size')
        plt.title('timestamp payload relationship')
        plt.legend()
        plt.savefig(filepath + "figure\\flow-" + str(index) + "-timestamp.png", dpi = 300)
        plt.close()

        #draw the character Graph
        x_bar = range(255)
        plt.hist(char, x_bar, histtype = 'stepfilled')
        plt.xlabel('character')
        plt.ylabel('count')
        plt.title('character count relationship')
        plt.legend()
        plt.savefig(filepath + "figure\\flow-" + str(index) + "-character.png", dpi = 300)
        plt.close()
        
        pcapreader.close()

##TODO
#write the class to accomplish the construct label function
class netflow2label:
    """transfer the netflow dataset to the label set
    
    label count must be minor than 256
    last version: 2018/3/20
    """
    def __init__(self):
        self._db_operator = mysql()
        
    def write_file(self):
        """read the file in the root and write the label to the label file"""
        root = global_var.get_value('filepath') + "flow\\"
        
        #initialize the variable and environment
        if not self._db_operator.connect():
            return
        label_list = self._db_operator.find_type()

        #read the quintuple in one pcap
        buf = b""
        count = 0
        for fn in os.listdir(root):
            #for the num of file is equalitive to the maximum index of flow num
            #so we can use the i to traverse the root of the pcap file fold
            filename = root + "flow-" + str(count) + ".pcap"

            #if count == 2944 or count == 50725 or count == 58613 or count == 69930 or count == 71286 or count == 72349 or count == 72729 or count == 74396 or count == 75081 or count == 75820 or count == 77060 or count == 77976:
            #    count += 1
            #    continue

            try:
                pcapreader = PcapReader(filename)
            except FileNotFoundError:
                print("%s not found" % filename)
                count += 1
                continue
            packet = pcapreader.read_packet()

            quintuple = '%s-%s-%s-%s-%s' % (packet['IP'].dst, packet['IP'].src, packet['IP'].payload.fields['dport'], packet['IP'].payload.fields['sport'], packet['IP'].fields['proto'])
            label = self._db_operator.find_label(quintuple)
            pcapreader.close()
            
            #find error

            if label == "NO RESULT":
                quintuple = '%s-%s-%s-%s-%s' % (packet['IP'].src, packet['IP'].dst, packet['IP'].payload.fields['sport'], packet['IP'].payload.fields['dport'], packet['IP'].fields['proto'])
                label = self._db_operator.find_label(quintuple)
                if label == "NO RESULT":
                    png_file = global_var.get_value('filepath') + "dataset\\image\\flow-" + str(count) + "-image.png"
                    print("find the quintuple failed, count = %d : delete png file : %s" % (count, png_file))
                    #delete png file
                    try:
                        os.remove(png_file)
                    except FileNotFoundError:
                        print("file count:%d not found" % count)
                    count += 1
                    continue


            label_index = label_list.index(label)
            buf += label_index.to_bytes(1, byteorder = 'big')
            
            #write to the file
            count += 1
            if not count % 100:
                with open(global_var.get_value('filepath') + "dataset\\label", "ab") as f:
                    f.write(buf)
                    buf = b""

        with open(global_var.get_value('filepath') + "dataset\\label", "ab") as f:
            f.write(buf)

        self._db_operator.close()