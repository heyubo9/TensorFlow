#!/usr/bin/python
#?-*-?coding:utf-8?-*-
from scapy.all import *
import socket
from struct import pack
from PIL import Image
import math

class Image_Reader:
    _size = 784
    def __init__(self,filename,store_name):
        try:
            #create string
            pkts = rdpcap(filename)
            length = 0
            str = b""
            starttime = 0
            endtime = 0
            src_byte = b""
            dst_byte = b""
            sport_byte = b""
            dport_byte = b""
            duration = 0
            proto = 0
            for pktno in range(len(pkts)):
                print('in there')
        
                #add payload
                length = length + len(pkts.res[pktno].original)
                str = str + pkts.res[pktno].original
        
                #get ip information
                ip = pkts.res[pktno].payload
                proto = pack('h',ip.fields['proto'])
                src = ip.fields['src']
                src_byte = socket.inet_aton(src)
                dst = ip.fields['dst']
                dst_byte = socket.inet_aton(dst)

                #get tcp information
                tcp = ip.payload
                sport = socket.htons(tcp.fields['sport'])
                sport_byte = pack('h',sport)
                dport = socket.htons(tcp.fields['dport'])
                dport_byte = pack('h',dport)
                if pktno == 0:
                    starttime = pkts.res[pktno].time
                endtime = pkts.res[pktno].time
            duration = endtime - starttime
            dur_byte = pack('f',duration)
            result = (proto+src_byte+sport_byte+dst_byte+dport_byte+pack('i',length)+dur_byte+pack('f',starttime)+str)[0:784]
    
            #judge size
            if length < size:
                result = result + bytes(size-length)

            #write file
            image = Image.frombytes('L',(int(math.sqrt(size)),int(math.sqrt(size))),result)
            image.save(store_name)
        except Scapy_Exception as e:
            print(e)

    def set_size(self,size):
        self._size = size

    def get_size(self):
        return self._size