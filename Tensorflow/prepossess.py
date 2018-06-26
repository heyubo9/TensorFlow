import sklearn
import pymining
from scapy.all import *
import csv

def flow_statistic(filename, srcIP, label, csvfile):
    #initilize the variable
    result = b""
    length = 0
    #delay_start = 0
    #direction = 0
    row = []
    head = ['packet1', 'packet2', 'packet3', 'label']
    try:
        #initilize the variable
        pcapreader = PcapReader(filename)
        i = 0
        while True:
            packet = pcapreader.read_packet()
            if packet is None or length > 65535:
                break
            
            src = socket.inet_aton(packet['IP'].src)
            dst = socket.inet_aton(packet['IP'].dst)
            
            if i > 3:
                ###spilt cluster
                if label == 'mm':
                    row.append(1)
                elif label == 'benign':
                    row.append(0)
                file = csv.writer(csvfile)
                file.writerow(row)
                row = []
                length = 0
                break

            ###for length appendment
            if srcIP == src:
                direction = 1
            else:
                direction = -1
            row.append(direction)
            #initialize
            i += 1
            length += len(packet)

            #get the application layer
            #if len(packet['IP'].payload.payload) != 0 and len(payload) < size:
            #    payload += packet['IP'].payload.payload.original
        
    except Scapy_Exception as e:
        print(e)

def heartbeat_filter(list, srcIP, support, confident):
    session = []
    fragmentation = []
    result = []
    for packet in list:

        src = socket.inet_aton(packet['IP'].src)
        dst = socket.inet_aton(packet['IP'].dst)
                        
        if srcIP == src:
            direction = 1
        else:
            direction = -1

        item = direction * len(packet)
        fragmentation.append(item)
        if len(fragmentation) > 3 or packet is None:
            session.append(fragmentation)
            fragmentation = []

    ###update data mining algorithm
    rule = pymining.assocrules.mine_assoc_rules(session, support, confident)
    print(result)
    for i in session:
        if i not in rule:
            result.append(i)

    return result
    pass

def split_flow(filename, think_delta, response_delta, SCALE):
    max_t = 0
    interval = 0
    ti_1 = 0
    t_last = 0
    cluster = []
    result = []
    try:
        #find max interval
        pcapreader = PcapReader(filename)
        while True:
            packet = pcapreader.read_packet()
            if ti_1 == 0:
                ti_1 = packet.time
                continue
            interval = packet.time - ti_1
            max_t = interval if interval > max_t else max_t
            ti_1 = packet.time

        pcapreader.close()
        ti_1 = 0
        pcapreader = PcapReader(filename)

        while True:
            packet = pcapreader.read_packet()
            if ti_1 == 0 and t_last == 0:
                ti_1 = packet.time
                t_last = packet.time
            if packet.time - ti_1 > response_delta:
                if packet.time - ti_1 > think_delta:
                    result.append(cluster)
                    t_last = packet.time
                elif packet.time - t_last > max_t:
                    result.append(cluster)
                    t_last = packet.time
                elif len(cluster) > 1:
                    if packet.time - ti_1 > SCALE * (ti_1 - t_last) / (len(cluster) - 1):
                        result.append(cluster)
                        t_last = packet.time
            cluster.append(packet)
        pcapreader.close()
        return result
    except Scapy_Exception as e:
        print(e)
    pass