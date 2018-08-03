#import sklearn
from assocrule import apriori, generateRules
from scapy.all import *
import csv

def flow_statistic(filename, srcIP, label, csvfile):
    #initilize the variable
    result = b""
    length = 0
    #delay_start = 0
    #direction = 0
    row = []
    head = ['packet1', 'packet2', 'packet3','packet4','packet5','packet6','packet7','packet8','packet9','packet10', 'label']
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
            
            if i > 10:
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
            if src in srcIP:
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
    rules = []
    for cluster in list:
        for packet in cluster:
            src = packet['IP'].src
            dst = packet['IP'].dst
                        
            if src in srcIP:
                direction = 1
            else:
                direction = -1

            item = direction * len(packet)
            fragmentation.append(item)

            if len(fragmentation) == 10:
                session.append(fragmentation)
                fragmentation = []
                break
        if len(fragmentation) != 0:
            session.append(fragmentation)
            fragmentation = []

    ###update data mining algorithm
    L, suppData = apriori(session, minSupport = confident)
    frozen_rule = generateRules(L, suppData, minConf = confident)
    #rule = assocrules.mine_assoc_rules(session, min_support = support, min_confidence = confident)
    for i in frozen_rule:
        rule = []
        for set in i:
            if not isinstance(set, float):
                for item in set:
                    rule.append(item)
        rules.append(rule)
    print(rules)
    for cluster in session:
        find = False
        for rule in rules:
            for i in range(0, len(cluster) - len(rule) + 1):
                if rule == cluster[i : i + len(rule)]:
                    start = i
                    end = i + len(rule)
                    for index in range(start, end):
                        del(cluster[start])
        if cluster:
            result.append(cluster)

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
            if packet is None:
                break

            if ti_1 == 0:
                ti_1 = packet.time
                continue
            interval = packet.time - ti_1
            max_t = interval if interval > max_t else max_t
            ti_1 = packet.time

        pcapreader.close()
        ti_1 = 0
        pcapreader = PcapReader(filename)

        i = 0
        j = 0
        k = 0
        prev_packet_seq = 0
        while True:
            packet = pcapreader.read_packet()
            if packet is None:
                break

            payload = packet['TCP'].payload
            ack = packet['TCP'].flags & 16 
            length = len(payload)
            if len(payload) == 0 and ack == 0:
                continue

            if ti_1 == 0 and t_last == 0:
                ti_1 = packet.time
                t_last = packet.time

            if packet['TCP'].seq == prev_packet_seq:
                ti_1 = packet.time
                continue

            if packet.time - ti_1 > response_delta:
                if packet.time - ti_1 > think_delta:
                    result.append(cluster)
                    t_last = packet.time
                    cluster = []
                    i += 1
                elif packet.time - t_last > max_t:
                    result.append(cluster)
                    t_last = packet.time
                    cluster = []
                    j += 1
                elif len(cluster) > 1:
                    if packet.time - ti_1 > SCALE * (ti_1 - t_last) / (len(cluster) - 1):
                        result.append(cluster)
                        t_last = packet.time
                        cluster = []
                        k += 1
            prev_packet_seq = packet['TCP'].seq
            cluster.append(packet)
            ti_1 = packet.time
        pcapreader.close()
        return result
    except Scapy_Exception as e:
        print(e)
    pass

def write_csv_file(list, label, filename):
    head = ['packet1', 'packet2', 'packet3','packet4','packet5','packet6','packet7','packet8','packet9','packet10', 'label']
    try:
        out = open(filename, 'a', newline = '')
        csv_writer = csv.writer(out, dialect = 'excel')
        #csv_writer.writerow(head)
        for split in list:
            while len(split) < 10:
                split.append(0)

            if label == 'benign':
                split.append(0)
            elif label == 'mm':
                split.append(1)

            csv_writer.writerow(split)

        out.close()
    except csv.Error as e:
        print(e)
        
    