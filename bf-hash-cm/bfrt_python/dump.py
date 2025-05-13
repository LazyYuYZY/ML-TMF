from ipaddress import ip_address

import sys
import time
import json
import signal
sys.path.append('/usr/local/lib/python3.5/dist-packages')
import redis
import os

p4 = bfrt.main_.pipe

interval = 60
base_output_dir = "/root/wjd/bf-hash-cm/bfrt_python/output_txt"
os.makedirs(base_output_dir, exist_ok=True)

def write_dict_to_txt(filename, data_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, filename), 'w') as f:
        for key, value in data_dict.items():
            f.write("{}:\n".format(key))
            for subkey, subvalue in value.items():
                f.write("  {}: {}\n".format(subkey, subvalue))
            f.write("\n")

round_id = 1

def pasermetadata(info,sketch):
    d1 = dict()
    for x in json.loads(info):
        index = x[u'key'][u'$REGISTER_INDEX']
        data = x[u'data'][sketch][1]
        d1[index] = data
    return d1

cm_sketch = p4.Ingress.cm_sketch
bloom_filter = p4.Ingress.bloom_filter
hash_table = p4.Ingress.hash_table


# Handle
count_dict_sketch = dict()
count_dict_bloom = dict()
count_dict_hash_src_addr = dict()
count_dict_hash_dst_addr = dict()
count_dict_hash_protocol = dict()
count_dict_hash_src_port = dict()
count_dict_hash_dst_port = dict()
count_dict_hash_table = dict()

next_time = time.time()

while True:
    start_time = time.time()

    count_dict_sketch.clear()
    count_dict_bloom.clear()
    count_dict_hash_src_addr.clear()
    count_dict_hash_dst_addr.clear()
    count_dict_hash_protocol.clear()
    count_dict_hash_src_port.clear()
    count_dict_hash_dst_port.clear()
    count_dict_hash_table.clear()

    # t = time.perf_counter()
    for table in hash_table.info(return_info=True, print_info=True):
        if table['type'] in ['REGISTER']:
            key1 = table['full_name'].split('.')[3]
            if key1.rstrip('0123456789') in ['hash_src_addr_']:
                info1 = table['node'].dump(json=True,return_ents=True,from_hw=True)
                value1 = pasermetadata(info1,'Ingress.hash_table.' + key1 +'.f1')
                count_dict_hash_src_addr[key1] = value1
                # print(count_dict_ingress)
    for table in hash_table.info(return_info=True, print_info=True):
        if table['type'] in ['REGISTER']:
            key1 = table['full_name'].split('.')[3]
            if key1.rstrip('0123456789') in ['hash_dst_addr_']:
                info1 = table['node'].dump(json=True,return_ents=True,from_hw=True)
                value1 = pasermetadata(info1,'Ingress.hash_table.' + key1 +'.f1')
                count_dict_hash_dst_addr[key1] = value1
                # print(count_dict_ingress)
    for table in hash_table.info(return_info=True, print_info=True):
        if table['type'] in ['REGISTER']:
            key1 = table['full_name'].split('.')[3]
            if key1.rstrip('0123456789') in ['hash_protocol_']:
                info1 = table['node'].dump(json=True,return_ents=True,from_hw=True)
                value1 = pasermetadata(info1,'Ingress.hash_table.' + key1 +'.f1')
                count_dict_hash_protocol[key1] = value1
                # print(count_dict_ingress)
    for table in hash_table.info(return_info=True, print_info=True):
        if table['type'] in ['REGISTER']:
            key1 = table['full_name'].split('.')[3]
            if key1.rstrip('0123456789') in ['hash_dst_addr_']:
                info1 = table['node'].dump(json=True,return_ents=True,from_hw=True)
                value1 = pasermetadata(info1,'Ingress.hash_table.' + key1 +'.f1')
                count_dict_hash_dst_addr[key1] = value1
                # print(count_dict_ingress)
    for table in hash_table.info(return_info=True, print_info=True):
        if table['type'] in ['REGISTER']:
            key1 = table['full_name'].split('.')[3]
            if key1.rstrip('0123456789') in ['hash_dst_port_']:
                info1 = table['node'].dump(json=True,return_ents=True,from_hw=True)
                value1 = pasermetadata(info1,'Ingress.hash_table.' + key1 +'.f1')
                count_dict_hash_dst_port[key1] = value1
                # print(count_dict_ingress)
    for table in hash_table.info(return_info=True, print_info=True):
        if table['type'] in ['REGISTER']:
            key1 = table['full_name'].split('.')[3]
            if key1 in ['hash_count']:
                info1 = table['node'].dump(json=True,return_ents=True,from_hw=True)
                value1 = pasermetadata(info1,'Ingress.hash_table.' + key1 +'.f1')
                count_dict_hash_table[key1] = value1
                # print(count_dict_ingress)

    for table in bloom_filter.info(return_info=True, print_info=True):
        if table['type'] in ['REGISTER']:
            key2 = table['full_name'].split('.')[3]
            if key2.rstrip('0123456789') in ['bf_count']:
                info2 = table['node'].dump(json=True,return_ents=True,from_hw=True)
                value2 = pasermetadata(info2,'Ingress.bloom_filter.' + key2 +'.f1')
                count_dict_bloom[key2] = value2
                # print(count_dict_egress)

    for table in cm_sketch.info(return_info=True, print_info=True):
        if table['type'] in ['REGISTER']:
            key1 = table['full_name'].split('.')[3]
            if key1.rstrip('0123456789') in ['sketch_count']:
                info1 = table['node'].dump(json=True,return_ents=True,from_hw=True)
                value1 = pasermetadata(info1,'Ingress.cm_sketch.' + key1 +'.f1')
                count_dict_sketch[key1] = value1
                # print(count_dict_ingress)


    # Clear Sketch
    for table in hash_table.info(return_info=True, print_info=False):
        if table['type'] in ['REGISTER']:
            table['node'].clear()

    for table in bloom_filter.info(return_info=True, print_info=False):
        if table['type'] in ['REGISTER']:
            table['node'].clear()

    for table in cm_sketch.info(return_info=True, print_info=False):
        if table['type'] in ['REGISTER']:
            table['node'].clear()

    round_output_dir = os.path.join(base_output_dir, "round_{:03d}".format(round_id))
    write_dict_to_txt("count_sketch.txt", count_dict_sketch, round_output_dir)
    write_dict_to_txt("count_bloom.txt", count_dict_bloom, round_output_dir)
    write_dict_to_txt("hash_src_addr.txt", count_dict_hash_src_addr, round_output_dir)
    write_dict_to_txt("hash_dst_addr.txt", count_dict_hash_dst_addr, round_output_dir)
    write_dict_to_txt("hash_protocol.txt", count_dict_hash_protocol, round_output_dir)
    write_dict_to_txt("hash_src_port.txt", count_dict_hash_src_port, round_output_dir)
    write_dict_to_txt("hash_dst_port.txt", count_dict_hash_dst_port, round_output_dir)
    write_dict_to_txt("hash_count_table.txt", count_dict_hash_table, round_output_dir)
    round_id += 1
    end_time = time.time()
    next_time += interval
    sleep_time = next_time - end_time
    if sleep_time > 0:
        time.sleep(sleep_time)
    else:
        print("本轮耗时超时 {:.2f}s，立即进入下一轮".format(end_time - start_time))
        next_time = time.time()