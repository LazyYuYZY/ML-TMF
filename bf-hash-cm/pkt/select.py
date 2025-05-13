# -*- coding: utf-8 -*-
import json

packet_count = 0
selected_packets = []

# 打开JSON文件并逐行读取
with open('202401031400.json') as file:
    for line in file:
        # 读取每行数据并解析为JSON对象
        data = json.loads(line)

        # 获取五元组信息
        src_ip = data['SrcIP']
        dst_ip = data['DstIP']
        protocol = data['Protocol']
        src_port = data['SrcPort']
        dst_port = data['DstPort']

        # 构造五元组信息的字典
        packet = {
            'SrcIP': src_ip,
            'DstIP': dst_ip,
            'Protocol': protocol,
            'SrcPort': src_port,
            'DstPort': dst_port
        }

        # 判断是否达到了筛选的条数上限
        if len(selected_packets) >= 200000:
            break

        # 将数据包信息添加到已选数据包列表中
        selected_packets.append(packet)
        # data = {"packets": selected_packets}

        # 累计数据包计数
        packet_count += 1

with open("dat-test.json", "w") as file:
        for pkt in selected_packets:
           json.dump(pkt, file)
           file.write('\n')
# 输出筛选结果
# print("筛选出的数据包信息:")
# for packet in selected_packets:
#     print(packet)

# 输出数据包计数
print("总共读取了", packet_count, "个数据包")