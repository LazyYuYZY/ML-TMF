import os
import sys
import json
import struct
import crcmod

# c4a45d1b 0x04C11DB7
# c46c3fad 0x741B8CD7
# 6e4a7b72 0xDB710641

crc32_func1 = crcmod.mkCrcFun(poly=0x104c11db7, rev=True, initCrc=0, xorOut=0xFFFFFFFF)
crc32_func2 = crcmod.mkCrcFun(poly=0x1741B8CD7, rev=True, initCrc=0, xorOut=0xFFFFFFFF)
crc32_func3 = crcmod.mkCrcFun(poly=0x1DB710641, rev=True, initCrc=0, xorOut=0xFFFFFFFF)

f = open("202401031400.json")
file_write_obj = open("202401031400-hash.json", 'w')
line = f.readline()
while line:
    s = json.loads(line)
    SrcIP = s["SrcIP"]
    DstIP = s["DstIP"]
    SrcPort = s["SrcPort"]
    DstPort = s["DstPort"]
    Protocol = s["Protocol"]
    print(SrcIP,DstIP,Protocol,SrcPort,DstPort)

    src_addr = ''.join('%02X' % int(i) for i in SrcIP.split('.'))
    dst_addr = ''.join('%02X' % int(i) for i in DstIP.split('.'))
    src_addr = int(src_addr,16)
    dst_addr = int(dst_addr,16)
    crc32_func1_exp_value = crc32_func1(struct.pack("!IIBHH", src_addr,dst_addr,17,SrcPort,DstPort)) & 0xffffffff
    crc32_func1_exp_value_hex = "{:08x}".format(crc32_func1_exp_value)
    # print(crc32_func1_exp_value_hex)

    crc32_func2_exp_value = crc32_func2(struct.pack("!IIBHH", src_addr,dst_addr,17,SrcPort,DstPort)) & 0xffffffff
    crc32_func2_exp_value_hex = "{:08x}".format(crc32_func2_exp_value)
    # print(crc32_func2_exp_value_hex)

    crc32_func3_exp_value = crc32_func3(struct.pack("!IIBHH", src_addr,dst_addr,17,SrcPort,DstPort)) & 0xffffffff
    crc32_func3_exp_value_hex = "{:08x}".format(crc32_func3_exp_value)
    # print(crc32_func3_exp_value_hex)

    dict_val = {'SrcIP':SrcIP,'DstIP':DstIP,'Protocol':Protocol,'SrcPort':SrcPort,'DstPort':DstPort,'HASH1':crc32_func1_exp_value_hex,'HASH2':crc32_func2_exp_value_hex,'HASH3':crc32_func3_exp_value_hex}
    json_val = json.dumps(dict_val)
    # print(json_val)
    file_write_obj.writelines(json_val)
    file_write_obj.write('\n')

    line = f.readline()
f.close()
file_write_obj.close()

# value1 = int(src_addr,16)
# value2 = int(dst_addr,16)
# value3 = int(protocol,16)
# value4 = int(src_port,16)
# value5 = int(dst_port,16)

# print(value1)
# print(value2)
# print(value3)
# print(value4)
# print(value5)
