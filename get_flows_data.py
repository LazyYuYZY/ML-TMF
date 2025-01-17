from scapy.all import *
from rw_files import *
import numpy as np

T_slice=5

'''将大数据包按流切分'''
def flowProcess(p):
    packets = rdpcap(p)  # 提取pcap
    # print(packets)  # 打印pcap信息
    flow_list = []  # 用于五元组信息统计
    # 用于统计不要的报文数
    arp = 0
    other = 0
    wrong_data = 0

    # 获取包的时间
    rd_time = time.time()
    #print("rdpacp_time:", rd_time - start_time)

    '''提取五元组信息'''
    # wirelen获取数据包长度
    for data in packets:
        # data_byte=len(data)
        # data_byte=data.wirelen()
        # data.show()  #展示当前类型包含的属性及值
        # 协议：其中1，标识ICMP、2标识IGMP、6标识TCP、17标识UDP、89标识OSPF。
        # data.payload.name:'IP','IPV6','ARP'或者其他
        # data=data.payload#20230103日数据增加
        if data.payload.name == 'IP':
            try:
                five_tuple = "{}:{} {}:{} {}".format(data['IP'].src, data.sport, data['IP'].dst, data.dport, data.proto)
                flow_list.append(five_tuple)
            except AttributeError:
                wrong_data = wrong_data + 1
        elif data.payload.name == 'IPV6':
            try:
                five_tuple = "{}:{} {}:{} {}".format(data['IPV6'].src, data.sport, data['IPV6'].dst, data.dport,
                                                     data.proto)
                flow_list.append(five_tuple)
            except AttributeError:
                wrong_data = wrong_data + 1
        elif data.payload.name == 'ARP':
            arp = arp + 1
        else:
            other = other + 1

    get_t_flow(flow_list)
    # end for
    # print('arp:', arp, ' other:', other, ' 数据包损坏:', wrong_data)
    packet_time = time.time()
    #print("packet_time:", packet_time - rd_time)

    '''统计流的数目与五元组信息，并保存txt'''
    i = 0
    dicts = {}  # 用于统计每个五元组出现次数

    while 1:
        try:
            packet_now = flow_list.pop()
        except IndexError:
            break
        if packet_now in dicts:
            dicts[packet_now] = dicts[packet_now] + 1
        else:
            dicts[packet_now] = 1

    flow_time = time.time()
    print("flow_time:", flow_time - packet_time)
    fileCreate(dicts, p)  # 保存txt

'''将五元组信息统计到文件中'''
def fileCreate(dicts, p):
    # 拆分拼接字符串 创建txt文件
    tlist = p.split('.pcap')
    txt_name = tlist[0] + '.txt'
    # 5s
    dict_list = txt_name.split('/testdata_pcap')
    new_dir=dict_list[0] + "/testdata_flows_"+str(T_slice)+"s"
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    flow_name = dict_list[0] + "/testdata_flows" + dict_list[1]
    f = open(flow_name, 'w')
    json_dicts = json.dumps(dicts, indent=1)
    f.write("the number of flows:" + str(len(dicts)) + "\n" + json_dicts)
    f.close()



def get_t_flow(flow_list):
    # 获取流映射
    flow_index_path = "./testdata_set_5s/flow_index.txt"
    flows_alltime_dict = rw_files.get_dict(flow_index_path)
    t_flows=[]
    for one_packet in flow_list:
        t_flows.append(int(flows_alltime_dict[one_packet]))
    # # 时序相关：
    # base_n=10000
    # for i in range(int(len(flow_list)/base_n)):
    #     result_path="./test-bloom-register/"+str(i).zfill(5)+".text"
    #     np.savetxt(result_path, np.array(flow_list)[i*base_n:base_n*(i+1)])

    result_path = "./test-bloom-register.text"
    np.savetxt(result_path, np.array(t_flows), fmt='%d')

# flowProcess函数用于将pcap按流切分成小pcap
# packetRead函数用于读取pcap中所有数据的五元组信息，可用于切分后验证。
# fileCreate函数保存统计好的五元组信息

if __name__ == '__main__':

    for T_slice in [5]:
        pcap_Ts_num = int(900/T_slice)
        start_time=time.time()
        # 选择要切分的流的位置
        # for pcap_now in range(0,pcap_Ts_num):
        for pcap_now in range(0, 1):
            start_onetime = time.time()
            print(T_slice,pcap_now)
            folder_name="./testdata_set_"+str(T_slice)+"s/testdata_pcap_"+str(T_slice)+"s/"
            # folder_name = "./caida/testdata_pcap_" + str(T_slice) + "s/"
            p = folder_name + str(pcap_now).zfill(5) + ".pcap"
            flow_list = flowProcess(p)
            end_onetime= time.time()
            print("one process:",end_onetime-start_onetime)
        print("all process:", end_onetime - start_time)

