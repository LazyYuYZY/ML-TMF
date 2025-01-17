import random

from rw_files import *

if __name__ == '__main__':
    # for T_slice in [5, 10, 15, 20, 25, 30]:
    for T_slice in [5]:
        pcap_num = int(900/T_slice)
        pcap_type = str(T_slice)+'s'
        # 获取所有流，将五元组映射为数字
        folder_path = "./testdata3_set_"+pcap_type+"/testdata_flows_"+pcap_type+"/"
        files = os.listdir(folder_path)  # 获取文件夹下的所有文件名
        flows_alltime = set()

        for file_now in range(10):
        # for file_now in range(pcap_num):
            # ".\testdata_set_5s\testdata_flows_5s\00000.txt"
            dataset_y_path = folder_path+ str(file_now).zfill(5) + ".txt"
            dict_now = rw_files.get_dict(dataset_y_path)
            flows_alltime.update(dict_now.keys())

        flows_alltime_dict = {}
        index = random.randint(10 ** 6, 10 ** 7)

        for one_flow in flows_alltime:
            flows_alltime_dict[one_flow] = index
            index = index + 1
        # 导出全局索引表
        rw_files.write_dict("./testdata3_set_"+pcap_type+"/flow_index.txt", flows_alltime_dict)
