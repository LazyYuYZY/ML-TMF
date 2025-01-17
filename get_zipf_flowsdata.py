import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zipfian
import os
import json
import random


class rw_files(object):
    @staticmethod
    def write_dict(txt_name, dicts):
        f = open(txt_name, 'w')
        json_dicts = json.dumps(dicts, indent=1)
        f.write(txt_name + '\n' + json_dicts)
        f.close()
    @staticmethod
    def get_dict(json_file_name):
        # 读取JSON文件并解析为字典对象
        with open(json_file_name, 'r') as json_file:
            lines = json_file.readlines()[1:]  # 读取所有行并从第二行开始
            json_data = json.loads("".join(lines))
        return json_data

    @staticmethod
    def change_name(folder_path, old_part, new_part):
        files = os.listdir(folder_path)  # 获取文件夹下的所有文件名

        for file in files:
            file_path = os.path.join(folder_path, file)  # 获取文件的完整路径
            tlist = file.split(old_part)
            new_file_name = tlist[0] + new_part
            new_file_path = os.path.join(folder_path, new_file_name)  # 构建新文件的完整路径
            # 移动文件并重命名
            os.rename(file_path, new_file_path)

if __name__ == '__main__':
    # 定义离散值范围和alpha值
    a = 1
    b = 10000
    # alpha_list = [1.2,1.4,1.5,1.6,1.8,2.0,2.5,3.0]
    alpha_list=[1.01,2.2,2.4]
    test_nums=10
    # 生成满足Zipf分布的数据集
    n_list = [10000,20000,40000,80000,120000,160000,200000,240000,320000,640000]

    for alpha in alpha_list:
        # 定义数值范围
        lower_bound = 100000
        upper_bound = 1000000

        # 生成n个不同的随机数
        n = 160000
        for i in range(test_nums):
            flows_index = np.random.choice(range(lower_bound, upper_bound + 1), n, replace=False)
            flows_index_str = flows_index.tolist()
            data = zipfian.rvs(alpha, b, size=n)
            data_str = data.tolist()
            # flows_dict = dict(zip(flows_index, data))
            flows_dict=dict(zip(flows_index_str,data_str))
            folder_path="./zipf_testdata_set/testdata_flows/"
            # 判断文件夹是否存在
            if not os.path.exists(folder_path):
                print("文件夹不存在")
                os.mkdir(folder_path)
            flow_name =  folder_path +str(int(alpha*10)) +"_"+str(n).zfill(7)+"_"+str(i).zfill(2)+".txt"
            rw_files.write_dict(flow_name,flows_dict)

            # 打印前10个数据
            print(data[:10])

            # 将数据分为第九类和第十类
            data[data >= 10] = 10
            print("small flows rate:",np.sum(data < 10)/data.shape[0])
            print("big flows rate:",np.sum(data == 10)/data.shape[0])

    # for n in n_list:
    #     # 定义数值范围
    #     lower_bound = 100000
    #     upper_bound = 1000000
    #
    #     # 生成n个不同的随机数
    #     alpha = 2
    #     for i in range(test_nums):
    #         flows_index = np.random.choice(range(lower_bound, upper_bound + 1), n, replace=False)
    #         flows_index_str = flows_index.tolist()
    #         data = zipfian.rvs(alpha, b, size=n)
    #         data_str = data.tolist()
    #         # flows_dict = dict(zip(flows_index, data))
    #         flows_dict = dict(zip(flows_index_str, data_str))
    #         folder_path = "./zipf_testdata_set/testdata_flows/"
    #         # 判断文件夹是否存在
    #         if not os.path.exists(folder_path):
    #             print("文件夹不存在")
    #             os.mkdir(folder_path)
    #         flow_name = folder_path + str(int(alpha * 10)) + "_" + str(n).zfill(7) + "_" + str(i).zfill(2) + ".txt"
    #         rw_files.write_dict(flow_name, flows_dict)
    #
    #         # 打印前10个数据
    #         print(data[:10])
    #
    #         # 将数据分为第九类和第十类
    #         data[data >= 10] = 10
    #         print("small flows rate:", np.sum(data < 10) / data.shape[0])
    #         print("big flows rate:", np.sum(data == 10) / data.shape[0])
    #
    # flows_folder="./testdata_set_5s/testdata_flows_5s"
    # # test_flows_path = flows_folder+"/20240103_5s_" + str(17).zfill(5) + ".txt"
    # test_flows_path="D:/大文件/LSTM-DNN/dataset/data_test/dataset_flows_5s/20231101_5s_00013.txt"
    # '''流数据集'''
    # test_flows_data = rw_files.get_dict(test_flows_path)
    # test_flows_data_list = list(test_flows_data.values())
    #
    # data = np.array(test_flows_data_list)
    #
    #
    #
    # data = zipfian.rvs(2, b, size=160000)
    # # 打印前10个数据
    # print(data[:10])
    #
    # # 将数据分为第九类和第十类
    # data[data >= 10] = 10
    # print("small flows rate:", np.sum(data < 10) / data.shape[0])
    # print("big flows rate:", np.sum(data == 10) / data.shape[0])
    # # 统计各类的频数
    # unique_val, counts = np.unique(data, return_counts=True)
    #
    # # 绘制条形图
    # plt.bar(range(len(counts)), counts)
    # plt.xlabel('class')
    # plt.ylabel('nums')
    # plt.xticks(range(len(counts)), range(1, len(counts) + 1))
    # plt.title('zipf')
    #
    # plt.show()
