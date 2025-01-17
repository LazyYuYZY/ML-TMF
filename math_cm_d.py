import random
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn as nn
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import os
import re
import sys
import time

'''CM Sketch类'''
class cm_sketch(object):
    def __init__(self, cm_d=3, cm_w=10 ** 5, flag=0, dict_cm={}, cm_sketch_load=np.full((3, 10 ** 5), 0)):
        self.flag = flag
        self.d = cm_d
        self.w = cm_w
        self.d_list = list(range(self.d))
        # 构造新的CM Sketch
        if self.flag == 0:
            """h(x) = (a*x + b) % p % w"""
            p_list = [6296197, 2254201, 7672057, 1002343, 9815713, 3436583, 4359587, 5638753, 8155451]
            selected_elements = random.sample(p_list, self.d)
            self.a = np.random.randint(low=10 ** 4, high=10 ** 5, size=(self.d, 1))
            self.b = np.random.randint(low=10 ** 4, high=10 ** 5, size=(self.d, 1))
            self.p = np.array(selected_elements).reshape(self.d, 1)
            self.offset = 10 ** 6 + 1
            self.Matrix = np.full((self.d, self.w), 0)  # cm存储的counter值
        else:  # 导入CM Sketch
            self.a = np.array(dict_cm['a'])
            self.b = np.array(dict_cm['b'])
            self.p = np.array(dict_cm['p'])
            self.offset = dict_cm['offset']
            self.Matrix = cm_sketch_load  # cm存储的counter值

    def insert_list(self, flow_list):

        for x, flow_num in enumerate(flow_list):
            x = x + self.offset
            h = (self.a * x + self.b) % self.p % self.w
            h_list = h.reshape(1, self.d).tolist()
            self.Matrix[self.d_list, h_list[0]] = self.Matrix[self.d_list, h_list[0]] + flow_num

    def insert_dict(self, flow_dict):

        for x, flow_num in flow_dict.items():
            x = x + self.offset
            h = (self.a * x + self.b) % self.p % self.w
            h_list = h.reshape(1, self.d).tolist()
            self.Matrix[self.d_list, h_list[0]] = self.Matrix[self.d_list, h_list[0]] + flow_num

    # 获取d个hash的查询值
    def query_d(self, five_turpe_list):
        x = np.array(five_turpe_list)
        x = x + self.offset
        x = x.reshape(1, -1)
        d_query_result = np.full((self.d, len(five_turpe_list)), 0)
        h = (self.a * x + self.b) % self.p % self.w
        for d in range(self.d):
            d_query_result[d] = self.Matrix[d][h[d]]

        return d_query_result

    def query_d_np(self, five_turpe_np):
        x = five_turpe_np
        x = x + self.offset
        x = x.reshape(1, -1)
        d_query_result = np.full((self.d, x.shape[1]), 0)
        h = (self.a * x + self.b) % self.p % self.w
        for d in range(self.d):
            d_query_result[d] = self.Matrix[d][h[d]]

        return d_query_result

    def query_one(self, five_tuple):
        x = five_tuple
        x = x + self.offset
        h = (self.a * x + self.b) % self.p % self.w
        h_list = h.reshape(1, self.d).tolist()
        d_query_result = self.Matrix[self.d_list, h_list[0]]
        oneflow_result = min(d_query_result)
        return oneflow_result

    # 所有流在cm的查询值 list实现
    def query_all_list(self, flow_list):
        flows_query_list = []
        for key in range(len(flow_list)):
            flows_query_list.append(self.query_one(key))
        return flows_query_list

    # 清空counter
    def clear(self):
        self.Matrix = np.full((self.d, self.w), 0)  # cm存储的counter值

    def save(self, file_name):
        dict_load = {'a': self.a.tolist(), 'b': self.b.tolist(), 'p': self.p.tolist(), 'offset': self.offset}
        rw_files.write_dict(file_name, dict_load)


class rw_files(object):
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

def conflict(index_np, dict_cm, w=22 * 10 ** 4, d=3, max_conf=1):
    a = np.array(dict_cm['a'])
    b = np.array(dict_cm['b'])
    p = np.array(dict_cm['p'])
    offset = dict_cm['offset']
    cm_d_id = (a * (index_np + offset) + b) % p % w
    data_x_d = np.full((index_np.shape[0], d * max_conf), -1)

    for i in range(d):
        unique_values, counts = np.unique(cm_d_id[i], return_counts=True)
        conf_items = unique_values[counts > 1]
        for conf_cm_id in conf_items:  # 冲突流在sketch中的索引
            conf_flows_id = np.where(np.isin(cm_d_id[i], conf_cm_id))[0]  # 单个conter冲突流在sketch的索引
            conf_flows_id_list = conf_flows_id.tolist()
            for conf_flow_id in conf_flows_id_list:
                k = 0
                for j in range(1 + max_conf):
                    if j >= len(conf_flows_id_list):
                        break
                    if conf_flows_id_list[j] != conf_flow_id:
                        data_x_d[conf_flow_id][i * max_conf + k] = index_np[conf_flows_id_list[j]]
                        k = k + 1
                        if k == max_conf:
                            break

    return data_x_d

if __name__ == '__main__':
    '''准备数据'''
    max_conf = 2
    param_type = 1
    param_dict = {1: "old", 0: "new"}

    file_now = random.randint(0, 180 - 1)
    # file_now = 179

    test_y_path = "D:/大文件/DNN_d_complete/testdata_set_5s/testdata_flows_5s/20240103_5s_" + str(file_now).zfill(
        5) + ".txt"

    # 处理cm数据
    flows_index_path = "D:/大文件/DNN_d_complete/testdata_set_5s/flow_index.txt"
    flows_alltime_dict = rw_files.get_dict(flows_index_path)
    file_name = "D:/大文件/DNN_d_complete/testdata_set_5s/" + param_dict[param_type] + "cm_sketch.txt"
    dict_cm = rw_files.get_dict(file_name)
    cm_d = 3
    if param_type==0:
        cm_w=dict_cm['w']
    else:
        cm_w=22*10**4

    cm_test_path = "D:/大文件/DNN_d_complete/testdata_set_5s/testdata_" + param_dict[
        param_type] + "cm_5s/20240103_5s_" + str(file_now).zfill(5) + ".txt"
    cm_sketch_now = np.loadtxt(cm_test_path)
    sketch_now = cm_sketch(cm_d=cm_d, cm_w=cm_w, flag=1, dict_cm=dict_cm, cm_sketch_load=cm_sketch_now)

    test_flows_data = rw_files.get_dict(test_y_path)
    test_flows_data_list = list(test_flows_data.values())

    get_value = np.vectorize(lambda x: flows_alltime_dict.get(x, 0))

    # 获取test时刻流信息
    test_x_key = list(test_flows_data.keys())
    test_index_array = get_value(np.array(test_x_key))
    test_x_ids = conflict(index_np=test_index_array, dict_cm=dict_cm,w=cm_w, max_conf=max_conf)

    '''获取所在时刻的d个查询值'''
    # 获取test
    test_x = np.full((test_x_ids.shape[0], cm_d * (1 + test_x_ids.shape[1])), 0)
    for i in range(cm_d * max_conf):
        conf_flows_id = np.where(test_x_ids[:, i] != -1)[0]
        test_x[conf_flows_id, i * cm_d:(i + 1) * cm_d] = sketch_now.query_d_np(test_x_ids[conf_flows_id, i]).T
    test_x[:, cm_d * test_x_ids.shape[1]:] = sketch_now.query_d_np(test_index_array).T

    '''test dataset'''
    test_cm_x = test_x[:,cm_d *cm_d * max_conf:]
    for i in range(cm_d):
        for j in range(max_conf):
            test_cm_x[:,i] =test_cm_x[:,i] -np.amin(test_x[:,(cm_d * (i*max_conf+j)):(cm_d * (i*max_conf+j+1))], axis=1)
    test_y = np.array(test_flows_data_list).reshape(-1, 1)
    test_cm = np.amin(sketch_now.query_d_np(test_index_array).T, axis=1).reshape(-1, 1)

    # 输出预测结果
    relative_math_error = np.amin(test_cm_x, axis=1).reshape(-1, 1) / test_y - 1
    # relative_pre_error = np.exp(pre_arr-test_y) - 1
    relative_cm_error = test_cm / test_y - 1
    print("relative cm error:" + "{:.6f}".format(np.mean(np.abs(relative_cm_error))))
    print("relative pre error:" + "{:.6f}".format(np.mean(np.abs(relative_math_error))))
    print("cm recall:" + "{:.6f}".format(np.sum(relative_cm_error == 0) / test_y.shape[0]))
    print("pre recall:" + "{:.6f}".format(np.sum(relative_math_error == 0) / test_y.shape[0]))

    cm_error = np.sum(np.abs(relative_cm_error) < 0.001)
    print("cm rate (error rate less than 0.1%):" + "{:.6f}".format(cm_error / test_y.shape[0]))
    pre_error = np.sum(np.abs(relative_math_error) < 0.001)
    print("pre rate (error rate less than 0.1%):" + "{:.6f}".format(pre_error / test_y.shape[0]))

    end_time = time.time()
