import numpy as np
from scipy.sparse import lil_matrix
import random
import matplotlib.pyplot as plt
import json
import os
import re
import sys
import time
from sklearn.linear_model import OrthogonalMatchingPursuit
'''OMP算法步骤'''
# min |X|(L1)
# Y=Phi*X
# λ = arg max i=1,..,n|⟨r, ϕi⟩|：
# 在Φ中找到与每次迭代过程中的当前残差r具有最高相关性的列
# 通过在第3行中添加相应的列索引来更新活动列索引集S
# 求解最小二乘问题arg min x <$y −Φ_Sx <$2
# 计算残差r = y − Φ_S <$x。
# 当残差r小于τ时，我们停止该过程，其中τ是误差阈值。

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


def conflict_matrix(index_np, dict_cm, w=22 * 10 ** 4, d=3):
    a = np.array(dict_cm['a'])
    b = np.array(dict_cm['b'])
    p = np.array(dict_cm['p'])
    offset = dict_cm['offset']
    cm_d_id = (a * (index_np + offset) + b) % p % w

    # 定义稀疏矩阵的大小
    rows = w * d
    cols = index_np.shape[0]
    # 创建一个lil_matrix对象，并设置数据类型为bool
    vector = lil_matrix((rows, cols), dtype=np.bool_)

    # 设置观测矩阵
    for i in range(d):
        unique_values, counts = np.unique(cm_d_id[i], return_counts=True)
        for conf_cm_id in unique_values:  # 冲突流在sketch中的索引
            conf_flows_id = np.where(np.isin(cm_d_id[i], conf_cm_id))[0]  # 单个conter冲突流在sketch的索引
            row=i * w + conf_cm_id
            col_np=conf_flows_id
            for col in col_np:
                vector[row,col]=True
    return vector

def cs_omp(y,Phi,K):
    residual=y  #初始化残差
    (M,N) = Phi.shape
    index=np.zeros(N,dtype=int)
    for i in range(N): #第i列被选中就是1，未选中就是-1
        index[i]= -1
    result=np.zeros((N,1))
    start_time=time.time()
    step3_time=start_time

    for j in range(K):  #迭代次数
        product=np.fabs(np.dot(Phi.T,residual))
        step1_time = time.time()
        print("step1:{}s".format(step1_time - step3_time, step2_time - step1_time,
                                                                  step3_time - step2_time, step3_time - start_time))

        pos=np.argmax(product)  #最大投影系数对应的位置
        index[pos]=1 #对应的位置取1

        my=np.linalg.pinv(Phi[:,index>=0]) #广义逆矩阵（伪逆）(A^T*A)^(-1)*A^T，最小二乘法得到的解和伪逆矩阵是等价的。
        step2_time = time.time()
        print("step2:{}s".format(step2_time - step1_time))

        a=np.dot(my,y) #最小二乘
        residual=y-np.dot(Phi[:,index>=0],a)
        step3_time = time.time()
        print("step3:{}s,iters:{}s".format(step3_time-step2_time,step3_time-start_time))
    result[index>=0]=a
    Candidate = np.where(index>=0) #返回所有选中的列
    return result, Candidate

if __name__ == '__main__':
    # 获取流信息
    param_type = 0
    param_dict = {1: "old", 0: "new"}

    file_now = random.randint(0, 180 - 1)
    # file_now = 179
    test_y_path = "D:/大文件/DNN_d_complete/testdata_set_5s/testdata_flows_5s/20240103_5s_" + str(file_now).zfill(
        5) + ".txt"
    test_flows_data = rw_files.get_dict(test_y_path)
    test_flows_data_list = list(test_flows_data.values())

    flows_index_path = "D:/大文件/DNN_d_complete/testdata_set_5s/flow_index.txt"
    flows_alltime_dict = rw_files.get_dict(flows_index_path)

    get_value = np.vectorize(lambda x: flows_alltime_dict.get(x, 0))

    # 获取test时刻流信息
    test_x_key = list(test_flows_data.keys())
    test_index_array = get_value(np.array(test_x_key))
    test_flows_num=test_index_array.shape[0]

    # Sketch参数

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


    # 定义Y:所有非零counter值
    Y = sketch_now.Matrix.reshape(-1, 1)

    # 定义观测矩阵
    Phi=conflict_matrix(test_index_array,dict_cm,cm_w,cm_d)

    cs_omp(Y,Phi,100)

    OrthogonalMatchingPursuit(n_nonzero_coefs=test_flows_num,tol=1e-5).fit(Phi.toarray(),Y)

