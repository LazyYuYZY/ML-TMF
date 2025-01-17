import random
import numpy as np

from rw_files import *

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
            self.offset = int(dict_cm['offset'])
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
        for key in flow_list:
            flows_query_list.append(self.query_one(key))
        return flows_query_list

    # 清空counter
    def clear(self):
        self.Matrix = np.full((self.d, self.w), 0)  # cm存储的counter值

    def save(self, file_name):
        dict_load = {'a': self.a.tolist(), 'b': self.b.tolist(), 'p': self.p.tolist(), 'offset': self.offset,'d':self.d,'w':self.w}
        rw_files.write_dict(file_name, dict_load)


'''Tower Sketch类'''
class tower_sketch(object):
    def __init__(self, tower_d=3, tower_w=10 ** 5, flag=0, dict_tower={}, tower_sketch_load=np.full((3, 10 ** 5), 0)):
        self.flag = flag
        self.d = tower_d
        self.w = np.array([tower_w,tower_w*2,tower_w*4]).reshape(-1,1)
        self.mark=[2**31-2,2**16-2,2**8-2]
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
            self.Matrix = np.full((self.d, int(self.w[-1])), 0)  # tower存储的counter值
        else:  # 导入CM Sketch
            self.a = np.array(dict_tower['a'])
            self.b = np.array(dict_tower['b'])
            self.p = np.array(dict_tower['p'])
            self.offset = int(dict_tower['offset'])
            self.Matrix = tower_sketch_load  # tower存储的counter值

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
            d_query_result[d][d_query_result[d]>self.mark[d]]=2**31-2
        return d_query_result

    def query_d_np(self, five_turpe_np):
        x = five_turpe_np
        x = x + self.offset
        x = x.reshape(1, -1)
        d_query_result = np.full((self.d, x.shape[1]), 0)
        h = (self.a * x + self.b) % self.p % self.w
        for d in range(self.d):
            d_query_result[d] = self.Matrix[d][h[d]]
            d_query_result[d][d_query_result[d] > self.mark[d]] = 2 ** 31 - 2

        return d_query_result

    def query_one(self, five_tuple):
        x = five_tuple
        x = x + self.offset
        h = (self.a * x + self.b) % self.p % self.w
        h_list = h.reshape(1, self.d).tolist()
        d_query_result = self.Matrix[self.d_list, h_list[0]]
        for d in range(self.d):
            d_query_result[d][d_query_result[d] > self.mark[d]] = 2 ** 31 - 1
        oneflow_result = min(d_query_result)
        return oneflow_result

    # 所有流在tower的查询值 list实现
    def query_all_list(self, flow_list):
        flows_query_list = []
        for key in range(len(flow_list)):
            flows_query_list.append(self.query_one(key))
        return flows_query_list

    # 清空counter
    def clear(self):
        self.Matrix = np.full((self.d, int(self.w[-1])), 0)  # tower存储的counter值

    def save(self, file_name):
        dict_load = {'a': self.a.tolist(), 'b': self.b.tolist(), 'p': self.p.tolist(), 'offset': self.offset,'d':self.d,'w':int(self.w[0])}
        rw_files.write_dict(file_name, dict_load)


'''CU Sketch类'''
class cu_sketch(object):
    def __init__(self, cu_d=3, cu_w=10 ** 5, flag=0, dict_cu={}, cu_sketch_load=np.full((3, 10 ** 5), 0)):
        self.flag = flag
        self.d = cu_d
        self.w = cu_w
        self.d_list = list(range(self.d))
        # 构造新的CU Sketch
        if self.flag == 0:
            """h(x) = (a*x + b) % p % w"""
            p_list = [6296197, 2254201, 7672057, 1002343, 9815713, 3436583, 4359587, 5638753, 8155451]
            selected_elements = random.sample(p_list, self.d)
            self.a = np.random.randint(low=10 ** 4, high=10 ** 5, size=(self.d, 1))
            self.b = np.random.randint(low=10 ** 4, high=10 ** 5, size=(self.d, 1))
            self.p = np.array(selected_elements).reshape(self.d, 1)
            self.offset = 10 ** 6 + 1
            self.Matrix = np.full((self.d, self.w), 0)  # cu存储的counter值
        else:  # 导入CU Sketch
            self.a = np.array(dict_cu['a'])
            self.b = np.array(dict_cu['b'])
            self.p = np.array(dict_cu['p'])
            self.offset = dict_cu['offset']
            self.Matrix = cu_sketch_load  # cu存储的counter值

    def insert_list(self, flow_list):
        for x, flow_num in enumerate(flow_list):
            x = x + self.offset
            h = (self.a * x + self.b) % self.p % self.w
            h_list = h.reshape(1, self.d).tolist()
            vector = self.Matrix[self.d_list, h_list[0]]
            min_value = np.min(vector)
            mask = (vector == min_value)
            vector[mask] += flow_num
            self.Matrix[self.d_list, h_list[0]] = vector

    def insert_dict(self, flow_dict):
        for x, flow_num in flow_dict.items():
            x = x + self.offset
            h = (self.a * x + self.b) % self.p % self.w
            h_list = h.reshape(1, self.d).tolist()
            vector = self.Matrix[self.d_list, h_list[0]]
            min_value = np.min(vector)
            mask = (vector == min_value)
            vector[mask] += flow_num
            self.Matrix[self.d_list, h_list[0]] = vector

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

    # 所有流在cu的查询值 list实现
    def query_all_list(self, flow_list):
        flows_query_list = []
        for key in range(len(flow_list)):
            flows_query_list.append(self.query_one(key))
        return flows_query_list

    # 清空counter
    def clear(self):
        self.Matrix = np.full((self.d, self.w), 0)  # cu存储的counter值

    def save(self, file_name):
        dict_load = {'a': self.a.tolist(), 'b': self.b.tolist(), 'p': self.p.tolist(), 'offset': self.offset,
                     'd': self.d, 'w': self.w}
        rw_files.write_dict(file_name, dict_load)

'''Count Sketch类'''
class count_sketch(object):
    def __init__(self, count_d=3, count_w=10 ** 5, flag=0, dict_count={}, count_sketch_load=np.full((3, 10 ** 5), 0)):
        self.flag = flag
        self.d = count_d
        self.w = count_w
        self.d_list = list(range(self.d))
        # 构造新的Count Sketch
        if self.flag == 0:
            """h(x) = (a*x + b) % p % w"""
            p_list = [6296197, 2254201, 7672057, 1002343, 9815713, 3436583, 4359587, 5638753, 8155451]
            selected_elements = random.sample(p_list, self.d)
            self.a = np.random.randint(low=10 ** 4, high=10 ** 5, size=(self.d, 1))
            self.b = np.random.randint(low=10 ** 4, high=10 ** 5, size=(self.d, 1))
            self.p = np.array(selected_elements).reshape(self.d, 1)
            self.offset = 10 ** 6 + 1
            self.Matrix = np.full((self.d, self.w), 0)  # count存储的counter值
        else:  # 导入Count Sketch
            self.a = np.array(dict_count['a'])
            self.b = np.array(dict_count['b'])
            self.p = np.array(dict_count['p'])
            self.offset = dict_count['offset']
            self.Matrix = count_sketch_load  # count存储的counter值

    def insert_list(self, flow_list):

        for x, flow_num in enumerate(flow_list):
            if x % 2 == 0:
                g = 1
            else:
                g = -1
            x = x + self.offset
            h = (self.a * x + self.b) % self.p % self.w
            h_list = h.reshape(1, self.d).tolist()
            self.Matrix[self.d_list, h_list[0]] = self.Matrix[self.d_list, h_list[0]] + g * flow_num

    def insert_dict(self, flow_dict):

        for x, flow_num in flow_dict.items():
            if x % 2 == 0:
                g = 1
            else:
                g = -1
            x = x + self.offset
            h = (self.a * x + self.b) % self.p % self.w
            h_list = h.reshape(1, self.d).tolist()
            self.Matrix[self.d_list, h_list[0]] = self.Matrix[self.d_list, h_list[0]] + g * flow_num

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

    # 所有流在count的查询值 list实现
    def query_all_list(self, flow_list):
        flows_query_list = []
        for key in range(len(flow_list)):
            flows_query_list.append(self.query_one(key))
        return flows_query_list

    # 清空counter
    def clear(self):
        self.Matrix = np.full((self.d, self.w), 0)  # count存储的counter值

    def save(self, file_name):
        dict_load = {'a': self.a.tolist(), 'b': self.b.tolist(), 'p': self.p.tolist(), 'offset': self.offset,
                     'd': self.d, 'w': self.w}
        rw_files.write_dict(file_name, dict_load)

'''Bloom类'''
class bloom_sketch(object):
    def __init__(self, bloom_d=3, bloom_w=10 ** 5, flag=0, dict_bloom={}, bloom_sketch_load=np.full((3, 10 ** 5), 0)):
        self.flag = flag
        self.d = bloom_d
        self.w = bloom_w
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
            self.Matrix = np.full((self.d, self.w), 0)  # bloom存储的counter值
        else:  # 导入CM Sketch
            self.a = np.array(dict_bloom['a'])
            self.b = np.array(dict_bloom['b'])
            self.p = np.array(dict_bloom['p'])
            self.offset = int(dict_bloom['offset'])
            self.Matrix = bloom_sketch_load  # bloom存储的counter值

    def insert_one(self,x):

        x = x + self.offset
        h = (self.a * x + self.b) % self.p % self.w
        h_list = h.reshape(1, self.d).tolist()
        self.Matrix[self.d_list, h_list[0]] = self.Matrix[self.d_list, h_list[0]] + 1


    def insert_list(self, flow_list):

        for x in flow_list:
            x = x + self.offset
            h = (self.a * x + self.b) % self.p % self.w
            h_list = h.reshape(1, self.d).tolist()
            self.Matrix[self.d_list, h_list[0]] = self.Matrix[self.d_list, h_list[0]] + 1

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

    # 所有流在bloom的查询值 list实现
    def query_all_list(self, flow_list):
        flows_query_list = []
        for key in flow_list:
            flows_query_list.append(self.query_one(key))
        return flows_query_list

    # 清空counter
    def clear(self):
        self.Matrix = np.full((self.d, self.w), 0)  # bloom存储的counter值

    def save(self, file_name):
        dict_load = {'a': self.a.tolist(), 'b': self.b.tolist(), 'p': self.p.tolist(), 'offset': self.offset,'d':self.d,'w':self.w}
        rw_files.write_dict(file_name, dict_load)
