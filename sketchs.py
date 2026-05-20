import random
import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit  # 可选，若没有则使用自己实现的OMP
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

    def insert_one(self, x):
        x = x + self.offset
        h = (self.a * x + self.b) % self.p % self.w
        h_list = h.reshape(1, self.d).tolist()
        self.Matrix[self.d_list, h_list[0]] = self.Matrix[self.d_list, h_list[0]] + 1

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



class ElasticSketch:
    """
    Elastic Sketch 的基本实现，包含重哈希表（heavy）和轻量级 CM Sketch（light）。
    插入使用 Ostracism 策略：每个桶只存储一个流，当冲突达到阈值时驱逐原流。
    查询时根据重哈希表的结果与轻量级部分合并返回。
    """
    def __init__(self, heavy_buckets=10000, lambda_threshold=8,
                 light_d=1, light_w=100000, flag=0, dict_elastic={},
                 heavy_load=None, light_load=None):
        """
        初始化 Elastic Sketch。
        :param heavy_buckets: 重哈希表的桶数量
        :param lambda_threshold: Ostracism 阈值 λ
        :param light_d: CM Sketch 的深度（哈希函数个数）
        :param light_w: CM Sketch 的宽度（每行的计数器数量）
        :param flag: 0=新建，1=从字典加载
        :param dict_elastic: 加载时使用的参数字典
        :param heavy_load: 加载时重哈希表的数据（列表，每个元素为 None 或 [key, pos, neg, flag]）
        :param light_load: 加载时轻量级 CM Sketch 的矩阵数据（numpy 数组）
        """
        self.flag = flag
        self.heavy_buckets = heavy_buckets
        self.lambda_threshold = lambda_threshold

        if self.flag == 0:
            # 初始化重哈希表：每个桶存储 None 或 [flow_id, positive_votes, negative_votes, flag]
            self.heavy = [None] * self.heavy_buckets

            # 初始化轻量级 CM Sketch
            self.light = cm_sketch(cm_d=light_d, cm_w=light_w, flag=0)

            # 为轻量级 CM Sketch 添加 insert_one 方法（如果不存在）
            if not hasattr(self.light, 'insert_one'):
                def insert_one(self_light, flow_id):
                    # 使用已有的 insert_list 方法插入单个流
                    self_light.insert_list([flow_id])
                self.light.insert_one = insert_one.__get__(self.light, cm_sketch)

            # 保存哈希参数（用于重哈希表索引计算，使用 light 的第一个哈希函数）
            self.hash_params = {
                'a': self.light.a[0][0],
                'b': self.light.b[0][0],
                'p': self.light.p[0][0],
                'offset': self.light.offset
            }
        else:
            # 从加载数据恢复
            self.heavy = heavy_load
            self.light = cm_sketch(cm_d=dict_elastic['light_d'],
                                   cm_w=dict_elastic['light_w'],
                                   flag=1,
                                   dict_cm=dict_elastic['light_dict'],
                                   cm_sketch_load=light_load)
            # 确保 light 有 insert_one 方法
            if not hasattr(self.light, 'insert_one'):
                def insert_one(self_light, flow_id):
                    self_light.insert_list([flow_id])
                self.light.insert_one = insert_one.__get__(self.light, cm_sketch)

            self.hash_params = dict_elastic['hash_params']
            self.lambda_threshold = dict_elastic.get('lambda_threshold', 8)

    def _hash(self, flow_id):
        """
        计算重哈希表的桶索引。
        使用与轻量级相同的哈希函数（取第一组参数），并模 heavy_buckets。
        """
        x = flow_id + self.hash_params['offset']
        a = self.hash_params['a']
        b = self.hash_params['b']
        p = self.hash_params['p']
        h = (a * x + b) % p % self.heavy_buckets
        return h

    def _insert_into_light(self, flow_id, delta):
        """
        向轻量级 CM Sketch 中插入指定增量。
        由于 cm_sketch 没有直接支持增量更新，这里循环调用 insert_one。
        （性能优化可后续在 cm_sketch 中添加带增量的方法）
        """
        for _ in range(delta):
            self.light.insert_one(flow_id)

    def insert_one(self, flow_id, count=1):
        """
        插入一个包（或批量包）到 sketch。
        :param flow_id: 流标识（整数）
        :param count: 包数量（默认1）
        """
        for _ in range(count):
            self._insert_single_packet(flow_id)

    def _insert_single_packet(self, flow_id):
        """
        单包插入的核心逻辑（Ostracism 策略）。
        """
        idx = self._hash(flow_id)
        bucket = self.heavy[idx]

        if bucket is None:   # 空桶，直接插入
            self.heavy[idx] = [flow_id, 1, 0, False]
            return

        # 桶非空
        key, pos, neg, flag = bucket
        if key == flow_id:
            # 同一流，增加正投票
            bucket[1] += 1
        else:
            # 不同流，增加负投票
            bucket[2] += 1
            # 判断是否满足驱逐条件：neg >= lambda * pos
            if bucket[2] >= self.lambda_threshold * bucket[1]:
                # 驱逐当前流，插入新流
                evicted_key, evicted_pos, _, _ = bucket
                # 新流：pos=1, neg=1, flag=True（因为部分可能已记录在 light）
                self.heavy[idx] = [flow_id, 1, 1, True]
                # 将被驱逐的流及其计数插入 light
                self._insert_into_light(evicted_key, evicted_pos)
            else:
                # 不驱逐，当前流进入 light
                self._insert_into_light(flow_id, 1)

    def insert_list(self, flow_list):
        """批量插入，flow_list 为流标识列表，每个元素插入一个包。"""
        for fid in flow_list:
            self.insert_one(fid)

    def insert_dict(self, flow_dict):
        """批量插入，flow_dict 为 {流标识: 包数}。"""
        for fid, cnt in flow_dict.items():
            self.insert_one(fid, cnt)

    def query_one(self, flow_id):
        """
        查询指定流的大小估计。
        :param flow_id: 流标识
        :return: 估计的包数
        """
        idx = self._hash(flow_id)
        bucket = self.heavy[idx]
        if bucket is not None and bucket[0] == flow_id:
            key, pos, neg, flag = bucket
            if flag:
                # 需要加上 light 中的部分
                light_val = self.light.query_one(flow_id)
                return pos + light_val
            else:
                return pos
        else:
            return self.light.query_one(flow_id)

    def query_list(self, flow_list):
        """批量查询，返回对应估计值列表。"""
        return [self.query_one(fid) for fid in flow_list]

    def query_dict(self, flow_dict):
        """批量查询，返回字典 {流标识: 估计值}。"""
        return {fid: self.query_one(fid) for fid in flow_dict}

    def clear(self):
        """清空所有数据（重置重哈希表和轻量级 CM Sketch）。"""
        self.heavy = [None] * self.heavy_buckets
        self.light.clear()

    def save(self, file_name):
        """
        保存 sketch 状态到文件（需配合 rw_files 模块使用）。
        保存内容：重哈希表数据、轻量级参数及矩阵。
        """
        # 准备重哈希表数据（可序列化形式）
        heavy_data = []
        for bucket in self.heavy:
            if bucket is None:
                heavy_data.append(None)
            else:
                heavy_data.append(bucket[:])  # 复制列表

        # 保存轻量级 CM Sketch 的参数字典
        light_dict = {
            'a': self.light.a.tolist(),
            'b': self.light.b.tolist(),
            'p': self.light.p.tolist(),
            'offset': self.light.offset,
        }
        dict_elastic = {
            'heavy_buckets': self.heavy_buckets,
            'lambda_threshold': self.lambda_threshold,
            'light_d': self.light.d,
            'light_w': self.light.w,
            'light_dict': light_dict,
            'hash_params': self.hash_params,
        }
        # 将矩阵转为列表存储
        light_data = self.light.Matrix.tolist()
        data = {
            'dict_elastic': dict_elastic,
            'heavy_data': heavy_data,
            'light_data': light_data,
        }
        rw_files.write_dict(file_name, data)

    @classmethod
    def load(cls, file_name):
        """从文件加载 ElasticSketch 实例。"""
        data = rw_files.get_dict(file_name)
        dict_elastic = data['dict_elastic']
        heavy_data = data['heavy_data']
        light_data = np.array(data['light_data'])

        # 重建轻量级 CM Sketch
        light = cm_sketch(cm_d=dict_elastic['light_d'],
                          cm_w=dict_elastic['light_w'],
                          flag=1,
                          dict_cm=dict_elastic['light_dict'],
                          cm_sketch_load=light_data)
        # 为 light 添加 insert_one 方法
        if not hasattr(light, 'insert_one'):
            def insert_one(self_light, flow_id):
                self_light.insert_list([flow_id])
            light.insert_one = insert_one.__get__(light, cm_sketch)

        # 重建重哈希表
        heavy = []
        for item in heavy_data:
            if item is None:
                heavy.append(None)
            else:
                heavy.append(item[:])  # 复制列表

        # 创建实例
        instance = cls(heavy_buckets=dict_elastic['heavy_buckets'],
                       lambda_threshold=dict_elastic['lambda_threshold'],
                       light_d=dict_elastic['light_d'],
                       light_w=dict_elastic['light_w'],
                       flag=1,
                       dict_elastic=dict_elastic,
                       heavy_load=heavy,
                       light_load=light_data)
        instance.heavy = heavy
        instance.light = light
        instance.hash_params = dict_elastic['hash_params']
        return instance


class fcm_sketch(object):
    """
    FCM-Sketch: Feed-forward Count-Min Sketch
    参数:
        num_trees: 独立树的数量 (d)
        num_stages: 层级数 (L)
        branch_factor: 分支因子 (k)
        bits_per_stage: 每层位宽元组，长度必须等于 num_stages
        base_w: 第一层(叶子层)宽度，后续层宽度 = base_w // (branch_factor ** stage)
        flag: 0 新建, 1 从文件加载
        dict_fcm: 加载时传入的参数字典
        fcm_sketch_load: 加载时传入的矩阵数据 (形状: num_trees, 各层宽度总和?)
                         为简化存储，我们将每层的矩阵拼接成一维存储，加载时再拆分。
    """

    def __init__(self,
                 num_trees=3,
                 num_stages=3,
                 branch_factor=8,
                 bits_per_stage=(8, 16, 32),
                 base_w=10**5,
                 flag=0,
                 dict_fcm=None,
                 fcm_sketch_load=None):

        self.flag = flag
        self.num_trees = num_trees
        self.num_stages = num_stages
        self.branch_factor = branch_factor
        self.bits_per_stage = bits_per_stage
        self.base_w = base_w

        # 计算每层最大计数值与溢出标记
        self.max_counts = [(1 << b) - 2 for b in bits_per_stage]      # 可正常计数的上限
        self.overflow_marks = [(1 << b) - 1 for b in bits_per_stage]  # 表示溢出的特殊值

        # 计算各层宽度
        self.widths = []
        for s in range(num_stages):
            w = base_w // (branch_factor ** s)
            if w < 1:
                w = 1  # 防止宽度为0
            self.widths.append(w)

        self.total_counters_per_tree = sum(self.widths)
        self.stage_offsets = [0]
        for s in range(num_stages - 1):
            self.stage_offsets.append(self.stage_offsets[-1] + self.widths[s])

        if self.flag == 0:
            # 初始化哈希函数参数 (每个树、每层都需要独立的哈希函数？)
            # 论文中每层通常共享同一组哈希函数(针对流ID)，但索引映射到不同宽度的数组时需要取模。
            # 为简化，我们为每个树生成一组哈希参数，用于所有层，查询时再根据层宽度取模。
            p_list = [6296197, 2254201, 7672057, 1002343, 9815713,
                      3436583, 4359587, 5638753, 8155451]
            self.hash_a = np.random.randint(low=10**4, high=10**5, size=(num_trees, 1))
            self.hash_b = np.random.randint(low=10**4, high=10**5, size=(num_trees, 1))
            selected = random.sample(p_list, num_trees)
            self.hash_p = np.array(selected).reshape(num_trees, 1)
            self.offset = 10**6 + 1

            # 初始化计数器矩阵: shape = (num_trees, total_counters_per_tree)
            self.Matrix = np.zeros((num_trees, self.total_counters_per_tree), dtype=np.uint32)
        else:
            # 从字典加载参数
            self.hash_a = np.array(dict_fcm['a'])
            self.hash_b = np.array(dict_fcm['b'])
            self.hash_p = np.array(dict_fcm['p'])
            self.offset = int(dict_fcm['offset'])
            self.Matrix = fcm_sketch_load

    def _hash_indices(self, flow_id):
        """
        计算流在每棵树、每层中的索引。
        返回: indices[tree][stage] = 在该层数组中的局部索引
        """
        x = flow_id + self.offset
        # 每棵树的基础哈希值
        h_vals = (self.hash_a * x + self.hash_b) % self.hash_p   # shape: (num_trees, 1)
        indices = []
        for t in range(self.num_trees):
            tree_idx = []
            h = h_vals[t, 0]
            for s in range(self.num_stages):
                w = self.widths[s]
                # 对每层宽度取模，为了更好的随机性，可以结合层级信息，这里直接用基础哈希取模
                stage_idx = h % w
                tree_idx.append(stage_idx)
            indices.append(tree_idx)
        return indices

    def _get_counter(self, tree, stage, local_idx):
        """获取指定树、指定层、局部索引的计数器值"""
        offset = self.stage_offsets[stage]
        return self.Matrix[tree, offset + local_idx]

    def _set_counter(self, tree, stage, local_idx, value):
        """设置计数器值"""
        offset = self.stage_offsets[stage]
        self.Matrix[tree, offset + local_idx] = value

    def insert_one(self, flow_id, count=1):
        """
        插入单个流ID (支持批量计数，但通常 count=1)
        实现溢出前馈逻辑。
        """
        indices = self._hash_indices(flow_id)
        for t in range(self.num_trees):
            for s in range(self.num_stages):
                idx = indices[t][s]
                val = self._get_counter(t, s, idx)
                max_val = self.max_counts[s]
                overflow_mark = self.overflow_marks[s]

                if val <= max_val:
                    # 未溢出，直接累加
                    new_val = val + count
                    if new_val > max_val:
                        # 本次累加导致溢出，设置为溢出标记
                        self._set_counter(t, s, idx, overflow_mark)
                        # 继续下一层（前馈）
                        continue
                    else:
                        self._set_counter(t, s, idx, new_val)
                        break  # 累加成功，停止
                else:
                    # 已经处于溢出状态，直接进入下一层
                    continue

    def insert_list(self, flow_list):
        """批量插入，flow_list 中每个元素为 (flow_id, count) 或仅 flow_id"""
        for item in flow_list:
            if isinstance(item, tuple):
                fid, cnt = item
                self.insert_one(fid, cnt)
            else:
                self.insert_one(item, 1)

    def insert_dict(self, flow_dict):
        """flow_dict: {flow_id: count}"""
        for fid, cnt in flow_dict.items():
            self.insert_one(fid, cnt)

    def query_one(self, flow_id):
        """
        查询单个流的估计值。
        对每棵树，从第一层开始累加，直到遇到未溢出计数器或到达最后一层，
        然后取各棵树结果的最小值。
        """
        indices = self._hash_indices(flow_id)
        tree_estimates = []
        for t in range(self.num_trees):
            total = 0
            for s in range(self.num_stages):
                idx = indices[t][s]
                val = self._get_counter(t, s, idx)
                overflow_mark = self.overflow_marks[s]
                if val == overflow_mark:
                    total += self.max_counts[s]
                else:
                    total += val
                    break  # 遇到未溢出计数器，停止累加
            tree_estimates.append(total)
        return min(tree_estimates)

    def query_d(self, flow_list):
        """返回每棵树对每个流的估计值，shape: (num_trees, len(flow_list))"""
        result = np.zeros((self.num_trees, len(flow_list)), dtype=np.uint32)
        for i, fid in enumerate(flow_list):
            indices = self._hash_indices(fid)
            for t in range(self.num_trees):
                total = 0
                for s in range(self.num_stages):
                    idx = indices[t][s]
                    val = self._get_counter(t, s, idx)
                    overflow_mark = self.overflow_marks[s]
                    if val == overflow_mark:
                        total += self.max_counts[s]
                    else:
                        total += val
                        break
                result[t, i] = total
        return result

    def query_all_list(self, flow_list):
        """返回每个流的最小估计值列表"""
        return [self.query_one(fid) for fid in flow_list]

    def clear(self):
        """清空所有计数器"""
        self.Matrix.fill(0)

    def save(self, file_name):
        """保存模型参数和计数器矩阵到文件"""
        dict_load = {
            'a': self.hash_a.tolist(),
            'b': self.hash_b.tolist(),
            'p': self.hash_p.tolist(),
            'offset': self.offset,
            'num_trees': self.num_trees,
            'num_stages': self.num_stages,
            'branch_factor': self.branch_factor,
            'bits_per_stage': list(self.bits_per_stage),
            'base_w': self.base_w,
            'widths': self.widths,
            'max_counts': self.max_counts,
            'overflow_marks': self.overflow_marks,
            'stage_offsets': self.stage_offsets,
            'total_counters_per_tree': self.total_counters_per_tree
        }
        rw_files.write_dict(file_name, dict_load)
        # 保存矩阵数据（可单独存为npy）
        np.save(file_name + "_matrix.npy", self.Matrix)




# 辅助函数：生成分数更新系数 g_i(k)，期望为 1/sqrt(r)
def _fractional_g(flow_id, row_idx, num_rows, seed_base=12345):
    h = hash((flow_id, row_idx, seed_base)) & 0xffffffff
    g = (h / (1 << 32)) * 2.0 - 1.0
    return g / np.sqrt(num_rows)


class SeqSketch:
    """
    SeqSketch: 顺序结构 + 压缩感知恢复 (OMP)
    """
    def __init__(self, num_hash_buckets=10000, bf_size=120000, bf_hash_num=7,
                 fs_rows=2, fs_cols=500000, flag=0, dict_seq=None,
                 heavy_load=None, bf_load=None, fs_load=None):
        self.flag = flag
        self.num_hash_buckets = num_hash_buckets
        self.bf_size = bf_size
        self.bf_hash_num = bf_hash_num
        self.fs_rows = fs_rows
        self.fs_cols = fs_cols

        if self.flag == 0:
            self.heavy = [None] * self.num_hash_buckets
            self.bf_bits = [0] * self.bf_size
            p_list = [6296197, 2254201, 7672057, 1002343, 9815713, 3436583, 4359587, 5638753, 8155451]
            self.bf_a = np.random.randint(low=10**4, high=10**5, size=(self.bf_hash_num, 1))
            self.bf_b = np.random.randint(low=10**4, high=10**5, size=(self.bf_hash_num, 1))
            selected = random.sample(p_list, self.bf_hash_num)
            self.bf_p = np.array(selected).reshape(self.bf_hash_num, 1)
            self.offset = 10**6 + 1

            self.fs_matrix = np.zeros((self.fs_rows, self.fs_cols), dtype=np.float64)
            self.fs_a = np.random.randint(low=10**4, high=10**5, size=(self.fs_rows, 1))
            self.fs_b = np.random.randint(low=10**4, high=10**5, size=(self.fs_rows, 1))
            self.fs_p = np.array(random.sample(p_list, self.fs_rows)).reshape(self.fs_rows, 1)

            # 记录所有出现过的流ID（用于恢复时的候选集）
            self.seen_flows = set()
        else:
            self.heavy = heavy_load
            self.bf_bits = bf_load
            self.fs_matrix = fs_load
            self.bf_a = np.array(dict_seq['bf_a'])
            self.bf_b = np.array(dict_seq['bf_b'])
            self.bf_p = np.array(dict_seq['bf_p'])
            self.fs_a = np.array(dict_seq['fs_a'])
            self.fs_b = np.array(dict_seq['fs_b'])
            self.fs_p = np.array(dict_seq['fs_p'])
            self.offset = dict_seq['offset']
            self.seen_flows = set(dict_seq.get('seen_flows', []))

        # 恢复结果存储
        self.recovered = None  # dict {flow_id: value}

    # ---------- Bloom Filter 内部方法 ----------
    def _bf_hash(self, flow_id):
        x = flow_id + self.offset
        h = (self.bf_a * x + self.bf_b) % self.bf_p % self.bf_size
        return h.flatten().tolist()

    def _bf_add(self, flow_id):
        idxs = self._bf_hash(flow_id)
        for idx in idxs:
            self.bf_bits[idx] = 1

    def _bf_check(self, flow_id):
        idxs = self._bf_hash(flow_id)
        return all(self.bf_bits[idx] == 1 for idx in idxs)

    # ---------- Fractional Sketch 内部方法 ----------
    def _fs_hash(self, flow_id, row):
        x = flow_id + self.offset
        h = (self.fs_a[row] * x + self.fs_b[row]) % self.fs_p[row] % self.fs_cols
        return int(h)

    def _fs_update(self, flow_id, delta):
        for row in range(self.fs_rows):
            col = self._fs_hash(flow_id, row)
            g = _fractional_g(flow_id, row, self.fs_rows)
            self.fs_matrix[row, col] += g * delta

    # ---------- 流贡献向量（用于恢复）----------
    def _flow_contribution(self, flow_id, value=1.0):
        """
        返回该流对测量向量的贡献：一个字典 {(row, col): contribution}
        因为每个流只影响每行的一个计数器。
        """
        contrib = {}
        for row in range(self.fs_rows):
            col = self._fs_hash(flow_id, row)
            g = _fractional_g(flow_id, row, self.fs_rows)
            contrib[(row, col)] = g * value
        return contrib

    # ---------- 主插入逻辑 ----------
    def insert_one(self, flow_id, count=1):
        for _ in range(count):
            self._insert_single(flow_id)

    def _insert_single(self, flow_id):
        j = hash(flow_id) % self.num_hash_buckets
        bucket = self.heavy[j]

        if bucket is None:
            self.heavy[j] = [flow_id, 1, 0]
        elif bucket[0] == flow_id:
            bucket[1] += 1
        else:
            bucket[2] += 1
            if bucket[2] > bucket[1]:
                evicted_id, evicted_c, _ = bucket
                self._fs_update(evicted_id, evicted_c)
                self.heavy[j] = [flow_id, 1, 0]

        self._fs_update(flow_id, 1)

        if not self._bf_check(flow_id):
            self.seen_flows.add(flow_id)
            self._bf_add(flow_id)

    def insert_list(self, flow_list):
        for fid in flow_list:
            self.insert_one(fid)

    def insert_dict(self, flow_dict):
        for fid, cnt in flow_dict.items():
            self.insert_one(fid, cnt)

    # ---------- 压缩感知恢复 ----------
    def get_known_flows(self):
        """返回 heavy 表中的已知流及其值"""
        known = {}
        for bucket in self.heavy:
            if bucket is not None:
                fid, c, _ = bucket
                known[fid] = c
        return known

    def _build_measurement_vector(self):
        """将 FS 矩阵展平为一维测量向量 y"""
        return self.fs_matrix.flatten()

    def _subtract_known_contributions(self, y, known_flows):
        """从测量向量 y 中减去已知流的贡献，返回剩余测量向量 b"""
        b = y.copy()
        for fid, val in known_flows.items():
            contrib = self._flow_contribution(fid, val)
            for (row, col), cval in contrib.items():
                idx = row * self.fs_cols + col
                b[idx] -= cval
        return b

    def _omp_recovery(self, candidate_flows, b, max_iter=None, tol=1e-6):
        """
        使用正交匹配追踪恢复稀疏向量 x (长度为 len(candidate_flows))
        返回字典 {flow_id: recovered_value}
        """
        N = len(candidate_flows)
        if N == 0:
            return {}
        # 如果候选流数量远大于测量数，设置最大迭代次数为测量数
        m = len(b)
        if max_iter is None:
            max_iter = min(m, N)

        # 预计算每个候选流的测量向量（稀疏表示）
        # 存储为列表，每个元素为 dict {(row,col): coefficient}
        flow_contribs = []
        for fid in candidate_flows:
            contrib = self._flow_contribution(fid, 1.0)  # 单位贡献
            flow_contribs.append(contrib)

        # 将 (row,col) 映射到全局索引
        def global_idx(row, col):
            return row * self.fs_cols + col

        # OMP 迭代
        residual = b.copy()
        support = []          # 选中的候选流索引
        x = np.zeros(N)       # 系数

        for _ in range(max_iter):
            # 计算每个候选流与残差的内积
            inner_prods = []
            for i, contrib in enumerate(flow_contribs):
                ip = 0.0
                for (r,c), coeff in contrib.items():
                    idx = global_idx(r, c)
                    ip += residual[idx] * coeff
                inner_prods.append(abs(ip))
            # 选择内积最大的索引
            best = np.argmax(inner_prods)
            if inner_prods[best] < tol:
                break
            support.append(best)

            # 构建子矩阵 A_S 和求解最小二乘
            # 由于每个流只影响少数测量，A_S 非常稀疏，我们直接构建正规方程
            ATA = np.zeros((len(support), len(support)))
            ATb = np.zeros(len(support))
            for si, idx_i in enumerate(support):
                contrib_i = flow_contribs[idx_i]
                # ATb[si] = <A_i, b>
                for (r,c), coeff_i in contrib_i.items():
                    gidx = global_idx(r, c)
                    ATb[si] += b[gidx] * coeff_i
                for sj, idx_j in enumerate(support[:si+1]):
                    contrib_j = flow_contribs[idx_j]
                    # 计算 A_i 与 A_j 的内积
                    dot = 0.0
                    # 由于每个流只有两个非零位置，内积只需检查重叠的 (row,col)
                    for (r,c), coeff_i in contrib_i.items():
                        if (r,c) in contrib_j:
                            dot += coeff_i * contrib_j[(r,c)]
                    ATA[si, sj] = dot
                    ATA[sj, si] = dot
            # 求解线性方程组 ATA * x_s = ATb
            try:
                x_s = np.linalg.solve(ATA, ATb)
            except np.linalg.LinAlgError:
                # 奇异矩阵，使用伪逆
                x_s = np.linalg.pinv(ATA) @ ATb
            # 更新系数
            for si, idx_i in enumerate(support):
                x[idx_i] = x_s[si]

            # 更新残差：b - A_S * x_S
            residual = b.copy()
            for si, idx_i in enumerate(support):
                contrib = flow_contribs[idx_i]
                coeff = x[idx_i]
                for (r,c), cval in contrib.items():
                    gidx = global_idx(r, c)
                    residual[gidx] -= coeff * cval

        # 构建结果字典
        result = {}
        for i, fid in enumerate(candidate_flows):
            if abs(x[i]) > 1e-6:
                result[fid] = x[i]
        return result

    def recover(self, candidate_flows=None, sparse_level=None):
        """
        执行压缩感知恢复。
        :param candidate_flows: 候选流ID列表，若为None则使用 self.seen_flows
        :param sparse_level: 稀疏度（最大迭代次数），默认自动设为 min(测量数, 候选数)
        :return: 恢复结果字典 {flow_id: estimated_value}
        """
        if candidate_flows is None:
            candidate_flows = list(self.seen_flows)
        if not candidate_flows:
            self.recovered = {}
            return {}

        # 已知流
        known = self.get_known_flows()
        # 未知流
        unknown = [fid for fid in candidate_flows if fid not in known]
        if not unknown:
            self.recovered = known.copy()
            return self.recovered

        # 测量向量
        y = self._build_measurement_vector()
        # 减去已知贡献
        b = self._subtract_known_contributions(y, known)

        # OMP 恢复
        if sparse_level is None:
            sparse_level = min(len(b), len(unknown))
        recovered_unknown = self._omp_recovery(unknown, b, max_iter=sparse_level)

        # 合并已知和恢复的未知流
        self.recovered = known.copy()
        self.recovered.update(recovered_unknown)
        return self.recovered

    def query_recovered(self, flow_id):
        """从恢复结果中查询流的值（必须先调用 recover）"""
        if self.recovered is None:
            raise RuntimeError("必须先调用 recover() 进行恢复")
        return self.recovered.get(flow_id, 0)

    # ---------- 传统查询（仅用于对比）----------
    def query_one(self, flow_id):
        j = hash(flow_id) % self.num_hash_buckets
        bucket = self.heavy[j]
        if bucket is not None and bucket[0] == flow_id:
            return bucket[1]
        else:
            estimates = []
            for row in range(self.fs_rows):
                col = self._fs_hash(flow_id, row)
                g = _fractional_g(flow_id, row, self.fs_rows)
                if abs(g) > 1e-9:
                    estimates.append(self.fs_matrix[row, col] / g)
            if estimates:
                return int(np.median(estimates))
            else:
                return 0

    def clear(self):
        self.heavy = [None] * self.num_hash_buckets
        self.bf_bits = [0] * self.bf_size
        self.fs_matrix.fill(0.0)
        self.seen_flows.clear()
        self.recovered = None

    def save(self, file_name):
        dict_seq = {
            'bf_a': self.bf_a.tolist(),
            'bf_b': self.bf_b.tolist(),
            'bf_p': self.bf_p.tolist(),
            'fs_a': self.fs_a.tolist(),
            'fs_b': self.fs_b.tolist(),
            'fs_p': self.fs_p.tolist(),
            'offset': self.offset,
            'num_hash_buckets': self.num_hash_buckets,
            'bf_size': self.bf_size,
            'bf_hash_num': self.bf_hash_num,
            'fs_rows': self.fs_rows,
            'fs_cols': self.fs_cols,
            'seen_flows': list(self.seen_flows)
        }
        data = {
            'dict_seq': dict_seq,
            'heavy_data': self.heavy,
            'bf_bits': self.bf_bits,
            'fs_matrix': self.fs_matrix.tolist(),
        }
        rw_files.write_dict(file_name, data)

    @classmethod
    def load(cls, file_name):
        data = rw_files.get_dict(file_name)
        dict_seq = data['dict_seq']
        heavy = data['heavy_data']
        bf_bits = data['bf_bits']
        fs_mat = np.array(data['fs_matrix'])
        return cls(flag=1, dict_seq=dict_seq, heavy_load=heavy,
                   bf_load=bf_bits, fs_load=fs_mat)


class EmbedSketch:
    """
    EmbedSketch 的压缩感知恢复（简化版）
    注意：EmbedSketch 的每个桶本身就是一个测量单元，但多行哈希会导致流被映射到多个桶。
    恢复时同样使用 OMP，每个流的贡献为所有命中桶的 c 和 d 的线性组合。
    这里给出基本框架，详细实现与 SeqSketch 类似，但需要根据桶结构调整贡献计算。
    """
    def __init__(self, fs_rows=2, fs_cols=500000, bf_bits_per_bucket=64,
                 flag=0, dict_embed=None, embed_load=None):
        self.flag = flag
        self.fs_rows = fs_rows
        self.fs_cols = fs_cols
        self.bf_bits_per_bucket = bf_bits_per_bucket

        if self.flag == 0:
            self.buckets = [[None, 0, 0, [0]*self.bf_bits_per_bucket] for _ in range(self.fs_cols)]
            p_list = [6296197, 2254201, 7672057, 1002343, 9815713, 3436583, 4359587, 5638753, 8155451]
            self.fs_a = np.random.randint(low=10**4, high=10**5, size=(self.fs_rows, 1))
            self.fs_b = np.random.randint(low=10**4, high=10**5, size=(self.fs_rows, 1))
            self.fs_p = np.array(random.sample(p_list, self.fs_rows)).reshape(self.fs_rows, 1)
            self.offset = 10**6 + 1
            self.bf_a = np.random.randint(low=10**4, high=10**5, size=(1, 1))
            self.bf_b = np.random.randint(low=10**4, high=10**5, size=(1, 1))
            self.bf_p = np.array([random.choice(p_list)]).reshape(1, 1)
            self.seen_flows = set()
        else:
            self.buckets = embed_load
            self.fs_a = np.array(dict_embed['fs_a'])
            self.fs_b = np.array(dict_embed['fs_b'])
            self.fs_p = np.array(dict_embed['fs_p'])
            self.bf_a = np.array(dict_embed['bf_a'])
            self.bf_b = np.array(dict_embed['bf_b'])
            self.bf_p = np.array(dict_embed['bf_p'])
            self.offset = dict_embed['offset']
            self.seen_flows = set(dict_embed.get('seen_flows', []))

        self.recovered = None

    def _hash_bucket(self, flow_id, row):
        x = flow_id + self.offset
        h = (self.fs_a[row] * x + self.fs_b[row]) % self.fs_p[row] % self.fs_cols
        return int(h)

    def _hash_bf(self, flow_id, bucket_idx):
        x = flow_id + self.offset + bucket_idx
        h = (self.bf_a[0] * x + self.bf_b[0]) % self.bf_p[0] % self.bf_bits_per_bucket
        return int(h)

    def _bf_check(self, flow_id, col):
        idx = self._hash_bf(flow_id, col)
        return self.buckets[col][3][idx] == 1

    def _bf_add(self, flow_id, col):
        idx = self._hash_bf(flow_id, col)
        self.buckets[col][3][idx] = 1

    def _flow_contribution(self, flow_id, value=1.0):
        """
        返回该流对测量向量的贡献。
        EmbedSketch 中，测量向量由所有桶的 c 值构成（或者 c+d？论文中使用 c 作为测量）。
        每个流会映射到 fs_rows 个桶（每行一个），每个桶贡献 value * g(row,flow_id) 到该桶的 c。
        注意：桶的 d 字段用于驱逐决策，不直接参与恢复方程。
        """
        contrib = {}
        for row in range(self.fs_rows):
            col = self._hash_bucket(flow_id, row)
            g = _fractional_g(flow_id, row, self.fs_rows)
            # 测量向量索引：桶索引 (col) 即可，因为每个桶只有一个 c 值
            contrib[col] = contrib.get(col, 0.0) + g * value
        return contrib

    def insert_one(self, flow_id, count=1):
        for _ in range(count):
            self._insert_single(flow_id)

    def _insert_single(self, flow_id):
        row = 0
        col = self._hash_bucket(flow_id, row)
        bucket = self.buckets[col]

        if bucket[0] is None:
            bucket[0] = flow_id
            bucket[1] = 1
            bucket[2] = 0
        elif bucket[0] == flow_id:
            bucket[1] += 1
        else:
            bucket[2] += 1
            if bucket[2] > bucket[1]:
                bucket[0] = flow_id
                bucket[1] = 1
                bucket[2] = 0

        if not self._bf_check(flow_id, col):
            self.seen_flows.add(flow_id)
            self._bf_add(flow_id, col)

    def insert_list(self, flow_list):
        for fid in flow_list:
            self.insert_one(fid)

    def insert_dict(self, flow_dict):
        for fid, cnt in flow_dict.items():
            self.insert_one(fid, cnt)

    def get_known_flows(self):
        known = {}
        for bucket in self.buckets:
            if bucket[0] is not None:
                known[bucket[0]] = bucket[1]
        return known

    def _build_measurement_vector(self):
        """测量向量 y = 所有桶的 c 值"""
        return np.array([bucket[1] for bucket in self.buckets], dtype=np.float64)

    def _subtract_known_contributions(self, y, known_flows):
        b = y.copy()
        for fid, val in known_flows.items():
            contrib = self._flow_contribution(fid, val)
            for col, cval in contrib.items():
                b[col] -= cval
        return b

    def _omp_recovery(self, candidate_flows, b, max_iter=None, tol=1e-6):
        N = len(candidate_flows)
        if N == 0:
            return {}
        m = len(b)
        if max_iter is None:
            max_iter = min(m, N)

        flow_contribs = []
        for fid in candidate_flows:
            contrib = self._flow_contribution(fid, 1.0)
            flow_contribs.append(contrib)

        residual = b.copy()
        support = []
        x = np.zeros(N)

        for _ in range(max_iter):
            inner_prods = []
            for i, contrib in enumerate(flow_contribs):
                ip = sum(residual[col] * coeff for col, coeff in contrib.items())
                inner_prods.append(abs(ip))
            best = np.argmax(inner_prods)
            if inner_prods[best] < tol:
                break
            support.append(best)

            # 构建正规方程
            ATA = np.zeros((len(support), len(support)))
            ATb = np.zeros(len(support))
            for si, idx_i in enumerate(support):
                contrib_i = flow_contribs[idx_i]
                ATb[si] = sum(b[col] * coeff for col, coeff in contrib_i.items())
                for sj, idx_j in enumerate(support[:si+1]):
                    contrib_j = flow_contribs[idx_j]
                    dot = sum(contrib_i.get(col, 0) * contrib_j.get(col, 0) for col in set(contrib_i) & set(contrib_j))
                    ATA[si, sj] = dot
                    ATA[sj, si] = dot
            try:
                x_s = np.linalg.solve(ATA, ATb)
            except np.linalg.LinAlgError:
                x_s = np.linalg.pinv(ATA) @ ATb
            for si, idx_i in enumerate(support):
                x[idx_i] = x_s[si]

            residual = b.copy()
            for si, idx_i in enumerate(support):
                contrib = flow_contribs[idx_i]
                coeff = x[idx_i]
                for col, cval in contrib.items():
                    residual[col] -= coeff * cval

        result = {}
        for i, fid in enumerate(candidate_flows):
            if abs(x[i]) > 1e-6:
                result[fid] = x[i]
        return result

    def recover(self, candidate_flows=None, sparse_level=None):
        if candidate_flows is None:
            candidate_flows = list(self.seen_flows)
        if not candidate_flows:
            self.recovered = {}
            return {}
        known = self.get_known_flows()
        unknown = [fid for fid in candidate_flows if fid not in known]
        if not unknown:
            self.recovered = known.copy()
            return self.recovered
        y = self._build_measurement_vector()
        b = self._subtract_known_contributions(y, known)
        if sparse_level is None:
            sparse_level = min(len(b), len(unknown))
        recovered_unknown = self._omp_recovery(unknown, b, max_iter=sparse_level)
        self.recovered = known.copy()
        self.recovered.update(recovered_unknown)
        return self.recovered

    def query_recovered(self, flow_id):
        if self.recovered is None:
            raise RuntimeError("必须先调用 recover() 进行恢复")
        return self.recovered.get(flow_id, 0)

    def query_one(self, flow_id):
        row = 0
        col = self._hash_bucket(flow_id, row)
        bucket = self.buckets[col]
        if bucket[0] == flow_id:
            return bucket[1]
        return 0

    def clear(self):
        self.buckets = [[None, 0, 0, [0]*self.bf_bits_per_bucket] for _ in range(self.fs_cols)]
        self.seen_flows.clear()
        self.recovered = None

    def save(self, file_name):
        dict_embed = {
            'fs_a': self.fs_a.tolist(),
            'fs_b': self.fs_b.tolist(),
            'fs_p': self.fs_p.tolist(),
            'bf_a': self.bf_a.tolist(),
            'bf_b': self.bf_b.tolist(),
            'bf_p': self.bf_p.tolist(),
            'offset': self.offset,
            'fs_rows': self.fs_rows,
            'fs_cols': self.fs_cols,
            'bf_bits_per_bucket': self.bf_bits_per_bucket,
            'seen_flows': list(self.seen_flows)
        }
        data = {'dict_embed': dict_embed, 'buckets': self.buckets}
        rw_files.write_dict(file_name, data)

    @classmethod
    def load(cls, file_name):
        data = rw_files.get_dict(file_name)
        return cls(flag=1, dict_embed=data['dict_embed'], embed_load=data['buckets'])



# class SeqSketch:
#     """
#     SeqSketch: 顺序结构，包含：
#         - 哈希表 H (key-value pairs) 用于大流
#         - Bloom Filter BF 用于去重
#         - Fractional Sketch FS 用于小流
#     每个条目: H[j] = [flow_id, c, d] 或 None
#         c: 属于该流的计数
#         d: 冲突计数（不属于该流）
#     """
#     def __init__(self, num_hash_buckets=10000, bf_size=120000, bf_hash_num=7,
#                  fs_rows=2, fs_cols=500000, flag=0, dict_seq=None,
#                  heavy_load=None, bf_load=None, fs_load=None):
#         """
#         初始化 SeqSketch。
#         :param num_hash_buckets: 哈希表 H 的桶数
#         :param bf_size: Bloom Filter 的位数
#         :param bf_hash_num: Bloom Filter 的哈希函数个数
#         :param fs_rows: Fractional Sketch 的行数 (r)
#         :param fs_cols: Fractional Sketch 的列数 (w)
#         :param flag: 0 新建，1 从字典加载
#         :param dict_seq: 加载时使用的参数字典
#         :param heavy_load: 加载时哈希表数据 (list)
#         :param bf_load: 加载时 Bloom Filter 位数组 (list of int)
#         :param fs_load: 加载时 Fractional Sketch 矩阵 (numpy array)
#         """
#         self.flag = flag
#         self.num_hash_buckets = num_hash_buckets
#         self.bf_size = bf_size
#         self.bf_hash_num = bf_hash_num
#         self.fs_rows = fs_rows
#         self.fs_cols = fs_cols
#
#         if self.flag == 0:
#             # 初始化哈希表 H: 每个桶为 None 或 [flow_id, c, d]
#             self.heavy = [None] * self.num_hash_buckets
#
#             # 初始化 Bloom Filter: 使用位数组
#             self.bf_bits = [0] * self.bf_size
#             # 生成 BF 的哈希函数参数 (使用与 CM 类似的结构)
#             p_list = [6296197, 2254201, 7672057, 1002343, 9815713, 3436583, 4359587, 5638753, 8155451]
#             self.bf_a = np.random.randint(low=10**4, high=10**5, size=(self.bf_hash_num, 1))
#             self.bf_b = np.random.randint(low=10**4, high=10**5, size=(self.bf_hash_num, 1))
#             selected = random.sample(p_list, self.bf_hash_num)
#             self.bf_p = np.array(selected).reshape(self.bf_hash_num, 1)
#             self.offset = 10**6 + 1
#
#             # 初始化 Fractional Sketch (FS) 矩阵，使用浮点数存储
#             self.fs_matrix = np.zeros((self.fs_rows, self.fs_cols), dtype=np.float64)
#
#             # 生成 FS 的哈希函数参数 (每行独立的一组哈希)
#             self.fs_a = np.random.randint(low=10**4, high=10**5, size=(self.fs_rows, 1))
#             self.fs_b = np.random.randint(low=10**4, high=10**5, size=(self.fs_rows, 1))
#             self.fs_p = np.array(random.sample(p_list, self.fs_rows)).reshape(self.fs_rows, 1)
#
#         else:
#             # 从加载数据恢复
#             self.heavy = heavy_load
#             self.bf_bits = bf_load
#             self.fs_matrix = fs_load
#             self.bf_a = np.array(dict_seq['bf_a'])
#             self.bf_b = np.array(dict_seq['bf_b'])
#             self.bf_p = np.array(dict_seq['bf_p'])
#             self.fs_a = np.array(dict_seq['fs_a'])
#             self.fs_b = np.array(dict_seq['fs_b'])
#             self.fs_p = np.array(dict_seq['fs_p'])
#             self.offset = dict_seq['offset']
#
#     # ---------- Bloom Filter 内部方法 ----------
#     def _bf_hash(self, flow_id):
#         """返回 BF 的所有哈希索引 (list of int)"""
#         x = flow_id + self.offset
#         h = (self.bf_a * x + self.bf_b) % self.bf_p % self.bf_size
#         return h.flatten().tolist()
#
#     def _bf_add(self, flow_id):
#         """将 flow_id 加入 Bloom Filter"""
#         idxs = self._bf_hash(flow_id)
#         for idx in idxs:
#             self.bf_bits[idx] = 1
#
#     def _bf_check(self, flow_id):
#         """检查 flow_id 是否可能在 Bloom Filter 中"""
#         idxs = self._bf_hash(flow_id)
#         return all(self.bf_bits[idx] == 1 for idx in idxs)
#
#     # ---------- Fractional Sketch 内部方法 ----------
#     def _fs_hash(self, flow_id, row):
#         """返回 FS 中第 row 行的哈希列索引"""
#         x = flow_id + self.offset
#         h = (self.fs_a[row] * x + self.fs_b[row]) % self.fs_p[row] % self.fs_cols
#         return int(h)
#
#     def _fs_update(self, flow_id, delta):
#         """在 FS 中更新 flow_id，增加 delta * g(row, flow_id)"""
#         for row in range(self.fs_rows):
#             col = self._fs_hash(flow_id, row)
#             g = _fractional_g(flow_id, row, self.fs_rows)
#             self.fs_matrix[row, col] += g * delta
#
#     # ---------- 主插入逻辑 ----------
#     def insert_one(self, flow_id, count=1):
#         """
#         插入一个包 (或 count 个包)。
#         严格按照论文 Algorithm 1 实现。
#         """
#         for _ in range(count):
#             self._insert_single(flow_id)
#
#     def _insert_single(self, flow_id):
#         j = hash(flow_id) % self.num_hash_buckets   # 简化哈希，实际应使用独立哈希
#         bucket = self.heavy[j]
#
#         if bucket is None:
#             # 空桶，直接插入
#             self.heavy[j] = [flow_id, 1, 0]
#         elif bucket[0] == flow_id:
#             # 同一流，增加 c
#             bucket[1] += 1
#         else:
#             # 不同流，增加 d
#             bucket[2] += 1
#             if bucket[2] > bucket[1]:
#                 # 驱逐当前流，发送 (flow_id, c) 到控制器（此处模拟打印或存储）
#                 # 实际应通过回调发送，我们暂记录在列表中
#                 evicted_id, evicted_c, _ = bucket
#                 # 将被驱逐的流插入 FS
#                 self._fs_update(evicted_id, evicted_c)
#                 # 插入新流
#                 self.heavy[j] = [flow_id, 1, 0]
#
#         # 无论是否冲突，都需要更新 FS（论文中每包都要更新 FS）
#         self._fs_update(flow_id, 1)
#
#         # 如果 flow_id 不在 BF 中，发送到控制器并加入 BF
#         if not self._bf_check(flow_id):
#             # 发送 flow_id 到控制器（模拟）
#             # print(f"Send new flow {flow_id} to controller")
#             self._bf_add(flow_id)
#
#     def insert_list(self, flow_list):
#         """批量插入，flow_list 为流标识列表，每个元素插入一个包"""
#         for fid in flow_list:
#             self.insert_one(fid)
#
#     def insert_dict(self, flow_dict):
#         """批量插入，flow_dict = {flow_id: count}"""
#         for fid, cnt in flow_dict.items():
#             self.insert_one(fid, cnt)
#
#     # ---------- 查询方法（简化，非论文全局恢复）----------
#     def query_one(self, flow_id):
#         """
#         查询单个流的估计值。
#         注意：论文中精确查询需要控制平面优化求解，这里仅提供简单估计：
#             优先从哈希表 H 中获取，否则从 FS 中估计（取各行的中位数除以期望系数）
#         """
#         j = hash(flow_id) % self.num_hash_buckets
#         bucket = self.heavy[j]
#         if bucket is not None and bucket[0] == flow_id:
#             return bucket[1]
#         else:
#             # 从 FS 估计：对每行，估计值为 counter / g，取中位数
#             estimates = []
#             for row in range(self.fs_rows):
#                 col = self._fs_hash(flow_id, row)
#                 g = _fractional_g(flow_id, row, self.fs_rows)
#                 if abs(g) > 1e-9:
#                     estimates.append(self.fs_matrix[row, col] / g)
#             if estimates:
#                 return int(np.median(estimates))
#             else:
#                 return 0
#
#     def query_list(self, flow_list):
#         return [self.query_one(fid) for fid in flow_list]
#
#     def query_all_list(self, flow_list):
#         """与 query_list 相同"""
#         return self.query_list(flow_list)
#
#     def clear(self):
#         """清空所有数据结构"""
#         self.heavy = [None] * self.num_hash_buckets
#         self.bf_bits = [0] * self.bf_size
#         self.fs_matrix.fill(0.0)
#
#     def save(self, file_name):
#         """保存到文件（需配合 rw_files）"""
#         dict_seq = {
#             'bf_a': self.bf_a.tolist(),
#             'bf_b': self.bf_b.tolist(),
#             'bf_p': self.bf_p.tolist(),
#             'fs_a': self.fs_a.tolist(),
#             'fs_b': self.fs_b.tolist(),
#             'fs_p': self.fs_p.tolist(),
#             'offset': self.offset,
#             'num_hash_buckets': self.num_hash_buckets,
#             'bf_size': self.bf_size,
#             'bf_hash_num': self.bf_hash_num,
#             'fs_rows': self.fs_rows,
#             'fs_cols': self.fs_cols,
#         }
#         data = {
#             'dict_seq': dict_seq,
#             'heavy_data': self.heavy,
#             'bf_bits': self.bf_bits,
#             'fs_matrix': self.fs_matrix.tolist(),
#         }
#         rw_files.write_dict(file_name, data)
#
#     @classmethod
#     def load(cls, file_name):
#         """从文件加载"""
#         data = rw_files.get_dict(file_name)
#         dict_seq = data['dict_seq']
#         heavy = data['heavy_data']
#         bf_bits = data['bf_bits']
#         fs_mat = np.array(data['fs_matrix'])
#         return cls(flag=1, dict_seq=dict_seq, heavy_load=heavy,
#                    bf_load=bf_bits, fs_load=fs_mat)
#
#
# class EmbedSketch:
#     """
#     EmbedSketch: 嵌入结构，每个桶包含：
#         - 一个 KV 对 (flow_id, c, d)
#         - 一个本地 Bloom Filter (位数组)
#     """
#     def __init__(self, fs_rows=2, fs_cols=500000, bf_bits_per_bucket=64,
#                  flag=0, dict_embed=None, embed_load=None):
#         self.flag = flag
#         self.fs_rows = fs_rows
#         self.fs_cols = fs_cols
#         self.bf_bits_per_bucket = bf_bits_per_bucket
#
#         if self.flag == 0:
#             self.buckets = [[None, 0, 0, [0]*self.bf_bits_per_bucket] for _ in range(self.fs_cols)]
#             p_list = [6296197, 2254201, 7672057, 1002343, 9815713, 3436583, 4359587, 5638753, 8155451]
#             self.fs_a = np.random.randint(low=10**4, high=10**5, size=(self.fs_rows, 1))
#             self.fs_b = np.random.randint(low=10**4, high=10**5, size=(self.fs_rows, 1))
#             self.fs_p = np.array(random.sample(p_list, self.fs_rows)).reshape(self.fs_rows, 1)
#             self.offset = 10**6 + 1
#             self.bf_a = np.random.randint(low=10**4, high=10**5, size=(1, 1))
#             self.bf_b = np.random.randint(low=10**4, high=10**5, size=(1, 1))
#             self.bf_p = np.array([random.choice(p_list)]).reshape(1, 1)
#         else:
#             self.buckets = embed_load
#             self.fs_a = np.array(dict_embed['fs_a'])
#             self.fs_b = np.array(dict_embed['fs_b'])
#             self.fs_p = np.array(dict_embed['fs_p'])
#             self.bf_a = np.array(dict_embed['bf_a'])
#             self.bf_b = np.array(dict_embed['bf_b'])
#             self.bf_p = np.array(dict_embed['bf_p'])
#             self.offset = dict_embed['offset']
#
#     def _hash_bucket(self, flow_id, row):
#         x = flow_id + self.offset
#         h = (self.fs_a[row] * x + self.fs_b[row]) % self.fs_p[row] % self.fs_cols
#         return int(h)
#
#     def _hash_bf(self, flow_id, bucket_idx):
#         x = flow_id + self.offset + bucket_idx
#         h = (self.bf_a[0] * x + self.bf_b[0]) % self.bf_p[0] % self.bf_bits_per_bucket
#         return int(h)
#
#     # ---------- 修正后的 BF 操作 ----------
#     def _bf_check(self, flow_id, col):
#         """检查 col 桶内的 BF 是否包含 flow_id"""
#         idx = self._hash_bf(flow_id, col)
#         bucket = self.buckets[col]
#         return bucket[3][idx] == 1
#
#     def _bf_add(self, flow_id, col):
#         """向 col 桶内的 BF 添加 flow_id"""
#         idx = self._hash_bf(flow_id, col)
#         bucket = self.buckets[col]
#         bucket[3][idx] = 1
#
#     def insert_one(self, flow_id, count=1):
#         for _ in range(count):
#             self._insert_single(flow_id)
#
#     def _insert_single(self, flow_id):
#         row = 0
#         col = self._hash_bucket(flow_id, row)
#         bucket = self.buckets[col]
#
#         if bucket[0] is None:
#             bucket[0] = flow_id
#             bucket[1] = 1
#             bucket[2] = 0
#         elif bucket[0] == flow_id:
#             bucket[1] += 1
#         else:
#             bucket[2] += 1
#             if bucket[2] > bucket[1]:
#                 # 驱逐
#                 bucket[0] = flow_id
#                 bucket[1] = 1
#                 bucket[2] = 0
#
#         if not self._bf_check(flow_id, col):
#             self._bf_add(flow_id, col)
#
#     def insert_list(self, flow_list):
#         for fid in flow_list:
#             self.insert_one(fid)
#
#     def insert_dict(self, flow_dict):
#         for fid, cnt in flow_dict.items():
#             self.insert_one(fid, cnt)
#
#     def query_one(self, flow_id):
#         row = 0
#         col = self._hash_bucket(flow_id, row)
#         bucket = self.buckets[col]
#         if bucket[0] == flow_id:
#             return bucket[1]
#         return 0
#
#     def query_list(self, flow_list):
#         return [self.query_one(fid) for fid in flow_list]
#
#     def query_all_list(self, flow_list):
#         return self.query_list(flow_list)
#
#     def clear(self):
#         self.buckets = [[None, 0, 0, [0]*self.bf_bits_per_bucket] for _ in range(self.fs_cols)]
#
#     def save(self, file_name):
#         dict_embed = {
#             'fs_a': self.fs_a.tolist(),
#             'fs_b': self.fs_b.tolist(),
#             'fs_p': self.fs_p.tolist(),
#             'bf_a': self.bf_a.tolist(),
#             'bf_b': self.bf_b.tolist(),
#             'bf_p': self.bf_p.tolist(),
#             'offset': self.offset,
#             'fs_rows': self.fs_rows,
#             'fs_cols': self.fs_cols,
#             'bf_bits_per_bucket': self.bf_bits_per_bucket,
#         }
#         data = {'dict_embed': dict_embed, 'buckets': self.buckets}
#         rw_files.write_dict(file_name, data)
#
#     @classmethod
#     def load(cls, file_name):
#         data = rw_files.get_dict(file_name)
#         return cls(flag=1, dict_embed=data['dict_embed'], embed_load=data['buckets'])
