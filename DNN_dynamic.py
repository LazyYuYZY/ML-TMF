from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import sys
import time

from DNN1 import *
from sketchs import *

P_flowdrop=0.01

# 控制台输出记录到文件
class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def conflict(index_np,dict_cm,w=22 * 10 ** 4,d=3,max_conf=1):
    a = np.array(dict_cm['a'])
    b = np.array(dict_cm['b'])
    p = np.array(dict_cm['p'])
    offset = dict_cm['offset']
    cm_d_id=(a*(index_np+offset)+b)%p%w
    data_x_d= np.full((index_np.shape[0],d*max_conf), -1)

    for i in range(d):
        unique_values, counts = np.unique(cm_d_id[i], return_counts=True)
        conf_items=unique_values[counts>1]
        for conf_cm_id in conf_items:#冲突流在sketch中的索引
            conf_flows_id=np.where(np.isin(cm_d_id[i], conf_cm_id))[0]#单个conter冲突流在sketch的索引
            conf_flows_id_list=conf_flows_id.tolist()
            for conf_flow_id in conf_flows_id_list:
                k = 0
                for j in range(1+max_conf):
                    if j>=len(conf_flows_id_list):
                        break
                    if conf_flows_id_list[j] != conf_flow_id:
                        data_x_d[conf_flow_id][i * max_conf + k] = index_np[conf_flows_id_list[j]]
                        k = k + 1
                        if k==max_conf:
                            break
    return data_x_d
def select_indices(index_array,p):
    # 计算要选择的行数
    num_rows =index_array.shape[0]
    num_selected_rows = int(num_rows * p)
    # 生成要选择的行索引
    selected_indices = np.random.choice(num_rows, size=num_selected_rows, replace=False)
    return selected_indices

import numpy as np
import hashlib
from typing import Tuple, Iterable, Dict

class HashTableFlowCollector:
    """
    模拟交换机上的专用哈希表，用于收集部分流作为训练数据。
    每个条目：13 字节流键 + 3 字节计数器（这里用 int64 和 int64 代替）。
    哈希函数独立于 CM Sketch 中使用的哈希函数。
    """
    def __init__(self, table_size: int = 1000, seed: int = 0):
        self.table_size = table_size
        self.seed = seed
        self.keys = np.full(table_size, -1, dtype=np.int64)   # -1 表示空
        self.counters = np.zeros(table_size, dtype=np.int64)

    def _hash(self, flow_id: int) -> int:
        """专用哈希函数，与 sketch 无关"""
        # 简单且快速的随机映射，使用 MD5 保证均匀性
        data = f"{self.seed}_{flow_id}".encode()
        h = hashlib.md5(data).digest()
        return int.from_bytes(h[:8], 'big') % self.table_size

    def process_packet(self, flow_id: int):
        idx = self._hash(flow_id)
        stored = self.keys[idx]
        if stored == -1:
            self.keys[idx] = flow_id
            self.counters[idx] = 1
        elif stored == flow_id:
            self.counters[idx] += 1
        # 其它情况：不作插入

    def process_packets(self, flow_ids: Iterable[int]):
        for fid in flow_ids:
            self.process_packet(fid)

    def collect_and_reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """返回当前周期收集到的 (flow_ids, counts) 并清空哈希表"""
        mask = self.keys != -1
        col_ids = self.keys[mask].copy()
        col_cnt = self.counters[mask].copy()
        # 清空
        self.keys[:] = -1
        self.counters[:] = 0
        return col_ids, col_cnt

def generate_packet_sequence(flow_dict: Dict[int, int],
                             num_packets: int,
                             rng: np.random.Generator = None) -> np.ndarray:
    """
    根据流大小加权随机生成数据包到达序列。
    flow_dict: {flow_id: flow_size}
    num_packets: 序列长度
    返回: shape (num_packets,) 的 flow_id 数组
    """
    if rng is None:
        rng = np.random.default_rng()
    ids = np.array(list(flow_dict.keys()))
    sizes = np.array(list(flow_dict.values()), dtype=float)
    probs = sizes / sizes.sum()
    return rng.choice(ids, size=num_packets, p=probs)




if __name__ == '__main__':
    '''自定义目录存放日志文件'''
    log_path = '../Logs/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    # 日志文件名按照程序运行时间设置
    log_file_name = log_path + 'log-' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.log'
    # 记录正常的 print 信息
    sys.stdout = Logger(log_file_name)
    # 记录 traceback 异常信息
    sys.stderr = Logger(log_file_name)

    '''准备数据'''
    max_conf=2


    alpha1=2.4
    alpha2=2.4
    n1=120000
    n2=240000

    # alpha1 = 2.4
    # alpha2 = 1.6
    # n1 = 120000
    # n2 = 120000

    number1=0
    number2=1

    test_y_path = "../zipf_testdata_set/testdata_flows/" + str(
                int(alpha2 * 10)) + "_" + str(n2).zfill(7) + "_" + str(number2).zfill(2) + ".txt"
    dataset_y_path="../zipf_testdata_set/testdata_flows/" + str(
                int(alpha1 * 10)) + "_" + str(n1).zfill(7) + "_" + str(number1).zfill(2) + ".txt"
    dataset2_y_path="../zipf_testdata_set/testdata_flows/" + str(
                int(alpha2 * 10)) + "_" + str(n2).zfill(7) + "_" + str(number1).zfill(2) + ".txt"
    # 处理cm数据
    file_name = "../sketch_params/160000_cm_sketch.txt"
    dict_cm = rw_files.get_dict(file_name)
    cm_d = 3
    cm_w = dict_cm['w']

    '''流数据集'''
    test_flows_data_str = rw_files.get_dict(test_y_path)
    test_flows_data = {}
    for key, value in test_flows_data_str.items():
        new_key = int(key)
        test_flows_data[new_key] = value
    test_flows_data_list = list(test_flows_data.values())

    cm_test_path = "../zipf_testdata_set/cm/" + str(cm_w).zfill(6) + "_" + str(int(alpha2 * 10)) + "_" + str(n2).zfill(7) + "_" + str(number2).zfill(2) + ".txt"
    # 若有对应cm数据集导入，没有就插入cm导出数据集
    if 1<0:#os.path.exists(cm_train_path):os.path.exists(cm_test_path):
        cm_sketch_load = np.loadtxt(cm_test_path)
        cm_sketch_now = cm_sketch(cm_d=cm_d, cm_w=cm_w, flag=1, dict_cm=dict_cm, cm_sketch_load=cm_sketch_load)
    else:
        cm_sketch_load = np.full((cm_d, cm_w), 0)  # cm存储的counter值
        cm_sketch_now = cm_sketch(cm_d=cm_d, cm_w=cm_w, flag=1, dict_cm=dict_cm, cm_sketch_load=cm_sketch_load)
        cm_sketch_now.insert_dict(test_flows_data)
        cm_sketch_load = cm_sketch_now.Matrix  # cm的counter值
        '''导出数据'''
        np.savetxt(cm_test_path, cm_sketch_load, fmt='%d')

    cm_sketch_now2 = np.loadtxt(cm_test_path)
    sketch_now2 = cm_sketch(cm_d=3, cm_w=cm_w, flag=1, dict_cm=dict_cm, cm_sketch_load=cm_sketch_now2)


    '''获取test时刻dnn_cm的输入,即冲突流的id'''
    test_index_array = np.array(list(test_flows_data.keys()))

    selected_indices = select_indices(test_index_array, 1 - P_flowdrop)
    test_index_array = test_index_array

    test_x_ids = conflict(index_np=test_index_array, dict_cm=dict_cm, w=cm_w, d=cm_d, max_conf=max_conf)

    '''获取所在时刻冲突流的d个查询值'''
    test_x = np.full((test_x_ids.shape[0], cm_d * (1 + test_x_ids.shape[1])), 0)
    for i in range(cm_d * max_conf):
        conf_flows_id = np.where(test_x_ids[:, i] != -1)[0]
        test_x[conf_flows_id, i * cm_d:(i + 1) * cm_d] = sketch_now2.query_d_np(test_x_ids[conf_flows_id, i]).T
    test_x[:, cm_d * test_x_ids.shape[1]:] = sketch_now2.query_d_np(test_index_array).T

    '''test dataset'''
    test_x = np.log(1 + test_x)
    test_y = np.array(test_flows_data_list).reshape(-1, 1)
    print(test_x.shape, test_y.shape)

    '''流数据集'''
    dataset_flows_data_str = rw_files.get_dict(dataset_y_path)
    dataset_flows_data = {}
    for key, value in dataset_flows_data_str.items():
        new_key = int(key)
        dataset_flows_data[new_key] = value
    dataset_flows_data_list = list(dataset_flows_data.values())


    cm_train_path = "../zipf_testdata_set/cm/" + str(cm_w).zfill(6) + "_" + str(int(alpha1 * 10)) + "_" + str(n1).zfill(
        7) + "_" + str(number1).zfill(2) + ".txt"

    # 若有对应cm数据集导入，没有就插入cm导出数据集
    if 1<0:#os.path.exists(cm_train_path):
        cm_sketch_load = np.loadtxt(cm_train_path)
        cm_sketch_now = cm_sketch(cm_d=cm_d, cm_w=cm_w, flag=1, dict_cm=dict_cm, cm_sketch_load=cm_sketch_load)
    else:
        cm_sketch_load = np.full((cm_d, cm_w), 0)  # cm存储的counter值
        cm_sketch_now = cm_sketch(cm_d=cm_d, cm_w=cm_w, flag=1, dict_cm=dict_cm, cm_sketch_load=cm_sketch_load)
        cm_sketch_now.insert_dict(dataset_flows_data)
        cm_sketch_load = cm_sketch_now.Matrix  # cm的counter值
        '''导出数据'''
        np.savetxt(cm_train_path, cm_sketch_load, fmt='%d')

    cm_sketch_now1 = np.loadtxt(cm_train_path)
    sketch_now1 = cm_sketch(cm_d=3, cm_w=cm_w, flag=1, dict_cm=dict_cm, cm_sketch_load=cm_sketch_now1)

    '''获取dataset时刻dnn_cm的输入,即冲突流的id'''
    dataset_index_array = np.array(list(dataset_flows_data.keys()))

    selected_indices = select_indices(dataset_index_array, 1 - P_flowdrop)
    dataset_index_array = dataset_index_array

    dataset_x_ids = conflict(index_np=dataset_index_array, dict_cm=dict_cm, w=cm_w, d=cm_d, max_conf=max_conf)

    '''获取所在时刻冲突流的d个查询值'''
    dataset_x = np.full((dataset_x_ids.shape[0], cm_d * (1 + dataset_x_ids.shape[1])), 0)
    for i in range(cm_d * max_conf):
        conf_flows_id = np.where(dataset_x_ids[:, i] != -1)[0]
        dataset_x[conf_flows_id, i * cm_d:(i + 1) * cm_d] = sketch_now1.query_d_np(dataset_x_ids[conf_flows_id, i]).T
    dataset_x[:, cm_d * dataset_x_ids.shape[1]:] = sketch_now1.query_d_np(dataset_index_array).T


    '''dataset dataset'''
    dataset_x = np.log(1 + dataset_x)
    dataset_y = np.array(dataset_flows_data_list).reshape(-1, 1)
    print(dataset_x.shape, dataset_y.shape)

    '''流数据集'''
    dataset2_flows_data_str = rw_files.get_dict(dataset2_y_path)
    dataset2_flows_data = {}
    for key, value in dataset2_flows_data_str.items():
        new_key = int(key)
        dataset2_flows_data[new_key] = value
    dataset2_flows_data_list = list(dataset2_flows_data.values())

    cm_train_path = "../zipf_testdata_set/cm/" + str(cm_w).zfill(6) + "_" + str(int(alpha2 * 10)) + "_" + str(n2).zfill(
        7) + "_" + str(number1).zfill(2) + ".txt"

    # 若有对应cm数据集导入，没有就插入cm导出数据集
    if 1<0:#os.path.exists(cm_train_path):os.path.exists(cm_train_path):
        cm_sketch_load = np.loadtxt(cm_train_path)
        cm_sketch_now = cm_sketch(cm_d=cm_d, cm_w=cm_w, flag=1, dict_cm=dict_cm, cm_sketch_load=cm_sketch_load)
    else:
        cm_sketch_load = np.full((cm_d, cm_w), 0)  # cm存储的counter值
        cm_sketch_now = cm_sketch(cm_d=cm_d, cm_w=cm_w, flag=1, dict_cm=dict_cm, cm_sketch_load=cm_sketch_load)
        cm_sketch_now.insert_dict(dataset2_flows_data)
        cm_sketch_load = cm_sketch_now.Matrix  # cm的counter值
        '''导出数据'''
        np.savetxt(cm_train_path, cm_sketch_load, fmt='%d')

    cm_sketch_now1 = np.loadtxt(cm_train_path)
    sketch_now1 = cm_sketch(cm_d=3, cm_w=cm_w, flag=1, dict_cm=dict_cm, cm_sketch_load=cm_sketch_now1)

    '''获取dataset2时刻dnn_cm的输入,即冲突流的id'''
    dataset2_index_array = np.array(list(dataset2_flows_data.keys()))

    selected_indices = select_indices(dataset2_index_array, 1 - P_flowdrop)
    dataset2_index_array = dataset2_index_array

    dataset2_x_ids = conflict(index_np=dataset2_index_array, dict_cm=dict_cm, w=cm_w, d=cm_d, max_conf=max_conf)

    '''获取所在时刻冲突流的d个查询值'''
    dataset2_x = np.full((dataset2_x_ids.shape[0], cm_d * (1 + dataset2_x_ids.shape[1])), 0)
    for i in range(cm_d * max_conf):
        conf_flows_id = np.where(dataset2_x_ids[:, i] != -1)[0]
        dataset2_x[conf_flows_id, i * cm_d:(i + 1) * cm_d] = sketch_now1.query_d_np(dataset2_x_ids[conf_flows_id, i]).T
    dataset2_x[:, cm_d * dataset2_x_ids.shape[1]:] = sketch_now1.query_d_np(dataset2_index_array).T

    '''dataset2 dataset2'''
    dataset2_x = np.log(1 + dataset2_x)
    dataset2_y = np.array(dataset2_flows_data_list).reshape(-1, 1)
    print(dataset2_x.shape, dataset2_y.shape)



    # 查询所有流在cm sketch中的值
    cm_flows_query_d = sketch_now2.query_d_np(test_index_array)
    cm_x = cm_flows_query_d.T
    # 流的查询值，即对每行求min:(n,1)
    flows_query_cm = np.min(cm_x, axis=1)
    cm_y = np.array(list(test_flows_data.values())).T

    # #获取cm的相对误差
    cm_num_array = flows_query_cm.T
    real_num_array = np.array(test_flows_data_list).reshape(-1,1).reshape(1,-1)
    relative_cm_error = np.abs((real_num_array - cm_num_array) / real_num_array)
    print("relative error:" + "{:.6f}".format(np.mean(relative_cm_error)))

    '''train dataset'''
    dataset_x=np.log(1+dataset_x)
    dataset_y = np.log(np.array(dataset_flows_data_list).reshape(-1, 1))

    '''train dataset2'''
    dataset2_x = np.log(1 + dataset2_x)
    dataset2_y = np.log(np.array(dataset2_flows_data_list).reshape(-1, 1))

    '''test dataset'''
    test_x=np.log(1+test_x)
    test_y = np.log(np.array(test_flows_data_list).reshape(-1,1))

    # dataset_x = preprocessing.StandardScaler().fit_transform(dataset_x)
    # test_x = preprocessing.StandardScaler().fit_transform(test_x)
    # 使用批训练方式
    dataset1=TensorDataset(torch.tensor(dataset_x,dtype=torch.float),torch.tensor(dataset_y,dtype=torch.float))
    dataset2 = TensorDataset(torch.tensor(dataset2_x, dtype=torch.float), torch.tensor(dataset2_y, dtype=torch.float))
    dataset0 = TensorDataset(torch.tensor(test_x, dtype=torch.float), torch.tensor(test_y, dtype=torch.float))

    # layers_list=[i for i in range(3,9,2)]
    layers_list = [4]
    # cell_list = [128,256]
    cell_num_list = [100]
    # lr_list=[0.01,0.001]
    lr_list=[0.01]
    drop_out_list=[0.0]
    # epoch_list=[100,500,1000]
    epoch_list = [100]
    # batch_list=[100,1000,10000]
    batch_list=[1000]
    dnn0net = DNN1(input_size=dataset_x.shape[1], deep_l=2, num=100, drop_out=0)


    for deep_len in layers_list:
        for cell_num in cell_num_list:
            for set_lr in lr_list:
                for drop_out in drop_out_list:
                    for set_epoch in epoch_list:
                        for set_batch in batch_list:
                            start_time = time.time()
                            print('#DNN层数:{},神经元数量:{},学习率:{},dropout:{},epoch:{},batch:{}#'.format(deep_len,cell_num,set_lr,drop_out,set_epoch,set_batch))
                            '''定义优化器和损失函数'''
                            dnn1net = DNN1(input_size=dataset_x.shape[1], deep_l=deep_len - 2,num=cell_num,drop_out=drop_out)
                            optim=torch.optim.Adam(DNN1.parameters(dnn1net), lr=set_lr)
                            dataloader = DataLoader(dataset1, batch_size=set_batch, shuffle=True)
                            scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.99)
                            Loss=nn.MSELoss(reduction='mean')
                            loss_list=[]

                            # 下面开始训练：
                            # 一共训练 1000次
                            for epoch in range(set_epoch):
                                loss=None
                                for batch_x,batch_y in dataloader:
                                    y_predict=dnn1net(batch_x)
                                    loss=Loss(y_predict,batch_y)
                                    optim.zero_grad()
                                    loss.backward()
                                    optim.step()
                                    # loss_list.append(loss.item())
                                # 每100次 的时候打印一次日志
                                if (epoch+1)%20==0:
                                    loss1 = None
                                    predict = dnn1net(torch.tensor(test_x, dtype=torch.float))
                                    loss1 = Loss(predict, torch.tensor(test_y, dtype=torch.float))
                                    print("step: {0} , train loss: {1} , test loss: {2}".format(epoch+1,loss.item(),loss1.item()))
                                    test_0=np.zeros(test_y.shape)
                                    if loss1==Loss(torch.tensor(test_0,dtype=torch.float),torch.tensor(test_y,dtype=torch.float)):
                                        break
                                scheduler.step()  # 学习率的更新

                            # 使用训练好的模型进行预测
                            predict=dnn1net(torch.tensor(test_x, dtype=torch.float))
                            loss1=Loss(predict,torch.tensor(test_y,dtype=torch.float))
                            print("last loss:{0}".format(loss1.item()))

                            pre_arr=predict.detach().numpy()
                            # relative_pre_error=np.exp(pre_arr-test_y)-1
                            relative_pre_error=np.round(np.exp(pre_arr))/np.exp(test_y)-1
                            print("relative pre error:"+"{:.6f}".format(np.mean(np.abs(relative_pre_error))))
                            print("max relative pre error:" + "{:.6f}".format(np.max(np.abs(relative_pre_error))))

                            end_time = time.time()
                            print('one train time:',end_time-start_time)
                            dnn0net=dnn1net

    for deep_len in layers_list:
        for cell_num in cell_num_list:
            for set_lr in lr_list:
                for drop_out in drop_out_list:
                    for set_epoch in epoch_list:
                        for set_batch in batch_list:
                            start_time = time.time()
                            print('#DNN层数:{},神经元数量:{},学习率:{},dropout:{},epoch:{},batch:{}#'.format(deep_len,cell_num,set_lr,drop_out,set_epoch,set_batch))
                            '''定义优化器和损失函数'''
                            dnn2net = DNN1(input_size=dataset_x.shape[1], deep_l=deep_len - 2,num=cell_num,drop_out=drop_out)
                            optim=torch.optim.Adam(DNN1.parameters(dnn2net), lr=set_lr)
                            dataloader = DataLoader(dataset2, batch_size=set_batch, shuffle=True)
                            scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.99)
                            Loss=nn.MSELoss(reduction='mean')
                            loss_list=[]

                            # 下面开始训练：
                            # 一共训练 1000次
                            for epoch in range(set_epoch):
                                loss=None
                                for batch_x,batch_y in dataloader:
                                    y_predict=dnn2net(batch_x)
                                    loss=Loss(y_predict,batch_y)
                                    optim.zero_grad()
                                    loss.backward()
                                    optim.step()
                                    # loss_list.append(loss.item())
                                # 每100次 的时候打印一次日志
                                if (epoch+1)%20==0:
                                    loss1 = None
                                    predict = dnn2net(torch.tensor(test_x, dtype=torch.float))
                                    loss1 = Loss(predict, torch.tensor(test_y, dtype=torch.float))
                                    print("step: {0} , train loss: {1} , test loss: {2}".format(epoch+1,loss.item(),loss1.item()))
                                    test_0=np.zeros(test_y.shape)
                                    if loss1==Loss(torch.tensor(test_0,dtype=torch.float),torch.tensor(test_y,dtype=torch.float)):
                                        break
                                scheduler.step()  # 学习率的更新

                            # 使用训练好的模型进行预测
                            predict=dnn2net(torch.tensor(test_x, dtype=torch.float))
                            loss1=Loss(predict,torch.tensor(test_y,dtype=torch.float))
                            print("last loss:{0}".format(loss1.item()))

                            pre_arr=predict.detach().numpy()
                            # relative_pre_error=np.exp(pre_arr-test_y)-1
                            relative_pre_error=np.round(np.exp(pre_arr))/np.exp(test_y)-1
                            print("relative pre error:"+"{:.6f}".format(np.mean(np.abs(relative_pre_error))))
                            print("max relative pre error:" + "{:.6f}".format(np.max(np.abs(relative_pre_error))))

                            end_time = time.time()
                            print('one train time:',end_time-start_time)
    for deep_len in layers_list:
        for cell_num in cell_num_list:
            for set_lr in lr_list:
                for drop_out in drop_out_list:
                    for set_epoch in [12]:
                        for set_batch in batch_list:
                            start_time = time.time()
                            print('#DNN层数:{},神经元数量:{},学习率:{},dropout:{},epoch:{},batch:{}#'.format(deep_len,cell_num,set_lr,drop_out,set_epoch,set_batch))
                            '''定义优化器和损失函数'''
                            # dnn1net = DNN1(input_size=dataset_x.shape[1], deep_l=deep_len - 2,num=cell_num,drop_out=drop_out)
                            dnn1net=dnn0net
                            optim=torch.optim.Adam(DNN1.parameters(dnn1net), lr=set_lr)
                            dataloader = DataLoader(dataset2, batch_size=set_batch, shuffle=True)
                            scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.99)
                            Loss=nn.MSELoss(reduction='mean')
                            loss_list=[]

                            # 下面开始训练：
                            # 一共训练 1000次
                            for epoch in range(set_epoch):
                                loss=None
                                for i in range(10):
                                    counter = 0
                                    for batch_x,batch_y in dataloader:
                                        y_predict = dnn1net(batch_x)
                                        loss = Loss(y_predict, batch_y)
                                        optim.zero_grad()
                                        loss.backward()
                                        optim.step()
                                        loss_list.append(loss.item())

                                        if counter==10:
                                            break

                                        counter = counter + 1

                                scheduler.step()  # 学习率的更新
                                # 使用训练好的模型进行预测
                                predict = dnn1net(torch.tensor(test_x, dtype=torch.float))
                                loss1 = Loss(predict, torch.tensor(test_y, dtype=torch.float))

                                pre_arr = predict.detach().numpy()
                                # relative_pre_error=np.exp(pre_arr-test_y)-1
                                relative_pre_error = np.round(np.exp(pre_arr)) / np.exp(test_y) - 1
                                print("relative pre error:" + "{:.6f}".format(np.mean(np.abs(relative_pre_error))))

                            # 使用训练好的模型进行预测
                            predict=dnn1net(torch.tensor(test_x, dtype=torch.float))
                            loss1=Loss(predict,torch.tensor(test_y,dtype=torch.float))
                            print("last loss:{0}".format(loss1.item()))

                            pre_arr=predict.detach().numpy()
                            # relative_pre_error=np.exp(pre_arr-test_y)-1
                            relative_pre_error=np.round(np.exp(pre_arr))/np.exp(test_y)-1
                            print("relative pre error:"+"{:.6f}".format(np.mean(np.abs(relative_pre_error))))
                            print("max relative pre error:" + "{:.6f}".format(np.max(np.abs(relative_pre_error))))

                            end_time = time.time()
                            print('12 epochs train and test time:',end_time-start_time)
                            # print(loss_list)